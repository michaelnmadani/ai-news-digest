#!/usr/bin/env python3
"""
AI News Digest
--------------
Fetches the previous day's AI headlines from Google News RSS and
Hacker News, then emails a formatted digest to the recipient.

Required environment variables:
  GMAIL_USER         - Gmail address used to send (and receive) the digest
  GMAIL_APP_PASSWORD - 16-character Gmail App Password (not account password)
"""

import html
import logging
import os
import re
import smtplib
import sys
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import feedparser
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RECIPIENT_EMAIL    = "michael.n.madani@gmail.com"
SENDER_EMAIL       = os.environ.get("GMAIL_USER", "")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD", "")
SMTP_HOST          = "smtp.gmail.com"
SMTP_PORT          = 587

GOOGLE_NEWS_URL = (
    "https://news.google.com/rss/search"
    "?q=artificial+intelligence+OR+%22machine+learning%22+OR+LLM"
    "+OR+OpenAI+OR+Anthropic+OR+%22generative+AI%22"
    "&hl=en-US&gl=US&ceid=US:en"
)
HN_SEARCH_URL = "https://hn.algolia.com/api/v1/search"

AI_KEYWORDS = {
    "ai", "artificial intelligence", "machine learning", "deep learning",
    "llm", "large language model", "gpt", "claude", "gemini", "openai",
    "anthropic", "neural network", "nlp", "natural language", "diffusion",
    "transformer", "chatbot", "generative ai", "gen ai", "ml", "copilot",
    "mistral", "llama", "stable diffusion", "midjourney", "sora", "nvidia",
    "hugging face", "pytorch", "tensorflow", "agi", "robotics",
}

MAX_GOOGLE_ARTICLES = 15
MAX_HN_ARTICLES     = 15

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def get_yesterday_eastern() -> tuple[datetime, datetime]:
    """
    Return (start_of_yesterday_utc, end_of_yesterday_utc) where
    'yesterday' is defined in US Eastern time (fixed UTC-5 offset).
    """
    eastern = timezone(timedelta(hours=-5))
    now_et  = datetime.now(tz=eastern)
    yest_et = now_et - timedelta(days=1)

    start = yest_et.replace(hour=0,  minute=0,  second=0,  microsecond=0)
    end   = yest_et.replace(hour=23, minute=59, second=59, microsecond=999999)

    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)

# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_google_news(start_utc: datetime, end_utc: datetime) -> list[dict]:
    """Fetch AI headlines from Google News RSS, filtered to yesterday."""
    logging.info("Fetching Google News RSS...")
    try:
        feed = feedparser.parse(GOOGLE_NEWS_URL)
    except Exception as exc:
        logging.error("Google News RSS fetch failed: %s", exc)
        return []

    articles = []
    for entry in feed.entries:
        if not getattr(entry, "published_parsed", None):
            continue

        pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        if not (start_utc <= pub_dt <= end_utc):
            continue

        # Strip HTML tags from the summary Google News provides
        raw_desc  = getattr(entry, "summary", "") or ""
        clean_desc = re.sub(r"<[^>]+>", "", raw_desc).strip()

        # Keep first 1-2 sentences, cap at 300 chars
        sentences = re.split(r"(?<=[.!?])\s+", clean_desc)
        excerpt   = " ".join(sentences[:2]).strip()
        if len(excerpt) > 300:
            excerpt = excerpt[:297] + "..."

        articles.append({
            "title":         html.unescape(entry.get("title", "Untitled").strip()),
            "link":          entry.get("link", "#"),
            "description":   excerpt,
            "published_utc": pub_dt,
        })

    logging.info("Google News: %d articles in date window", len(articles))
    return articles[:MAX_GOOGLE_ARTICLES]


def fetch_hacker_news(start_utc: datetime, end_utc: datetime) -> list[dict]:
    """Fetch AI-related Hacker News stories from yesterday via Algolia API."""
    logging.info("Fetching Hacker News via Algolia API...")

    params = {
        "tags":              "story",
        "numericFilters":    (
            f"created_at_i>{int(start_utc.timestamp())},"
            f"created_at_i<{int(end_utc.timestamp())},"
            "points>9"
        ),
        "hitsPerPage":       100,
        "attributesToRetrieve": "title,url,points,num_comments,created_at,objectID",
    }

    try:
        resp = requests.get(HN_SEARCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logging.error("Hacker News API fetch failed: %s", exc)
        return []

    articles = []
    for hit in data.get("hits", []):
        title = (hit.get("title") or "").strip()
        if not title:
            continue

        # Keep only AI-related stories
        title_lower = title.lower()
        if not any(kw in title_lower for kw in AI_KEYWORDS):
            continue

        hn_url = f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}"
        url    = hit.get("url") or hn_url

        try:
            pub_dt = datetime.fromisoformat(
                hit.get("created_at", "").replace("Z", "+00:00")
            )
        except (ValueError, AttributeError):
            continue

        points   = hit.get("points", 0) or 0
        comments = hit.get("num_comments", 0) or 0

        articles.append({
            "title":         title,
            "link":          url,
            "hn_link":       hn_url,
            "description":   f"{points} points \u00b7 {comments} comments on Hacker News",
            "published_utc": pub_dt,
            "points":        points,
        })

    articles.sort(key=lambda x: x["points"], reverse=True)
    logging.info("Hacker News: %d AI articles found", len(articles))
    return articles[:MAX_HN_ARTICLES]

# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "of", "in", "on", "for", "is", "to", "and",
    "or", "at", "with", "by", "from", "that", "this", "its", "it",
    "as", "be", "are", "has", "have", "had", "was", "were", "will",
}

def _tokens(title: str) -> set[str]:
    words = re.findall(r"\w+", title.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def deduplicate(
    google_articles: list[dict],
    hn_articles: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Remove HN articles whose titles overlap > 50% with a Google News title.
    Operates on HN list only; Google list is returned unchanged.
    """
    google_token_sets = [_tokens(a["title"]) for a in google_articles]

    unique_hn = []
    for hn_art in hn_articles:
        hn_toks    = _tokens(hn_art["title"])
        duplicate  = False
        for g_toks in google_token_sets:
            if not g_toks or not hn_toks:
                continue
            overlap = len(hn_toks & g_toks) / max(len(hn_toks), len(g_toks))
            if overlap > 0.5:
                duplicate = True
                break
        if not duplicate:
            unique_hn.append(hn_art)

    return google_articles, unique_hn

# ---------------------------------------------------------------------------
# Email formatting
# ---------------------------------------------------------------------------

_COLOR_HEADER  = "#1a1a2e"
_COLOR_ACCENT  = "#0066cc"
_COLOR_BG      = "#f0f2f5"
_COLOR_CARD    = "#ffffff"
_COLOR_BORDER  = "#e4e6ea"
_COLOR_MUTED   = "#666666"


def _article_row(article: dict, index: int, show_hn_link: bool = False) -> str:
    title = html.escape(article["title"])
    link  = article["link"]
    desc  = html.escape(article.get("description", ""))

    hn_link_html = ""
    if show_hn_link and "hn_link" in article:
        hn_link_html = (
            f' &nbsp;<a href="{article["hn_link"]}" '
            f'style="color:{_COLOR_MUTED};font-size:12px;text-decoration:none;">'
            f'[HN thread]</a>'
        )

    desc_html = (
        f'<br><span style="color:#555555;font-size:13px;line-height:1.5;'
        f'display:block;margin-top:3px;margin-left:22px;">{desc}</span>'
        if desc else ""
    )

    return (
        f'<tr><td style="padding:12px 0;border-bottom:1px solid {_COLOR_BORDER};">'
        f'<span style="color:#aaaaaa;font-size:12px;font-weight:700;margin-right:6px;">{index}.</span>'
        f'<a href="{link}" style="color:{_COLOR_ACCENT};font-weight:600;'
        f'font-size:15px;text-decoration:none;line-height:1.4;">{title}</a>'
        f'{hn_link_html}'
        f'{desc_html}'
        f'</td></tr>'
    )


def build_html_email(google_articles: list[dict], hn_articles: list[dict]) -> str:
    today_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y").replace(" 0", " ")

    google_rows = (
        "".join(_article_row(a, i + 1) for i, a in enumerate(google_articles))
        if google_articles
        else (
            f'<tr><td style="color:{_COLOR_MUTED};padding:14px 0;font-style:italic;">'
            f'No Google News AI articles found for yesterday.</td></tr>'
        )
    )

    hn_rows = (
        "".join(_article_row(a, i + 1, show_hn_link=True) for i, a in enumerate(hn_articles))
        if hn_articles
        else (
            f'<tr><td style="color:{_COLOR_MUTED};padding:14px 0;font-style:italic;">'
            f'No Hacker News AI stories found for yesterday.</td></tr>'
        )
    )

    g_count = len(google_articles)
    h_count = len(hn_articles)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1.0">
  <title>AI News Digest &mdash; {today_str}</title>
</head>
<body style="margin:0;padding:0;background:{_COLOR_BG};font-family:Arial,Helvetica,sans-serif;">
<table width="100%" cellpadding="0" cellspacing="0" border="0"
       style="background:{_COLOR_BG};">
  <tr><td align="center" style="padding:28px 12px;">
    <table width="640" cellpadding="0" cellspacing="0" border="0"
           style="max-width:640px;width:100%;">

      <!-- HEADER -->
      <tr>
        <td style="background:{_COLOR_HEADER};padding:28px 32px;
            border-radius:10px 10px 0 0;">
          <p style="margin:0;color:#7090c8;font-size:12px;
              letter-spacing:2px;text-transform:uppercase;">Daily Briefing</p>
          <h1 style="margin:8px 0 4px;color:#ffffff;font-size:26px;
              font-weight:700;letter-spacing:-0.5px;">AI News Digest</h1>
          <p style="margin:0;color:#8aa4cc;font-size:14px;">{today_str}</p>
        </td>
      </tr>

      <!-- SUMMARY BAR -->
      <tr>
        <td style="background:#0d2d52;padding:10px 32px;">
          <p style="margin:0;color:#7aaee8;font-size:13px;">
            {g_count} headline{'s' if g_count != 1 else ''} from Google News
            &nbsp;&middot;&nbsp;
            {h_count} stor{'ies' if h_count != 1 else 'y'} from Hacker News
          </p>
        </td>
      </tr>

      <!-- GOOGLE NEWS -->
      <tr>
        <td style="background:{_COLOR_CARD};padding:26px 32px 18px;">
          <h2 style="margin:0 0 16px;font-size:17px;color:{_COLOR_HEADER};
              border-left:4px solid #cc2200;padding-left:12px;">
            Google News &mdash; AI Headlines
          </h2>
          <table width="100%" cellpadding="0" cellspacing="0" border="0">
            {google_rows}
          </table>
        </td>
      </tr>

      <!-- DIVIDER -->
      <tr>
        <td style="background:{_COLOR_CARD};padding:0 32px;">
          <hr style="border:none;border-top:2px solid {_COLOR_BORDER};margin:0;">
        </td>
      </tr>

      <!-- HACKER NEWS -->
      <tr>
        <td style="background:{_COLOR_CARD};padding:26px 32px 18px;
            border-radius:0 0 10px 10px;">
          <h2 style="margin:0 0 16px;font-size:17px;color:{_COLOR_HEADER};
              border-left:4px solid #ff6600;padding-left:12px;">
            Hacker News &mdash; Top AI Stories
          </h2>
          <table width="100%" cellpadding="0" cellspacing="0" border="0">
            {hn_rows}
          </table>
        </td>
      </tr>

      <!-- FOOTER -->
      <tr>
        <td style="padding:20px 32px;text-align:center;">
          <p style="margin:0;color:#aaaaaa;font-size:12px;line-height:1.6;">
            Automated daily digest &middot; Delivered at 5&nbsp;am&nbsp;ET<br>
            Powered by Google News RSS &amp; Hacker News API
          </p>
        </td>
      </tr>

    </table>
  </td></tr>
</table>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Email sender
# ---------------------------------------------------------------------------

def send_email(subject: str, html_body: str) -> bool:
    """Send HTML email via Gmail SMTP (STARTTLS). Returns True on success."""
    if not SENDER_EMAIL or not GMAIL_APP_PASSWORD:
        logging.error(
            "Missing GMAIL_USER or GMAIL_APP_PASSWORD environment variable."
        )
        return False

    msg             = MIMEMultipart("alternative")
    msg["Subject"]  = subject
    msg["From"]     = f"AI Digest <{SENDER_EMAIL}>"
    msg["To"]       = RECIPIENT_EMAIL

    plain = re.sub(r"<[^>]+>", "", html_body)
    plain = re.sub(r"\n{3,}", "\n\n", plain).strip()

    msg.attach(MIMEText(plain,     "plain", "utf-8"))
    msg.attach(MIMEText(html_body, "html",  "utf-8"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SENDER_EMAIL, GMAIL_APP_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, msg.as_string())
        logging.info("Email sent to %s", RECIPIENT_EMAIL)
        return True
    except smtplib.SMTPAuthenticationError:
        logging.error(
            "SMTP authentication failed. "
            "Verify GMAIL_USER and GMAIL_APP_PASSWORD secrets."
        )
        return False
    except smtplib.SMTPException as exc:
        logging.error("SMTP error: %s", exc)
        return False
    except Exception as exc:
        logging.error("Unexpected error sending email: %s", exc)
        return False

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    start_utc, end_utc = get_yesterday_eastern()
    logging.info("Date window: %s to %s", start_utc, end_utc)

    # Fetch sources independently — one failure won't block the other
    google_articles = fetch_google_news(start_utc, end_utc)
    hn_articles     = fetch_hacker_news(start_utc, end_utc)

    google_articles, hn_articles = deduplicate(google_articles, hn_articles)

    total     = len(google_articles) + len(hn_articles)
    date_str  = datetime.now(timezone.utc).strftime("%b %d").lstrip("0")
    subject   = f"AI News Digest \u2014 {date_str} ({total} {'story' if total == 1 else 'stories'})"

    logging.info("Total unique articles: %d", total)

    html_body = build_html_email(google_articles, hn_articles)
    success   = send_email(subject, html_body)

    if not success:
        logging.error("Email delivery failed.")
        sys.exit(1)

    logging.info("Digest complete.")


if __name__ == "__main__":
    main()
