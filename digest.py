#!/usr/bin/env python3
"""
AI News Digest
--------------
Fetches the previous day's AI headlines from Google News RSS and
Hacker News, then emails a formatted digest to the recipient.

Required environment variables:
  GMAIL_USER         - Gmail address used to send (and receive) the digest
  GMAIL_APP_PASSWORD - 16-character Gmail App Password (not account password)

Optional flags:
  --dry-run          - Print digest to terminal instead of sending email
"""

import argparse
import base64
import html
import logging
import os
import re
import smtplib
import sys
import time
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from zoneinfo import ZoneInfo

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
    "transformer", "chatbot", "generative ai", "gen ai", "copilot",
    "mistral", "llama", "stable diffusion", "midjourney", "sora", "nvidia",
    "hugging face", "pytorch", "tensorflow", "agi", "robotics",
    "language model", "foundation model", "multimodal", "autonomous",
}

MAX_GOOGLE_ARTICLES = 10
MAX_HN_ARTICLES     = 10
HTTP_TIMEOUT        = 20   # seconds
MAX_RETRIES         = 3
RETRY_BACKOFF       = 2    # seconds (doubles each retry)

# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def get_yesterday_eastern() -> tuple[datetime, datetime]:
    """
    Return (start_of_yesterday_utc, end_of_yesterday_utc) where 'yesterday'
    is defined in US Eastern time — correctly handles EST/EDT transitions.
    """
    eastern = ZoneInfo("America/New_York")
    now_et  = datetime.now(tz=eastern)
    yest_et = now_et - timedelta(days=1)

    start = yest_et.replace(hour=0,  minute=0,  second=0,  microsecond=0)
    end   = yest_et.replace(hour=23, minute=59, second=59, microsecond=999999)

    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _contains_ai_keyword(text: str) -> bool:
    """Return True if text contains at least one AI-related keyword."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in AI_KEYWORDS)


def _http_get_with_retry(url: str, params: dict | None = None) -> requests.Response:
    """
    GET request with simple exponential-backoff retry (up to MAX_RETRIES).
    Raises requests.RequestException on final failure.
    """
    delay = RETRY_BACKOFF
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=HTTP_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < MAX_RETRIES:
                logging.warning(
                    "HTTP request failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, MAX_RETRIES, exc, delay,
                )
                time.sleep(delay)
                delay *= 2
            else:
                logging.error(
                    "HTTP request failed after %d attempts: %s", MAX_RETRIES, exc
                )

    raise last_exc

# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def fetch_google_news(start_utc: datetime, end_utc: datetime) -> list[dict]:
    """Fetch AI headlines from Google News RSS, filtered to yesterday."""
    logging.info("Fetching Google News RSS...")

    # Use requests (with timeout + retry) to fetch the feed, then let feedparser parse it
    try:
        resp = _http_get_with_retry(GOOGLE_NEWS_URL)
        feed = feedparser.parse(resp.text)
    except Exception as exc:
        logging.error("Google News RSS fetch failed: %s", exc)
        return []

    if feed.bozo and not feed.entries:
        logging.error("Google News RSS parse error: %s", feed.bozo_exception)
        return []

    articles = []
    seen_tokens: list[set] = []  # for within-source deduplication

    for entry in feed.entries:
        if not getattr(entry, "published_parsed", None):
            continue

        pub_dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
        if not (start_utc <= pub_dt <= end_utc):
            continue

        title     = html.unescape((entry.get("title") or "Untitled").strip())
        link      = entry.get("link") or "#"

        if link == "#":
            logging.warning("Google News article missing link: %s", title)

        # Strip HTML tags from summary
        raw_desc   = getattr(entry, "summary", "") or ""
        clean_desc = re.sub(r"<[^>]+>", "", raw_desc).strip()
        clean_desc = html.unescape(clean_desc)

        # Keep first 1-2 sentences, cap at 300 chars
        sentences = re.split(r"(?<=[.!?])\s+", clean_desc)
        excerpt   = " ".join(sentences[:2]).strip()
        if len(excerpt) > 300:
            excerpt = excerpt[:297] + "..."

        # Apply AI keyword filter — title OR description must match
        if not _contains_ai_keyword(title) and not _contains_ai_keyword(excerpt):
            continue

        # Within-source deduplication
        entry_tokens = _tokens(title)
        is_dup = False
        for seen in seen_tokens:
            if seen and entry_tokens:
                overlap = len(entry_tokens & seen) / max(len(entry_tokens), len(seen))
                if overlap > 0.5:
                    is_dup = True
                    break
        if is_dup:
            continue

        seen_tokens.append(entry_tokens)
        articles.append({
            "title":         title,
            "link":          link,
            "description":   excerpt,
            "published_utc": pub_dt,
        })

    logging.info("Google News: %d articles after filtering", len(articles))
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
        resp = _http_get_with_retry(HN_SEARCH_URL, params=params)
        data = resp.json()
    except Exception as exc:
        logging.error("Hacker News API fetch failed: %s", exc)
        return []

    articles = []
    seen_tokens: list[set] = []

    for hit in data.get("hits", []):
        title = (hit.get("title") or "").strip()
        if not title:
            continue

        # Keep only AI-related stories
        if not _contains_ai_keyword(title):
            continue

        # Within-source deduplication
        entry_tokens = _tokens(title)
        is_dup = False
        for seen in seen_tokens:
            if seen and entry_tokens:
                overlap = len(entry_tokens & seen) / max(len(entry_tokens), len(seen))
                if overlap > 0.5:
                    is_dup = True
                    break
        if is_dup:
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

        seen_tokens.append(entry_tokens)
        articles.append({
            "title":         title,
            "link":          url,
            "hn_link":       hn_url,
            "description":   f"{points} points \u00b7 {comments} comments on Hacker News",
            "published_utc": pub_dt,
            "points":        points,
        })

    articles.sort(key=lambda x: x["points"], reverse=True)
    logging.info("Hacker News: %d AI articles after filtering", len(articles))
    return articles[:MAX_HN_ARTICLES]

# ---------------------------------------------------------------------------
# Deduplication (cross-source)
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "of", "in", "on", "for", "is", "to", "and",
    "or", "at", "with", "by", "from", "that", "this", "its", "it",
    "as", "be", "are", "has", "have", "had", "was", "were", "will",
    "new", "says", "say", "just", "use", "using",
}


def _tokens(title: str) -> set[str]:
    words = re.findall(r"\w+", title.lower())
    return {w for w in words if w not in _STOP_WORDS and len(w) > 2}


def deduplicate(
    google_articles: list[dict],
    hn_articles: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Remove HN articles whose titles overlap > 50% with any Google News title.
    (Within-source deduplication is handled inside each fetcher.)
    """
    google_token_sets = [_tokens(a["title"]) for a in google_articles]

    unique_hn = []
    for hn_art in hn_articles:
        hn_toks   = _tokens(hn_art["title"])
        duplicate = False
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
# Plain-text email body (built from data, not by stripping HTML)
# ---------------------------------------------------------------------------

def build_plain_text(google_articles: list[dict], hn_articles: list[dict], date_str: str) -> str:
    lines = [
        f"AI NEWS DIGEST — {date_str}",
        "=" * 50,
        "",
    ]

    lines.append("GOOGLE NEWS — AI HEADLINES")
    lines.append("-" * 30)
    if google_articles:
        for i, a in enumerate(google_articles, 1):
            lines.append(f"{i}. {a['title']}")
            lines.append(f"   {a['link']}")
            if a.get("description"):
                lines.append(f"   {a['description']}")
            lines.append("")
    else:
        lines.append("No Google News AI articles found for yesterday.")
        lines.append("")

    lines.append("")
    lines.append("HACKER NEWS — TOP AI STORIES")
    lines.append("-" * 30)
    if hn_articles:
        for i, a in enumerate(hn_articles, 1):
            lines.append(f"{i}. {a['title']}")
            lines.append(f"   {a['link']}")
            lines.append(f"   {a['description']}")
            if a.get("hn_link") and a["hn_link"] != a["link"]:
                lines.append(f"   HN thread: {a['hn_link']}")
            lines.append("")
    else:
        lines.append("No Hacker News AI stories found for yesterday.")
        lines.append("")

    lines.append("")
    lines.append("Automated daily digest · Delivered at 5am ET")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# HTML email formatting
# ---------------------------------------------------------------------------

_COLOR_HEADER = "#1a1a2e"
_COLOR_ACCENT = "#0066cc"
_COLOR_BG     = "#f0f2f5"
_COLOR_CARD   = "#ffffff"
_COLOR_BORDER = "#e4e6ea"
_COLOR_MUTED  = "#666666"

# ---------------------------------------------------------------------------
# Old English / Blackletter numeral images (20x20px, base64 data URI SVGs)
#
# Gmail strips inline <svg> tags entirely, so we embed each numeral as an
# <img src="data:image/svg+xml;base64,..."> — Gmail allows data URIs in img src.
# Each SVG is built as a string then base64-encoded at runtime.
# Style: gold-ringed dark badge with Old English blackletter serif digit.
# ---------------------------------------------------------------------------

def _build_numeral_svg(n: int) -> str:
    """
    Build a raw SVG string for a given numeral (1-10).
    Uses large viewBox (0 0 100 100) for crisp rendering, scaled to 20x20px display.
    Style: dark parchment circle badge with Old English blackletter digit in gold/cream.
    The SVG is then base64-encoded and embedded as a data URI in an <img> tag,
    which Gmail fully supports (unlike inline <svg> which Gmail strips).
    """
    # Blackletter / Old English Unicode numerals from the Mathematical Fraktur block.
    # These render in a genuine blackletter style on all major platforms.
    fraktur = {
        1: "𝟏", 2: "𝟐", 3: "𝟑", 4: "𝟒", 5: "𝟓",
        6: "𝟔", 7: "𝟕", 8: "𝟖", 9: "𝟗", 10: "𝟏𝟎",
    }
    # Fallback plain digit for "10" which needs two characters
    label = fraktur.get(n, str(n))
    font_size = "52" if n < 10 else "38"
    x_pos = "50" if n < 10 else "50"

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 100 100">
  <defs>
    <radialGradient id="bg" cx="40%" cy="35%" r="60%">
      <stop offset="0%" stop-color="#3a2a0a"/>
      <stop offset="100%" stop-color="#1a0f00"/>
    </radialGradient>
    <filter id="shadow">
      <feDropShadow dx="1.5" dy="2" stdDeviation="2" flood-color="#000" flood-opacity="0.6"/>
    </filter>
    <filter id="txtshadow">
      <feDropShadow dx="1" dy="1.5" stdDeviation="1" flood-color="#000" flood-opacity="0.8"/>
    </filter>
  </defs>
  <!-- Outer ring -->
  <circle cx="50" cy="50" r="48" fill="#8b6914" filter="url(#shadow)"/>
  <!-- Inner ring -->
  <circle cx="50" cy="50" r="44" fill="#c8961e"/>
  <!-- Main badge face -->
  <circle cx="50" cy="50" r="40" fill="url(#bg)"/>
  <!-- Inner decorative ring -->
  <circle cx="50" cy="50" r="36" fill="none" stroke="#8b6914" stroke-width="1.5" opacity="0.8"/>
  <!-- Numeral -->
  <text
    x="{x_pos}" y="65"
    font-family="'Palatino Linotype','Book Antiqua',Palatino,'Times New Roman',Georgia,serif"
    font-size="{font_size}"
    font-weight="900"
    font-style="italic"
    fill="#f0d080"
    text-anchor="middle"
    filter="url(#txtshadow)"
    letter-spacing="-2"
  >{label}</text>
  <!-- Highlight glint top-left -->
  <ellipse cx="35" cy="32" rx="10" ry="6" fill="white" opacity="0.08" transform="rotate(-30 35 32)"/>
</svg>'''
    return svg


def _old_english_numeral_svg(n: int) -> str:
    """
    Return an <img> tag with a base64-encoded SVG data URI.
    Gmail fully supports <img src="data:image/svg+xml;base64,..."> tags
    but strips all inline <svg> elements — this approach works reliably.
    """
    if not (1 <= n <= 10):
        return (
            f'<span style="color:#aaaaaa;font-size:11px;font-weight:700;'
            f'margin-right:4px;">{n}.</span>'
        )
    raw_svg   = _build_numeral_svg(n)
    b64       = base64.b64encode(raw_svg.encode("utf-8")).decode("ascii")
    data_uri  = f"data:image/svg+xml;base64,{b64}"
    return (
        f'<img src="{data_uri}" width="20" height="20" alt="{n}" '
        f'style="display:inline-block;vertical-align:middle;margin-right:7px;'
        f'border:0;flex-shrink:0;" />'
    )


def _article_row(article: dict, index: int, show_hn_link: bool = False) -> str:
    title   = html.escape(article["title"])
    link    = article["link"]
    desc    = html.escape(article.get("description", ""))
    numeral = _old_english_numeral_svg(index)

    hn_link_html = ""
    if show_hn_link and article.get("hn_link") and article["hn_link"] != article["link"]:
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
        f'<tr><td style="padding:12px 0;border-bottom:1px solid {_COLOR_BORDER};vertical-align:middle;">'
        f'{numeral}'
        f'<a href="{link}" style="color:{_COLOR_ACCENT};font-weight:600;'
        f'font-size:15px;text-decoration:none;line-height:1.4;">{title}</a>'
        f'{hn_link_html}'
        f'{desc_html}'
        f'</td></tr>'
    )


def build_html_email(
    google_articles: list[dict],
    hn_articles: list[dict],
    date_str: str,
) -> str:
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
  <title>AI News Digest &mdash; {date_str}</title>
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
          <p style="margin:0;color:#8aa4cc;font-size:14px;">{date_str}</p>
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

def send_email(subject: str, html_body: str, plain_body: str) -> bool:
    """Send HTML + plain-text email via Gmail SMTP (STARTTLS). Returns True on success."""
    msg            = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"AI Digest <{SENDER_EMAIL}>"
    msg["To"]      = RECIPIENT_EMAIL

    # Plain-text part first (lower priority — clients prefer HTML if available)
    msg.attach(MIMEText(plain_body, "plain", "utf-8"))
    msg.attach(MIMEText(html_body,  "html",  "utf-8"))

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
    parser = argparse.ArgumentParser(description="AI News Digest emailer")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print digest to terminal instead of sending email",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )

    # Validate credentials immediately (fail-fast before any network calls)
    if not args.dry_run:
        if not SENDER_EMAIL or not GMAIL_APP_PASSWORD:
            logging.error(
                "GMAIL_USER and GMAIL_APP_PASSWORD environment variables must be set."
            )
            sys.exit(1)

    start_utc, end_utc = get_yesterday_eastern()
    logging.info("Date window: %s  →  %s", start_utc, end_utc)

    # Fetch both sources independently
    google_articles = fetch_google_news(start_utc, end_utc)
    hn_articles     = fetch_hacker_news(start_utc, end_utc)

    # Cross-source deduplication
    google_articles, hn_articles = deduplicate(google_articles, hn_articles)

    total    = len(google_articles) + len(hn_articles)
    date_str = datetime.now(ZoneInfo("America/New_York")).strftime("%A, %B %-d, %Y")
    subject  = f"AI News Digest \u2014 {datetime.now(ZoneInfo('America/New_York')).strftime('%b %-d')} ({total} {'story' if total == 1 else 'stories'})"

    logging.info("Total unique articles: %d", total)

    html_body  = build_html_email(google_articles, hn_articles, date_str)
    plain_body = build_plain_text(google_articles, hn_articles, date_str)

    if args.dry_run:
        print("\n" + "=" * 60)
        print(f"DRY RUN — Subject: {subject}")
        print("=" * 60)
        print(plain_body)
        print("=" * 60)
        print("(HTML email not sent — dry-run mode)")
        return

    success = send_email(subject, html_body, plain_body)
    if not success:
        logging.error("Email delivery failed.")
        sys.exit(1)

    logging.info("Digest complete.")


if __name__ == "__main__":
    main()
