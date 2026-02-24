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

def get_yesterday_sydney() -> tuple[datetime, datetime]:
    """
    Return (start_of_yesterday_utc, end_of_yesterday_utc) where 'yesterday'
    is defined in Sydney time — correctly handles AEST/AEDT transitions.
    """
    sydney  = ZoneInfo("Australia/Sydney")
    now_syd = datetime.now(tz=sydney)
    yest    = now_syd - timedelta(days=1)

    start = yest.replace(hour=0,  minute=0,  second=0,  microsecond=0)
    end   = yest.replace(hour=23, minute=59, second=59, microsecond=999999)

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
            lines.append("")
    else:
        lines.append("No Hacker News AI stories found for yesterday.")
        lines.append("")

    lines.append("")
    lines.append("Automated daily digest · Delivered at 5am Sydney time")

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

def _article_row(article: dict, index: int, show_hn_link: bool = False) -> str:
    title = html.escape(article["title"])
    link  = article["link"]
    desc  = html.escape(article.get("description", ""))

    hn_link_html = ""

    desc_html = (
        f'<span style="color:#555555;font-size:10px;line-height:1.5;'
        f'display:block;margin-top:4px;">{desc}</span>'
        if desc else ""
    )

    return (
        f'<tr><td style="padding:12px 0;border-bottom:1px solid {_COLOR_BORDER};">'
        f'<table width="100%" cellpadding="0" cellspacing="0" border="0">'
        f'<tr>'
        # Number column — width% is responsive (shrinks on mobile), min-width keeps
        # it readable on narrow screens. 8% of 640px ≈ 51px; floor is 36px on mobile.
        f'<td width="8%" style="vertical-align:top;padding-top:2px;'
        f'min-width:36px;max-width:60px;white-space:nowrap;">'
        f'<span style="font-size:30px;font-weight:900;color:#111111;'
        f'font-family:Georgia,\'Times New Roman\',serif;line-height:1;">'
        f'{index}</span>'
        f'</td>'
        # Content column — takes all remaining space, shrinks naturally
        f'<td style="vertical-align:top;padding-left:8px;width:92%;">'
        f'<a href="{link}" style="color:{_COLOR_ACCENT};font-weight:600;'
        f'font-size:13px;text-decoration:none;line-height:1.4;display:block;">'
        f'{title}</a>'
        f'{hn_link_html}'
        f'{desc_html}'
        f'</td>'
        f'</tr>'
        f'</table>'
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
            Automated daily digest &middot; Delivered at 5&nbsp;am&nbsp;Sydney&nbsp;time<br>
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
# Webpage builder
# ---------------------------------------------------------------------------

def build_webpage(
    google_articles: list[dict],
    hn_articles: list[dict],
    date_str: str,
) -> str:
    """
    Build a full HTML webpage matching the email digest design.
    Unlike the email (inline CSS only), the webpage can use a proper
    <style> block, Google Fonts, CSS variables, and media queries.
    """
    updated_at = datetime.now(ZoneInfo("Australia/Sydney")).strftime(
        "%A, %-d %B %Y · %-I:%M %p Sydney time"
    )
    g_count = len(google_articles)
    h_count = len(hn_articles)

    def article_card(article: dict, index: int) -> str:
        title = html.escape(article["title"])
        link  = article["link"]
        desc  = html.escape(article.get("description", ""))
        desc_html = (
            f'<p class="desc">{desc}</p>' if desc else ""
        )
        return (
            f'<div class="article">'
            f'<div class="num">{index}</div>'
            f'<div class="content">'
            f'<a href="{link}" target="_blank" rel="noopener noreferrer" class="title">{title}</a>'
            f'{desc_html}'
            f'</div>'
            f'</div>'
        )

    google_cards = (
        "".join(article_card(a, i + 1) for i, a in enumerate(google_articles))
        if google_articles
        else '<p class="empty">No Google News AI articles found for yesterday.</p>'
    )
    hn_cards = (
        "".join(article_card(a, i + 1) for i, a in enumerate(hn_articles))
        if hn_articles
        else '<p class="empty">No Hacker News AI stories found for yesterday.</p>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>AI News Digest &mdash; {date_str}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    :root {{
      --color-bg:      #f0f2f5;
      --color-header:  #1a1a2e;
      --color-card:    #ffffff;
      --color-accent:  #0066cc;
      --color-border:  #e4e6ea;
      --color-muted:   #666666;
      --color-num:     #111111;
      --color-desc:    #555555;
      --color-google:  #cc2200;
      --color-hn:      #ff6600;
      --color-gold:    #c8961e;
    }}

    body {{
      background: var(--color-bg);
      font-family: 'Inter', system-ui, sans-serif;
      color: #222;
      min-height: 100vh;
    }}

    /* ── Header ── */
    header {{
      background: var(--color-header);
      padding: 28px 24px 22px;
      position: sticky;
      top: 0;
      z-index: 10;
      box-shadow: 0 2px 12px rgba(0,0,0,0.4);
    }}
    .header-inner {{
      max-width: 800px;
      margin: 0 auto;
    }}
    .header-label {{
      font-size: 11px;
      letter-spacing: 3px;
      text-transform: uppercase;
      color: #7090c8;
      font-family: 'Inter', sans-serif;
      margin-bottom: 6px;
    }}
    header h1 {{
      font-family: 'Playfair Display', Georgia, serif;
      font-size: 32px;
      font-weight: 900;
      color: #ffffff;
      letter-spacing: -0.5px;
      line-height: 1.1;
    }}
    header .date {{
      margin-top: 6px;
      color: #8aa4cc;
      font-size: 14px;
    }}

    /* ── Summary bar ── */
    .summary-bar {{
      background: #0d2d52;
      padding: 9px 24px;
    }}
    .summary-bar p {{
      max-width: 800px;
      margin: 0 auto;
      color: #7aaee8;
      font-size: 13px;
    }}

    /* ── Main content ── */
    main {{
      max-width: 800px;
      margin: 28px auto;
      padding: 0 16px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }}

    /* ── Section card ── */
    .section {{
      background: var(--color-card);
      border-radius: 10px;
      overflow: hidden;
      box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }}
    .section-header {{
      padding: 18px 24px 14px;
      border-bottom: 2px solid var(--color-border);
    }}
    .section-header h2 {{
      font-family: 'Playfair Display', Georgia, serif;
      font-size: 18px;
      font-weight: 700;
      color: var(--color-header);
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .section-header h2::before {{
      content: '';
      display: inline-block;
      width: 4px;
      height: 20px;
      border-radius: 2px;
      flex-shrink: 0;
    }}
    .section--google .section-header h2::before {{ background: var(--color-google); }}
    .section--hn     .section-header h2::before {{ background: var(--color-hn); }}

    /* ── Article row ── */
    .article {{
      display: flex;
      align-items: flex-start;
      gap: 0;
      padding: 14px 24px;
      border-bottom: 1px solid var(--color-border);
    }}
    .article:last-child {{ border-bottom: none; }}

    .num {{
      font-family: Georgia, 'Times New Roman', serif;
      font-size: 28px;
      font-weight: 900;
      color: var(--color-num);
      line-height: 1;
      min-width: 5%;
      max-width: 52px;
      padding-top: 1px;
      flex-shrink: 0;
    }}

    .content {{
      flex: 1;
      min-width: 0;
      padding-left: 10px;
    }}

    a.title {{
      display: block;
      font-size: 14px;
      font-weight: 600;
      color: var(--color-accent);
      text-decoration: none;
      line-height: 1.4;
    }}
    a.title:hover {{ text-decoration: underline; }}

    .desc {{
      margin-top: 5px;
      font-size: 12px;
      color: var(--color-desc);
      line-height: 1.5;
    }}

    .empty {{
      padding: 16px 24px;
      color: var(--color-muted);
      font-style: italic;
      font-size: 14px;
    }}

    /* ── Footer ── */
    footer {{
      text-align: center;
      padding: 24px 16px 40px;
      color: #999;
      font-size: 12px;
      line-height: 1.7;
    }}
    footer a {{ color: #999; }}

    /* ── Responsive ── */
    @media (max-width: 480px) {{
      header h1    {{ font-size: 24px; }}
      .num         {{ font-size: 22px; min-width: 32px; }}
      a.title      {{ font-size: 13px; }}
      .article     {{ padding: 12px 16px; }}
      .section-header {{ padding: 14px 16px 12px; }}
    }}
  </style>
</head>
<body>

  <header>
    <div class="header-inner">
      <p class="header-label">Daily Briefing</p>
      <h1>AI News Digest</h1>
      <p class="date">{date_str}</p>
    </div>
  </header>

  <div class="summary-bar">
    <p>
      {g_count} headline{'s' if g_count != 1 else ''} from Google News
      &nbsp;&middot;&nbsp;
      {h_count} stor{'ies' if h_count != 1 else 'y'} from Hacker News
    </p>
  </div>

  <main>

    <section class="section section--google">
      <div class="section-header">
        <h2>Google News &mdash; AI Headlines</h2>
      </div>
      {google_cards}
    </section>

    <section class="section section--hn">
      <div class="section-header">
        <h2>Hacker News &mdash; Top AI Stories</h2>
      </div>
      {hn_cards}
    </section>

  </main>

  <footer>
    Last updated: {updated_at}<br>
    Powered by Google News RSS &amp; Hacker News API
  </footer>

</body>
</html>"""


def save_webpage(html: str, path: str = "public/index.html") -> None:
    """Write the webpage HTML to disk, creating the directory if needed."""
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logging.info("Webpage written to %s", path)


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

    start_utc, end_utc = get_yesterday_sydney()
    logging.info("Date window: %s  →  %s", start_utc, end_utc)

    # Fetch both sources independently
    google_articles = fetch_google_news(start_utc, end_utc)
    hn_articles     = fetch_hacker_news(start_utc, end_utc)

    # Cross-source deduplication
    google_articles, hn_articles = deduplicate(google_articles, hn_articles)

    total    = len(google_articles) + len(hn_articles)
    date_str = datetime.now(ZoneInfo("Australia/Sydney")).strftime("%A, %B %-d, %Y")
    subject  = f"AI News Digest \u2014 {datetime.now(ZoneInfo('Australia/Sydney')).strftime('%b %-d')} ({total} {'story' if total == 1 else 'stories'})"

    logging.info("Total unique articles: %d", total)

    html_body  = build_html_email(google_articles, hn_articles, date_str)
    plain_body = build_plain_text(google_articles, hn_articles, date_str)

    # Always generate and save the webpage (email and webpage are independent)
    webpage = build_webpage(google_articles, hn_articles, date_str)
    save_webpage(webpage)

    if args.dry_run:
        print("\n" + "=" * 60)
        print(f"DRY RUN — Subject: {subject}")
        print("=" * 60)
        print(plain_body)
        print("=" * 60)
        print("(HTML email not sent — dry-run mode)")
        print(f"(Webpage written to public/index.html)")
        return

    success = send_email(subject, html_body, plain_body)
    if not success:
        logging.error("Email delivery failed.")
        sys.exit(1)

    logging.info("Digest complete.")


if __name__ == "__main__":
    main()
