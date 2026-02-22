# AI News Digest

Automated daily email digest of AI headlines, delivered every morning at **5 am ET**.

Sources: Google News RSS · Hacker News (via Algolia API)
Delivery: Gmail SMTP · Hosted: GitHub Actions (free)

---

## Setup (one-time, ~10 minutes)

### Step 1 — Push this repository to GitHub

1. Create a new repository on [github.com](https://github.com/new)
   - Name it anything, e.g. `ai-news-digest`
   - Set to **Private** (recommended — it will hold your email credentials as secrets)
2. Open a terminal in this folder and run:

```bash
git init
git add .
git commit -m "Initial commit: AI news digest"
git remote add origin https://github.com/YOUR_USERNAME/ai-news-digest.git
git branch -M main
git push -u origin main
```

---

### Step 2 — Create a Gmail App Password

Your Gmail account must use a **16-character App Password** (not your regular password).
App Passwords work even from GitHub's servers without triggering Google's security blocks.

1. Go to [myaccount.google.com](https://myaccount.google.com)
2. Click **Security** in the left sidebar
3. Under "How you sign in to Google", click **2-Step Verification** and enable it if not already on
4. Go back to Security and search for **App Passwords** (or visit [this direct link](https://myaccount.google.com/apppasswords))
5. Under "App name", type `AI Digest` then click **Create**
6. Copy the **16-character password** shown — you will not see it again

---

### Step 3 — Add GitHub Secrets

In your GitHub repository:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret** and add these two secrets:

| Secret name        | Value                              |
|--------------------|------------------------------------|
| `GMAIL_USER`       | `michael.n.madani@gmail.com`       |
| `GMAIL_APP_PASSWORD` | The 16-character App Password from Step 2 |

---

### Step 4 — Test the workflow manually

1. Go to your repository on GitHub
2. Click the **Actions** tab
3. Click **AI News Digest** in the left sidebar
4. Click **Run workflow** → **Run workflow**
5. Check your inbox at `michael.n.madani@gmail.com` within 2 minutes

---

## Schedule

The digest runs automatically every day at **10:00 UTC** (= 5:00 AM EST / 6:00 AM EDT).

GitHub Actions uses UTC exclusively. The cron expression `0 10 * * *` translates to:

| Timezone | Time     |
|----------|----------|
| EST (winter) | 5:00 AM |
| EDT (summer) | 6:00 AM |

> Note: GitHub Actions scheduled workflows may run a few minutes late during high-traffic periods. This is normal.

---

## File Structure

```
ai-news-digest/
├── .github/
│   └── workflows/
│       └── ai-digest.yml   # GitHub Actions workflow (runs daily)
├── digest.py               # Core script: fetch, filter, format, send
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

---

## How it works

1. **Fetch** — pulls AI headlines from Google News RSS and Hacker News Algolia API for the previous calendar day (US Eastern time)
2. **Filter** — applies keyword matching to keep only AI/ML-relevant stories
3. **Deduplicate** — removes HN stories that appear to be the same as a Google News story
4. **Format** — builds a clean HTML email with two sections
5. **Send** — delivers via Gmail SMTP using your App Password

The email is always sent, even if no articles are found — this confirms the workflow ran successfully.

---

## Troubleshooting

**Email not received**
- Check the Actions tab for workflow run status (green = success, red = failed)
- Verify both secrets are correctly named (`GMAIL_USER`, `GMAIL_APP_PASSWORD`)
- Make sure 2-Step Verification is enabled on your Google account before generating App Passwords

**Authentication error in logs**
- Regenerate the App Password in your Google account and update the `GMAIL_APP_PASSWORD` secret

**No articles in email**
- Google News occasionally returns cached results. The workflow will automatically include articles from the correct date range based on publish timestamps.
- Hacker News requires stories to have 10+ points to appear — quiet news days may show fewer stories
