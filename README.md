# SLT Bug Analytics Dashboard

This repository contains a Streamlit dashboard for analysing bug reports in the SLT Selfcare app.

## Running Locally

Install dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

The application reads configuration from environment variables. For local
development you can place them in a `.env` file, while in GitHub Actions these
values come from repository secrets. The `OPENAI_API_KEY` is used by the data
processing scripts and `bug_email_notifier.py` requires email credentials. A minimal `.env` might look like:

```env
OPENAI_API_KEY=
# EMAIL_USER=you@example.com
# EMAIL_PASS=app_password
# EMAIL_TO=
```

## Deployment Notes

When deploying on platforms such as Streamlit Community Cloud or other headless environments, include the `.streamlit/config.toml` file in the repository. This file ensures the server runs in headless mode and disables CORS, which is required for remote access:

```toml
[server]
headless = true
enableCORS = false
```

Having these settings in place lets the dashboard start automatically without needing a browser on the server and prevents CORS issues when accessing the app.
