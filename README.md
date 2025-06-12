# SLT Bug Analytics Dashboard

This repository contains a Streamlit dashboard for analysing bug reports in the SLT Selfcare app.

## Running Locally

Install dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

The application requires a few environment variables. Create a `.env` file (see
the provided example) and set the login credentials:

```env
APP_EMAIL=web2directory84@gmail.com
APP_PASSWORD_HASH=<SHA256 hashed password>
```

The repository includes an example `.env` with the hashed value for
`Test!234`. You can update these variables to use your own credentials.

## Deployment Notes

When deploying on platforms such as Streamlit Community Cloud or other headless environments, include the `.streamlit/config.toml` file in the repository. This file ensures the server runs in headless mode and disables CORS, which is required for remote access:

```toml
[server]
headless = true
enableCORS = false
```

Having these settings in place lets the dashboard start automatically without needing a browser on the server and prevents CORS issues when accessing the app.
