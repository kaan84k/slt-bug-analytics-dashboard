# SLT Bug Analytics Dashboard

This repository contains a Streamlit dashboard for analysing bug reports in the SLT Selfcare app.
Bug categories are annotated with standard syslog levels (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`) so you can filter and visualize bugs by severity. The data-processing pipeline assigns these levels automatically when `run_pipeline.py` is executed.

## Running Locally

Install dependencies and start Streamlit:

```bash
pip install -r requirements.txt
streamlit run src/dashboard/app.py
```

## Deployment Notes

When deploying on platforms such as Streamlit Community Cloud or other headless environments, include the `.streamlit/config.toml` file in the repository. This file ensures the server runs in headless mode and disables CORS, which is required for remote access:

```toml
[server]
headless = true
enableCORS = false
```

Having these settings in place lets the dashboard start automatically without needing a browser on the server and prevents CORS issues when accessing the app.
