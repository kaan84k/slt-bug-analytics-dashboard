SYSLOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

PRIORITY_TO_SYSLOG = {
    "Critical": "CRITICAL",
    "High": "ERROR",
    "Medium": "WARNING",
    "Low": "INFO",
    "Unknown": "DEBUG",
}

SYSLOG_COLOR_MAP = {
    "CRITICAL": "red",
    "ERROR": "orange",
    "WARNING": "gold",
    "INFO": "blue",
    "DEBUG": "gray",
}


def main() -> None:
    """Annotate bug datasets with syslog levels."""
    import pandas as pd

    summaries_path = "data/developer_bug_summaries.csv"
    bugs_path = "data/reclassified_bugs_with_sbert.csv"

    try:
        summaries_df = pd.read_csv(summaries_path)
        bugs_df = pd.read_csv(bugs_path)
    except Exception as exc:  # pragma: no cover - simple script
        print(f"Error loading data: {exc}")
        return

    summaries_df["syslog_level"] = (
        summaries_df.get("priority_level")
        .map(PRIORITY_TO_SYSLOG)
        .fillna("DEBUG")
    )

    category_map = summaries_df.set_index("bug_category")["syslog_level"].to_dict()
    bugs_df["syslog_level"] = bugs_df["bug_category"].map(category_map).fillna("DEBUG")

    summaries_df.to_csv(summaries_path, index=False)
    bugs_df.to_csv(bugs_path, index=False)

    print("Syslog levels annotated in developer summaries and bug data")


if __name__ == "__main__":  # pragma: no cover - manual execution
    main()
