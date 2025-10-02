import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
from data_processing.non_bug import UserExperienceAnalyzer
from dashboard.bug_email_notifier import main as send_bug_digest
from dashboard.bug_attend_notification import load_attended, mark_tickets_attended
from db_utils import load_df, table_exists
from ticket_utils import assign_ticket_ids

@st.cache_data
def load_data(path: str, table: str) -> pd.DataFrame:
    """Load data from CSV if available, otherwise from the database."""
    if os.path.exists(path):
        return pd.read_csv(path)
    if table_exists(table):
        return load_df(table)
    raise FileNotFoundError(f"{path} not found and table {table} missing")

@st.cache_resource
def load_analyzer(df: pd.DataFrame) -> UserExperienceAnalyzer:
    """Load UserExperienceAnalyzer with caching to avoid reinitialization."""
    return UserExperienceAnalyzer(df)
import io
import base64
import re

st.set_page_config(page_title="SLT Bug Analytics Dashboard", layout="wide")

# --- Syslog severity mappingt ---
SYSLOG_MAP = {
    "Crash/Freeze":      ("crit",    2),
    "Server Error":      ("crit",    2),
    "Service Unavailability": ("crit", 2),
    "Login Error":       ("err",     3),
    "Payment Issue":     ("err",     3),
    "Contact Change Issue": ("err", 3),
    "Network/Connection": ("err",     3),
    "Data Sync Issue":   ("err",     3),
    "General Error":     ("err",     3),
    "OTP Issue":       ("warning", 4),
    "Update Issue":      ("warning", 4),
    "Slow Performance":  ("warning", 4),
    "Notification Problem": ("notice", 5),
    "UI Issue":          ("info",    6),
    "Other":             ("debug",   7)
}

# Alias used in developer insights
CATEGORY_SEVERITY_MAP = SYSLOG_MAP
SYSLOG_LEVELS = ["crit", "err", "warning", "notice", "info", "debug"]
SYSLOG_COLORS = {
    "crit":    "#ff0000",
    "err":     "#ff9500",
    "warning": "#ffd900",
    "notice":  "#0066ff",
    "info":    "#d97575",
    "debug":   "#fabebe"
}

# Set page configuration first to avoid StreamlitAPIException
st.title("SLT Selfcare App - Bug ")

# --- Load and process data ---
try:
    df = load_data("data/categorized_bugs.csv", "categorized_bugs")
except Exception as e:
    st.error(f"Error loading categorized_bugs.csv: {str(e)}")
    st.stop()

# Assign syslog_level and severity_code
def map_syslog(row):
    cat = str(row['bug_category'])
    level, code = SYSLOG_MAP.get(cat, ("debug", 7))
    return pd.Series([level, code])

df[['syslog_level', 'severity_code']] = df.apply(map_syslog, axis=1)

# Generate stable TicketID using a hash of bug fields
df = assign_ticket_ids(df)

# Convert review_date to datetime and sort
df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
df = df.sort_values('review_date', ascending=False)

# Add Timestamp column (alias for review_date)
df['Timestamp'] = df['review_date']

# Reorder columns
display_cols = [
    "TicketID", "Timestamp", "syslog_level", "severity_code",
    "bug_category", "appVersion", "review_date", "review_description"
]
df_display = df[display_cols]

# Load tickets already marked as attended
attended_df = load_attended()


# --- Sidebar filter ---
# --- Sidebar: main container for filters and actions ---
filter_expander = st.sidebar.expander("🔍 Filter Options", expanded=True)

with filter_expander:
    st.markdown("**Syslog Level**")
    selected_levels = st.multiselect(
        "Syslog Level", SYSLOG_LEVELS, default=SYSLOG_LEVELS
    )

filtered_df = df_display[df_display["syslog_level"].isin(selected_levels)]

download_expander = st.sidebar.expander("📥 Download", expanded=False)

with download_expander:
    csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV", csv_bytes, "filtered_bug_tickets.csv", "text/csv"
    )

    # Syslog-style log file
    def to_syslog_row(row):
        return (
            f"<{row['severity_code']}> [{row['TicketID']}] ({row['syslog_level'].upper()}) "
            f"{row['bug_category']} in v{row['appVersion']} on "
            f"{row['review_date'].date() if pd.notna(row['review_date']) else 'N/A'}: "
            f"{row['review_description']}"
        )

    syslog_lines = filtered_df.apply(to_syslog_row, axis=1)
    syslog_bytes = "\n".join(syslog_lines).encode("utf-8")
    st.download_button(
        "Download Syslog Log", syslog_bytes, "bug_tickets.log", "text/plain"
    )

# --- Styled table with row highlighting ---
def highlight_row(row):
    color = SYSLOG_COLORS.get(row['syslog_level'], "#f7f7f7")
    return ['background-color: %s' % color]*len(row)


# Clean and prepare version data
def clean_versions(versions):
    cleaned = []
    for v in versions:
        if pd.notna(v):
            ver_str = str(v).strip()
            if ver_str:
                cleaned.append(ver_str)
    return sorted(
        cleaned,
        key=lambda x: [int(n) if n.isdigit() else n.lower() for n in x.replace('.', ' ').split()],
    )


# --- Existing dashboard code ---

# Load categorized bugs and NLP summaries with error handling
try:
    bug_df = load_data("data/categorized_bugs.csv", "categorized_bugs")
    nlp_df = load_data("data/developer_bug_summaries.csv", "developer_bug_summaries")
    predictions_df = load_data("data/bug_predictions.csv", "bug_predictions")
except Exception as e:
    st.error(f"Error loading data files: {str(e)}")
    st.info("Please run the full pipeline to generate all required files.")
    st.stop()

# Convert review_date to datetime with error handling
try:
    if 'review_date' in predictions_df.columns:
        predictions_df['review_date'] = pd.to_datetime(predictions_df['review_date'])
except Exception as e:
    st.warning(f"Error processing review dates: {str(e)}")
    
# Focus on non-bug feedback (is_bug=0)
feedback_df = (
    predictions_df[predictions_df["is_bug_report"] == 0].copy()
    if "is_bug_report" in predictions_df.columns
    else pd.DataFrame()
)

# Additional filter options
with filter_expander:
    if "date" in bug_df.columns:
        bug_df["date"] = pd.to_datetime(bug_df["date"])
        min_date = bug_df["date"].min()
        max_date = bug_df["date"].max()
        date_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

    # Get clean versions
    available_versions = clean_versions(bug_df["appVersion"].unique())
    latest_version = available_versions[-1] if available_versions else "N/A"

    selected_versions = st.multiselect(
        "Select App Versions",
        options=available_versions,
        default=available_versions,
    )

    selected_categories = st.multiselect(
        "Select Bug Categories",
        options=sorted(bug_df["bug_category"].unique()),
        default=bug_df["bug_category"].unique(),
    )

# Filter data
filtered_bug_df = bug_df[
    (bug_df['appVersion'].astype(str).isin(selected_versions)) &
    (bug_df['bug_category'].isin(selected_categories))
]

if 'date' in bug_df.columns:
    filtered_bug_df = filtered_bug_df[
        (filtered_bug_df['date'] >= pd.Timestamp(date_range[0])) &
        (filtered_bug_df['date'] <= pd.Timestamp(date_range[1]))
    ]

# Metrics summary
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Bugs", len(filtered_bug_df))
with col2:
    st.metric("Categories", len(selected_categories))
with col3:
    st.metric("App Versions", len(selected_versions))
with col4:
    st.metric("Latest Version", latest_version)

# Add filtered data download to the download section
with download_expander:
    st.download_button(
        "📥 Download Filtered Data",
        filtered_bug_df.to_csv(index=False).encode("utf-8"),
        "filtered_bugs.csv",
        "text/csv",
    )


# Organize visualizations in tabs
tab_tickets, tab1, tab2, tab3, tab4, tab_attended = st.tabs([
    "Bug Tickets",
    "Bug Categories",
    "Time Analysis",
    "Developer Insights",
    "Sentiment Patterns",
    "Attended Tickets",
])

with tab_tickets:
    st.markdown("### Bug Tickets")
    styled = filtered_df.style.apply(highlight_row, axis=1)
    st.dataframe(styled, use_container_width=True, hide_index=True)

with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Bug Count per Category")
        filtered_bug_counts = filtered_bug_df['bug_category'].value_counts().reset_index()
        filtered_bug_counts.columns = ['Bug Category', 'Count']
        st.bar_chart(filtered_bug_counts.set_index('Bug Category'))
    
    with col2:
        st.subheader("Category Distribution")
        fig_pie = px.pie(filtered_bug_counts, names='Bug Category', values='Count')
        st.plotly_chart(fig_pie, use_container_width=True)

with tab2:
    st.subheader("Bug Categories Across App Versions")
    filtered_version_category = pd.crosstab(
        filtered_bug_df['appVersion'], 
        filtered_bug_df['bug_category']
    ).reset_index().melt(id_vars='appVersion', var_name='Bug Category', value_name='Count')
    
    fig_bar = px.bar(
        filtered_version_category,
        x='appVersion',
        y='Count',
        color='Bug Category',
        labels={'appVersion': 'App Version'},
        hover_data={'Count': True, 'Bug Category': True}
    )
    fig_bar.update_layout(barmode='stack', xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Updated: Bug Categories aggregated by day of month ---
    st.subheader("Bug Categories by Day of Month")
    if 'review_date' in filtered_bug_df.columns:
        try:
            df_day = filtered_bug_df.copy()
            df_day['review_date'] = pd.to_datetime(df_day['review_date'])
            df_day['day'] = df_day['review_date'].dt.day
            day_cat = (
                df_day.groupby(['day', 'bug_category']).size().reset_index(name='Count')
            )
            day_cat = day_cat.pivot(index='day', columns='bug_category', values='Count')
            day_cat = day_cat.reindex(range(1, 32), fill_value=0)
            stacked = day_cat.reset_index().melt(id_vars='day', var_name='Bug Category', value_name='Count')
            fig_daily = px.bar(
                stacked,
                x='day',
                y='Count',
                color='Bug Category',
                labels={'day': 'Date'}
            )
            fig_daily.update_layout(barmode='stack', xaxis=dict(dtick=1))
            st.plotly_chart(fig_daily, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating time series graph: {str(e)}")
    else:
        st.info("review_date column required for Time Analysis.")

    # --- User Journey View: Bug Timeline & Sentiment Heatmap ---
    with st.expander("🧭 User Journey View: Bug Timeline & Sentiment Heatmap", expanded=False):
        st.markdown("#### Timeline of Bugs by App Version and Sentiment")
        if 'review_date' in filtered_bug_df.columns and 'appVersion' in filtered_bug_df.columns:
            try:
                # Prepare data for heatmap
                heatmap_df = filtered_bug_df.copy()
                heatmap_df['review_date'] = pd.to_datetime(heatmap_df['review_date'])
                if 'sentiment' in heatmap_df.columns:
                    sentiment_col = 'sentiment'
                else:
                    sentiment_col = None
                # Group by review_date, appVersion, and optionally sentiment
                if sentiment_col:
                    grouped_heatmap = heatmap_df.groupby([
                        pd.Grouper(key='review_date', freq='W'), 'appVersion', 'bug_category'
                    ])[sentiment_col].mean().reset_index()
                    fig = px.density_heatmap(
                        grouped_heatmap,
                        x='review_date',
                        y='appVersion',
                        z=sentiment_col,
                        color_continuous_scale='RdYlGn',
                        title='Bug Sentiment Heatmap by App Version (Weekly)'
                    )
                else:
                    grouped_heatmap = heatmap_df.groupby([
                        pd.Grouper(key='review_date', freq='W'), 'appVersion', 'bug_category'
                    ]).size().reset_index(name='count')
                    fig = px.density_heatmap(
                        grouped_heatmap,
                        x='review_date',
                        y='appVersion',
                        z='count',
                        color_continuous_scale='YlOrRd',
                        title='Bug Count Heatmap by App Version (Weekly)'
                    )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating heatmap: {str(e)}")
        else:
            st.info("review_date and appVersion columns required for User Journey View.")


with tab3:
    st.subheader("Developer Insights")

    # Add severity filter
    severity_filter = st.selectbox(
        "Filter by Severity",
        ["All", "crit", "err", "warning", "notice", "info", "debug"]
    )

    # Helper functions
            
    def clean_text(text):
        if pd.isna(text) or not text:
            return ""
            
        text = str(text)
        # Remove markdown headers, asterisks, etc.
        text = text.replace("### 1.", "").replace("### 2.", "")
        text = text.replace("### 3.", "").replace("### 4.", "")
        text = text.replace("****", "")
        text = text.strip()
        return text
    
    def remove_label_prefixes(text):
        pattern = r'(?im)^\s*["\']?(SUMMARY|KEY FINDINGS|SUGGESTED ACTIONS|PRIORITY LEVEL):\**\s*'
        return re.sub(pattern, '', text).strip()

    # Format text into a bulleted list
    def format_bullet_list(text):
        if pd.isna(text) or not text:
            return []

        text = clean_text(text)
        bullet_items = []
        
        # Try various delimiters to identify bullet points
        if '\n-' in text:
            # Split by lines that start with a dash
            raw_items = text.split('\n-')
            for i, item in enumerate(raw_items):
                if i == 0 and not item.strip().startswith('-'):
                    # First item typically doesn't have a dash prefix
                    bullet_items.append(item.strip())
                else:
                    bullet_items.append(item.strip())
        elif '\n' in text and len(text.split('\n')) > 1:
            # Split by newlines if multiple lines exist
            bullet_items = [line.strip() for line in text.split('\n') if line.strip()]
        elif '-' in text and text.count('-') > 1:
            # Split by dashes if multiple dashes exist
            bullet_items = [item.strip() for item in text.split('-') if item.strip()]
        elif '.' in text and text.count('.') > 2:
            # Split by periods if multiple periods exist (and not just single sentence)
            bullet_items = [item.strip() for item in text.split('.') if item.strip()]
        else:
            # Just use the whole text as one item
            bullet_items = [text]
            
        # Clean up each item
        clean_items = []
        for item in bullet_items:
            # Remove any bullet points or numbers at the beginning
            clean_item = item.strip().lstrip('1234567890.-*•›✓✅✔︎☑︎■□●○•⦿⁃◦-–—−⋅᛫∙ ')
            if clean_item:  # Skip empty items
                clean_items.append(clean_item)
                
        return clean_items
    
    # Filter categories by severity
    filtered_categories = []
    for category in selected_categories:
        try:
            severity_code, severity_number = CATEGORY_SEVERITY_MAP.get(category, ("unknown", 0))
            if severity_filter == "All" or severity_code == severity_filter:
                filtered_categories.append(category)
        except Exception as e:
            st.warning(f"Error processing category {category}: {str(e)}")
            if severity_filter == "All":
                filtered_categories.append(category)
    
    # Display insights for filtered categories
    for category in filtered_categories:
        try:
            category_insights = nlp_df[nlp_df['bug_category'] == category]
            if not category_insights.empty:
                with st.expander(f"📌 {category}", expanded=True):
                    try:
                        # Category summary metrics
                        col1, col2 = st.columns(2)
                        with col1:
                            bug_count = len(filtered_bug_df[filtered_bug_df['bug_category'] == category])
                            st.metric("Bug Count", bug_count)
                        with col2:
                            severity_code, severity_number = CATEGORY_SEVERITY_MAP.get(category, ("unknown", 0))
                            st.metric("Severity", f"{severity_code} ({severity_number})")
                        # Display summary if available
                        if 'summary' in category_insights.columns and pd.notna(category_insights['summary'].iloc[0]):
                            summary_text = remove_label_prefixes(clean_text(category_insights['summary'].iloc[0]))
                            if summary_text:
                                st.markdown("### 📝 Summary")
                                st.write(summary_text)
                        # Display key findings
                        if 'key_findings' in category_insights.columns and pd.notna(category_insights['key_findings'].iloc[0]):
                            findings = remove_label_prefixes(clean_text(category_insights['key_findings'].iloc[0]))
                            st.markdown("### 🔍 Key Findings")
                            findings_list = format_bullet_list(findings)
                            for finding in findings_list:
                                if finding.strip():
                                    st.markdown(f"- {finding}")
                        # Display suggested actions
                        if 'suggested_actions' in category_insights.columns and pd.notna(category_insights['suggested_actions'].iloc[0]):
                            action_text = remove_label_prefixes(clean_text(category_insights['suggested_actions'].iloc[0]))
                            st.markdown("### 💡 Suggested Actions")
                            actions = format_bullet_list(action_text)
                            for idx, action in enumerate(actions, 1):
                                if action.strip():
                                    st.markdown(f"{idx}. {action}")
                        # if 'additional_notes' in category_insights.columns and pd.notna(category_insights['additional_notes'].iloc[0]):
                        #     notes = remove_label_prefixes(clean_text(category_insights['additional_notes'].iloc[0]))
                        #     if notes.strip() and notes.lower() != 'nan':
                        #         st.markdown("### 📊 Priority Analysis")
                        #         st.info(notes)
                    except Exception as e:
                        st.error(f"Error displaying insights for {category}: {str(e)}")
        except Exception as e:
            st.error(f"Error processing category {category}: {str(e)}")
with tab4:
    st.subheader("Advanced Sentiment Analysis")
    
    try:
        # Initialize the analyzer
        bug_pred_df = load_data('data/bug_predictions.csv', 'bug_predictions')
        analyzer = load_analyzer(bug_pred_df)
        
        # Display summary metrics
        summary = analyzer.get_sentiment_summary()
        
        # Create metric columns
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Non-Bug Reviews", summary['total_reviews'])
        with col2:
            st.metric("Average Sentiment", f"{summary['average_sentiment']:.2f}")
        with col3:
            st.metric("Average Rating", f"{summary['average_rating']:.2f}")
        
        # Show sentiment trends
        st.subheader("📈 Sentiment Trends")
        sentiment_trend_fig = analyzer.analyze_sentiment_trends()
        st.plotly_chart(sentiment_trend_fig, use_container_width=True)
        
        # Show version analysis
        st.subheader("📱 Version Analysis")
        version_sentiment_fig = analyzer.analyze_version_sentiment()
        st.plotly_chart(version_sentiment_fig, use_container_width=True)
        
        # Topic Analysis
        st.subheader("🔍 Topic Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            # Key phrases
            st.markdown("### Key Phrases")
            key_phrases = analyzer.extract_key_phrases()
            st.dataframe(key_phrases)
        
        with col2:
            # Topic clusters
            st.markdown("### Topic Clusters")
            topics = analyzer.create_topic_clusters()
            st.dataframe(topics)
        
        # Word Cloud
        st.subheader("☁️ Word Cloud")
        wordcloud_plt = analyzer.generate_wordcloud()
        
        # Convert matplotlib plot to image
        buf = io.BytesIO()
        wordcloud_plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        st.image(buf)
        
        # Sentiment Distribution
        st.subheader("📊 Sentiment Distribution")
        sentiment_dist = pd.DataFrame.from_dict(
            summary['sentiment_distribution'],
            orient='index',
            columns=['Count']
        )
        st.bar_chart(sentiment_dist)
    except Exception as e:
        st.error(f"Error in sentiment analysis: {str(e)}")

with tab_attended:
    st.subheader("Attended Tickets")
    if attended_df.empty:
        st.info("No tickets marked as attended.")
    else:
        merged = pd.merge(attended_df, df_display, on="TicketID", how="left")
        st.dataframe(
            merged.sort_values("attended_at", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# --- Export Options ---
export_expander = st.sidebar.expander("📤 Export Options", expanded=False)
with export_expander:
    try:
        export_category = st.selectbox(
            "Export Summary for Category",
            options=["All"] + list(nlp_df["bug_category"].unique()),
        )
        if export_category == "All":
            export_df = nlp_df.drop(columns=["review_description"], errors="ignore")
            filename = "developer_bug_summaries_all.csv"
        else:
            export_df = nlp_df[nlp_df["bug_category"] == export_category].drop(
                columns=["review_description"], errors="ignore"
            )
            filename = f"developer_bug_summaries_{export_category}.csv"
        st.download_button(
            label=f"Download Summary Report ({export_category})",
            data=export_df.to_csv(index=False).encode("utf-8"),
            file_name=filename,
            mime="text/csv",
        )
    except Exception as e:
        st.warning(f"Error setting up export: {str(e)}")

# --- Manual Email Trigger ---
email_expander = st.sidebar.expander("📧 Bug Digest Email", expanded=False)
with email_expander:
    if st.button("Send Email Now"):
        try:
            send_bug_digest()
            st.success("Bug digest sent")
        except Exception as e:
            st.error(f"Failed to send email: {e}")

# --- Bug Attendance ---
attendance_expander = st.sidebar.expander("\u2705 Mark Tickets Attended", expanded=False)
with attendance_expander:
    attend_selection = st.multiselect(
        "Select Ticket IDs",
        options=df_display["TicketID"],
    )
    if st.button("Mark as Attended"):
        mark_tickets_attended(attend_selection)
        attended_df = load_attended()
        st.success("Tickets updated")

