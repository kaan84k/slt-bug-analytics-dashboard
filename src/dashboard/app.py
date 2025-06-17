import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
from data_processing.non_bug import UserExperienceAnalyzer
from dashboard.bug_email_notifier import main as send_bug_digest
from db_utils import load_df, table_exists

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

# --- Syslog severity mapping ---
SYSLOG_MAP = {
    "Crash/Freeze":      ("crit",    2),
    "Server Error":      ("crit",    2),
    "Login Error":       ("err",     3),
    "Payment Issue":     ("err",     3),
    "Contact Change Issue": ("err", 3),
    "Network/Connection": ("err",     3),
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
st.title("SLT Selfcare App - Bug Analytics Dashboard")

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

# Generate TicketID (BUG-1001, ...)
df = df.reset_index(drop=True)
df['TicketID'] = ["BUG-%04d" % (i+1001) for i in range(len(df))]

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


# --- Sidebar filter ---
st.sidebar.header("Filter by Syslog Level")
selected_levels = st.sidebar.multiselect(
    "Syslog Level", SYSLOG_LEVELS, default=SYSLOG_LEVELS
)
filtered_df = df_display[df_display['syslog_level'].isin(selected_levels)]

# --- Download buttons ---
st.sidebar.header("Download")
csv_bytes = filtered_df.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(
    "Download CSV", csv_bytes, "filtered_bug_tickets.csv", "text/csv"
)

# Syslog-style log file
def to_syslog_row(row):
    # <severity_code> [TicketID] (LEVEL) Category in vX on YYYY-MM-DD: description
    return f"<{row['severity_code']}> [{row['TicketID']}] ({row['syslog_level'].upper()}) {row['bug_category']} in v{row['appVersion']} on {row['review_date'].date() if pd.notna(row['review_date']) else 'N/A'}: {row['review_description']}"

syslog_lines = filtered_df.apply(to_syslog_row, axis=1)
syslog_bytes = "\n".join(syslog_lines).encode('utf-8')
st.sidebar.download_button(
    "Download Syslog Log", syslog_bytes, "bug_tickets.log", "text/plain"
)

# --- Styled table with row highlighting ---
def highlight_row(row):
    color = SYSLOG_COLORS.get(row['syslog_level'], "#f7f7f7")
    return ['background-color: %s' % color]*len(row)


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
feedback_df = predictions_df[predictions_df['is_bug_report'] == 0].copy() if 'is_bug_report' in predictions_df.columns else pd.DataFrame()

# Sidebar filters
st.sidebar.header("📎 Filter Options")

# Date range filter (assuming your data has a 'date' column)
if 'date' in bug_df.columns:
    bug_df['date'] = pd.to_datetime(bug_df['date'])
    min_date = bug_df['date'].min()
    max_date = bug_df['date'].max()
    date_range = st.sidebar.date_input(
        "Select Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

# Clean and prepare version data
def clean_versions(versions):
    # Convert all versions to strings and filter out empty/null values
    cleaned = []
    for v in versions:
        if pd.notna(v):
            # Handle both string and numeric versions
            ver_str = str(v).strip()
            if ver_str:  # Check if not empty string
                cleaned.append(ver_str)
    # Natural sort for version numbers
    return sorted(cleaned, key=lambda x: [int(n) if n.isdigit() else n.lower() 
                                        for n in x.replace('.', ' ').split()])

# Get clean versions
available_versions = clean_versions(bug_df['appVersion'].unique())

selected_versions = st.sidebar.multiselect(
    "Select App Versions",
    options=available_versions,
    default=available_versions
)

selected_categories = st.sidebar.multiselect(
    "Select Bug Categories",
    options=sorted(bug_df['bug_category'].unique()),
    default=bug_df['bug_category'].unique()
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
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Bugs", len(filtered_bug_df))
with col2:
    st.metric("Categories", len(selected_categories))
with col3:
    st.metric("App Versions", len(selected_versions))

# Download button for filtered data
st.sidebar.download_button(
    "📥 Download Filtered Data",
    filtered_bug_df.to_csv(index=False).encode('utf-8'),
    "filtered_bugs.csv",
    "text/csv"
)


# Organize visualizations in tabs
tab_tickets, tab1, tab2, tab3, tab4 = st.tabs([
    "Bug Tickets",
    "Bug Categories",
    "Time Analysis",
    "Developer Insights",
    "Sentiment Patterns",
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

# --- Export Options ---
st.sidebar.header("📤 Export Options")
try:
    # Export summary reports per category
    export_category = st.sidebar.selectbox(
        "Export Summary for Category",
        options=["All"] + list(nlp_df['bug_category'].unique())
    )
    if export_category == "All":
        export_df = nlp_df.drop(columns=['review_description'], errors='ignore')
        filename = "developer_bug_summaries_all.csv"
    else:
        export_df = nlp_df[nlp_df['bug_category'] == export_category].drop(columns=['review_description'], errors='ignore')
        filename = f"developer_bug_summaries_{export_category}.csv"
    st.sidebar.download_button(
        label=f"Download Summary Report ({export_category})",
        data=export_df.to_csv(index=False).encode('utf-8'),
        file_name=filename,
        mime='text/csv'
    )
except Exception as e:
    st.sidebar.warning(f"Error setting up export: {str(e)}")

# --- Manual Email Trigger ---
st.sidebar.header("📧 Bug Digest Email")
if st.sidebar.button("Send Email Now"):
    try:
        send_bug_digest()
        st.sidebar.success("Bug digest sent")
    except Exception as e:
        st.sidebar.error(f"Failed to send email: {e}")

