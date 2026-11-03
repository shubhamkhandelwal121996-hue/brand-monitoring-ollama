
import streamlit as st
import pandas as pd
import plotly.express as px

from src.brand_monitoring_ollama import backend as monitor


# ============================================================
# 1. STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="BrandPulse AI",
    page_icon="📡",
    layout="wide",
)


# ============================================================
# 2. INITIALIZATION
# ============================================================

@st.cache_resource
def initialize_database():
    monitor.init_db()
    return True


initialize_database()


def initialize_state():
    defaults = {
        "brand": "OpenAI",
        "report": None,
        "report_title": None,
        "report_brand": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# ============================================================
# 3. DATA ACCESS
# ============================================================

def get_brand_data(brand: str) -> pd.DataFrame:
    """Load and clean stored Hacker News mentions."""

    data = monitor.get_all_mentions_as_df(brand)

    if data.empty:
        return data

    return monitor.clean_html(data)


# ============================================================
# 4. DATA COLLECTION
# ============================================================

def fetch_brand_mentions(brand: str, pages: int) -> None:
    """Collect brand discussions from Hacker News."""

    if not brand:
        st.warning("Please enter a brand name.")
        return

    with st.spinner(f"Searching Hacker News for '{brand}'..."):
        try:
            new_mentions = monitor.fetch_hn_mentions(
                brand_name=brand,
                max_pages=pages,
            )
        except Exception as error:
            st.error(f"Data collection failed: {error}")
            return

    st.toast(f"Collected {new_mentions} new mentions.")
    st.rerun()


# ============================================================
# 5. OLLAMA ANALYSIS
# ============================================================

def analyze_brand_mentions(brand: str, pending_count: int) -> None:
    """Classify pending mentions using the backend batch processor."""

    progress = st.progress(0, text="Preparing AI analysis...")

    def update_progress(completed: int, total: int) -> None:
        progress.progress(
            completed / total,
            text=f"Analyzing {completed}/{total} mentions",
        )

    try:
        analyzed_count = monitor.analyze_pending(
            brand_name=brand,
            max_workers=2,
            on_progress=update_progress,
        )

    except Exception as error:
        progress.empty()
        st.error(f"AI analysis failed: {error}")
        return

    progress.empty()

    if analyzed_count == pending_count:
        st.success(f"Successfully analyzed {analyzed_count} mentions.")
    else:
        st.warning(
            f"Analyzed {analyzed_count} of {pending_count} mentions. "
            "Some items may require another attempt."
        )

    st.rerun()


# ============================================================
# 6. SIDEBAR
# ============================================================

def render_sidebar():
    with st.sidebar:
        st.title("📡 BrandPulse AI")
        st.caption("Hacker News brand monitoring with Ollama")

        st.info(f"**Ollama Model:** `{monitor.MODEL}`")

        st.divider()

        st.subheader("Monitoring Settings")

        brand = st.text_input(
            "Brand or Keyword",
            key="brand",
            placeholder="Enter a brand name",
        ).strip()

        pages = st.slider(
            "Pages to Fetch",
            min_value=1,
            max_value=10,
            value=3,
            help="Each page contains up to 50 Hacker News comments.",
        )

        if st.button(
            "Fetch Hacker News Mentions",
            type="primary",
            use_container_width=True,
        ):
            fetch_brand_mentions(brand, pages)

        st.divider()

        records = (
            get_brand_data(brand)
            if brand
            else pd.DataFrame()
        )

        pending = (
            records[records["sentiment"].isna()]
            if not records.empty
            else pd.DataFrame()
        )

        pending_count = len(pending)

        st.metric("Pending Analysis", pending_count)

        if pending_count > 0:
            if st.button(
                f"Analyze {pending_count} Mentions",
                use_container_width=True,
            ):
                analyze_brand_mentions(brand, pending_count)

    return brand, records


# ============================================================
# 7. DASHBOARD METRICS
# ============================================================

def display_metrics(records: pd.DataFrame) -> None:
    """Display important brand monitoring statistics."""

    total_mentions = len(records)

    analyzed = records.dropna(
        subset=["sentiment", "topic", "urgency"]
    )

    sentiment_values = (
        analyzed["sentiment"].astype(str).str.strip()
    )

    urgency_values = (
        analyzed["urgency"].astype(str).str.strip()
    )

    positive_count = sentiment_values.eq("Positive").sum()
    negative_count = sentiment_values.eq("Negative").sum()
    high_urgency_count = urgency_values.eq("High Urgency").sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Mentions", total_mentions)
    col2.metric("Positive", int(positive_count))
    col3.metric("Negative", int(negative_count))
    col4.metric("High Urgency", int(high_urgency_count))


# ============================================================
# 8. DATA VISUALIZATION
# ============================================================

def display_charts(records: pd.DataFrame) -> None:
    """Display sentiment and topic distributions."""

    analyzed = records.dropna(
        subset=["sentiment", "topic", "urgency"]
    ).copy()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Analysis")

        sentiment_counts = (
            analyzed["sentiment"]
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Count")
        )

        if sentiment_counts.empty:
            st.info("No sentiment data available.")
        else:
            sentiment_chart = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Count",
                hole=0.4,
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#22C55E",
                    "Negative": "#EF4444",
                    "Neutral": "#3B82F6",
                },
            )

            st.plotly_chart(
                sentiment_chart,
                use_container_width=True,
            )

    with col2:
        st.subheader("Discussion Topics")

        topic_counts = (
            analyzed["topic"]
            .value_counts()
            .rename_axis("Topic")
            .reset_index(name="Count")
        )

        if topic_counts.empty:
            st.info("No topics have been classified yet.")
        else:
            topic_chart = px.bar(
                topic_counts,
                x="Topic",
                y="Count",
                color="Topic",
            )

            topic_chart.update_layout(
                showlegend=False,
                xaxis_tickangle=-30,
            )

            st.plotly_chart(
                topic_chart,
                use_container_width=True,
            )


# ============================================================
# 9. AI-GENERATED REPORTS
# ============================================================

REPORT_FUNCTIONS = {
    "Positive Feedback": monitor.generate_positive_report_summary,
    "Negative Feedback": monitor.generate_negative_report_summary,
    "Product Suggestions": monitor.generate_report_summary,
}


def display_reports(records: pd.DataFrame, brand: str) -> None:
    """Generate and display AI-powered brand insights."""

    st.subheader("AI-Generated Business Insights")

    analyzed = records.dropna(
        subset=["sentiment", "topic", "urgency"]
    )

    if analyzed.empty:
        st.info("Analyze some mentions to generate reports.")
        return

    columns = st.columns(3)

    for column, (title, report_function) in zip(
        columns,
        REPORT_FUNCTIONS.items(),
    ):
        with column:
            if st.button(
                title,
                use_container_width=True,
            ):
                with st.spinner(f"Generating {title.lower()}..."):
                    try:
                        report = report_function(analyzed)

                        st.session_state["report"] = report
                        st.session_state["report_title"] = title
                        st.session_state["report_brand"] = brand

                    except Exception as error:
                        st.error(f"Report generation failed: {error}")

    if (
        st.session_state["report"]
        and st.session_state["report_brand"] == brand
    ):
        st.divider()

        st.markdown(
            f"### {st.session_state['report_title']}"
        )

        st.markdown(st.session_state["report"])


# ============================================================
# 10. MAIN DASHBOARD
# ============================================================

def display_dashboard(brand: str, records: pd.DataFrame) -> None:
    st.title("📊 Brand Reputation Dashboard")
    st.caption(f"Tracking Hacker News conversations about {brand}")

    if records.empty:
        st.info(
            "No mentions available. "
            "Use the sidebar to fetch Hacker News comments."
        )
        return

    display_metrics(records)

    st.divider()

    dashboard_tab, data_tab = st.tabs(
        ["Analytics Dashboard", "Raw Mentions"]
    )

    with dashboard_tab:
        display_charts(records)

        st.divider()

        display_reports(records, brand)

    with data_tab:
        st.subheader(f"Hacker News Mentions: {brand}")

        st.dataframe(
            records,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# 11. APPLICATION ENTRY POINT
# ============================================================

def main():
    brand, records = render_sidebar()

    if not brand:
        st.warning("Enter a brand name to continue.")
        return

    display_dashboard(brand, records)


if __name__ == "__main__":
    main()
