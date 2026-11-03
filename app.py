import streamlit as st
import pandas as pd
import plotly.express as px

from src.brand_monitoring_ollama import backend as monitor


# --------------------------------------------------
# Application Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="BrandPulse AI",
    page_icon="📡",
    layout="wide",
)


# --------------------------------------------------
# Database Initialization
# --------------------------------------------------

@st.cache_resource
def setup_database():
    monitor.init_db()
    return True


setup_database()


# --------------------------------------------------
# Session State
# --------------------------------------------------

def initialize_session():
    defaults = {
        "brand": "OpenAI",
        "report": None,
        "report_type": None,
        "report_brand": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()


# --------------------------------------------------
# Data Access
# --------------------------------------------------

def load_brand_mentions(brand):
    """Retrieve and prepare stored brand mentions."""

    records = monitor.get_all_mentions_as_df(brand)

    if records.empty:
        return records

    return monitor.clean_html(records.copy())


# --------------------------------------------------
# Data Collection
# --------------------------------------------------

def collect_mentions(brand, pages):
    """Fetch recent Hacker News discussions."""

    if not brand.strip():
        st.warning("Enter a brand name before searching.")
        return

    with st.spinner(f"Collecting mentions for {brand}..."):
        try:
            count = monitor.fetch_hn_mentions(
                brand_name=brand,
                max_pages=pages,
            )
        except Exception as exc:
            st.error(f"Unable to collect mentions: {exc}")
            return

    st.success(f"{count} new mentions collected.")
    st.rerun()


# --------------------------------------------------
# AI Classification
# --------------------------------------------------

def classify_mentions(pending_records):
    """Analyze unprocessed mentions using Ollama."""

    total = len(pending_records)
    completed = 0
    failed = 0

    progress = st.progress(0, text="Starting AI analysis...")

    for index, record in enumerate(
        pending_records.itertuples(index=False),
        start=1,
    ):
        try:
            sentiment = monitor.get_sentiment(record.text)
            topic = monitor.get_topic(record.text)
            urgency = monitor.get_urgency(record.text)

            if all([sentiment, topic, urgency]):
                monitor.update_mention_analysis(
                    record.id,
                    sentiment,
                    topic,
                    urgency,
                )
                completed += 1
            else:
                failed += 1

        except Exception as exc:
            failed += 1
            st.warning(
                f"Could not analyze mention {record.id}: {exc}"
            )

        progress.progress(
            index / total,
            text=f"Processed {index} of {total} mentions",
        )

    progress.empty()

    if failed:
        st.warning(
            f"Analysis finished: {completed} successful, "
            f"{failed} unsuccessful."
        )
    else:
        st.success(f"Successfully analyzed {completed} mentions.")

    st.rerun()


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

def render_sidebar():
    with st.sidebar:
        st.title("📡 BrandPulse AI")
        st.caption("AI-powered Hacker News brand intelligence")

        st.info(
            f"**Ollama model:** "
            f"`{monitor.DEFAULT_OLLAMA_MODEL.strip()}`"
        )

        st.divider()

        st.subheader("Search Settings")

        brand = st.text_input(
            "Brand or keyword",
            key="brand",
            placeholder="e.g. OpenAI",
        ).strip()

        page_count = st.slider(
            "Search depth (50 comments per page)",
            min_value=1,
            max_value=10,
            value=3,
        )

        if st.button(
            "Fetch Mentions",
            use_container_width=True,
            type="primary",
        ):
            collect_mentions(brand, page_count)

        st.divider()

        data = load_brand_mentions(brand) if brand else pd.DataFrame()

        pending = (
            data[data["sentiment"].isna()]
            if not data.empty
            else pd.DataFrame()
        )

        st.metric("Pending Analysis", len(pending))

        if not pending.empty:
            if st.button(
                f"Analyze {len(pending)} Mentions",
                use_container_width=True,
            ):
                classify_mentions(pending)

    return brand, data


# --------------------------------------------------
# Dashboard Metrics
# --------------------------------------------------

def render_metrics(records):
    total = len(records)

    analyzed = records.dropna(
        subset=["sentiment", "topic", "urgency"]
    )

    positive = (
        analyzed["sentiment"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("positive")
        .sum()
    )

    negative = (
        analyzed["sentiment"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("negative")
        .sum()
    )

    high_risk = (
        analyzed["urgency"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .eq("high urgency")
        .sum()
    )

    columns = st.columns(4)

    columns[0].metric("Total Mentions", total)
    columns[1].metric("Positive", int(positive))
    columns[2].metric("Negative", int(negative))
    columns[3].metric("High Urgency", int(high_risk))


# --------------------------------------------------
# Sentiment and Topic Charts
# --------------------------------------------------

def render_visualizations(records):
    analyzed = records.dropna(subset=["sentiment"]).copy()

    left, right = st.columns(2)

    with left:
        st.subheader("Sentiment Distribution")

        sentiment_counts = (
            analyzed["sentiment"]
            .astype(str)
            .str.strip()
            .value_counts()
            .rename_axis("Sentiment")
            .reset_index(name="Mentions")
        )

        if sentiment_counts.empty:
            st.info("No sentiment analysis available.")
        else:
            chart = px.pie(
                sentiment_counts,
                names="Sentiment",
                values="Mentions",
                hole=0.4,
                color="Sentiment",
                color_discrete_map={
                    "Positive": "#16A34A",
                    "Negative": "#DC2626",
                    "Neutral": "#64748B",
                },
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )

    with right:
        st.subheader("Discussion Topics")

        topic_counts = (
            analyzed["topic"]
            .dropna()
            .astype(str)
            .str.strip()
            .value_counts()
            .rename_axis("Topic")
            .reset_index(name="Mentions")
        )

        if topic_counts.empty:
            st.info("No classified topics available.")
        else:
            chart = px.bar(
                topic_counts,
                x="Topic",
                y="Mentions",
                color="Topic",
            )

            chart.update_layout(
                showlegend=False,
                xaxis_tickangle=-35,
            )

            st.plotly_chart(
                chart,
                use_container_width=True,
            )


# --------------------------------------------------
# AI-Generated Business Insights
# --------------------------------------------------

REPORT_GENERATORS = {
    "Positive Feedback": monitor.generate_positive_report_summary,
    "Negative Feedback": monitor.generate_negative_report_summary,
    "Product Suggestions": monitor.generate_report_summary,
}


def render_insights(records, brand):
    st.subheader("AI-Generated Insights")

    analyzed = records.dropna(subset=["sentiment"]).copy()

    if analyzed.empty:
        st.info("Analyze some mentions to generate insights.")
        return

    report_columns = st.columns(3)

    for column, (label, generator) in zip(
        report_columns,
        REPORT_GENERATORS.items(),
    ):
        with column:
            if st.button(label, use_container_width=True):
                with st.spinner(f"Generating {label.lower()}..."):
                    try:
                        result = generator(analyzed)

                        st.session_state.report = result
                        st.session_state.report_type = label
                        st.session_state.report_brand = brand

                    except Exception as exc:
                        st.error(f"Report generation failed: {exc}")

    if (
        st.session_state.report
        and st.session_state.report_brand == brand
    ):
        st.divider()
        st.markdown(
            f"#### {st.session_state.report_type}"
        )
        st.markdown(st.session_state.report)


# --------------------------------------------------
# Main Dashboard
# --------------------------------------------------

def render_dashboard(brand, records):
    st.title("📊 Brand Reputation Intelligence")
    st.caption(f"Monitoring Hacker News discussions about {brand}")

    if records.empty:
        st.info(
            "No mentions found yet. "
            "Use the sidebar to collect Hacker News comments."
        )
        return

    render_metrics(records)

    st.divider()

    overview_tab, mentions_tab = st.tabs(
        ["Analytics", "Mention Explorer"]
    )

    with overview_tab:
        render_visualizations(records)

        st.divider()

        render_insights(records, brand)

    with mentions_tab:
        st.subheader(f"Collected Mentions: {brand}")

        st.dataframe(
            records,
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------
# Application Entry Point
# --------------------------------------------------

def main():
    brand, records = render_sidebar()

    if not brand:
        st.warning("Please enter a brand or keyword.")
        return

    render_dashboard(brand, records)


if __name__ == "__main__":
    main()
