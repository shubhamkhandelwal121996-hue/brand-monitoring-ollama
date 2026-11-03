import streamlit as st
import backend_utils as bu
import plotly.express as px

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Hacker News Brand Monitor",
    page_icon="🤖",
    layout="wide"
)

# --- Initialize Database ---
bu.init_db()

# --- Streamlit Session State ---
if 'brand_name' not in st.session_state:
    st.session_state['brand_name'] = "OpenAI"

# --- Sidebar (Configuration & Actions) ---
with st.sidebar:
    st.title("🤖 Hand N BrMonitor")
    st.info(f"Ollama (`{bu.DEFAULT_OLLAMA_MODEL}`) must be running locally.")

    st.header("Configuration")
    st.session_state.brand_name = st.text_input(
        "Brand/Keyword to Monitor",
        st.session_state.brand_name
    )

    # Added a slider so you can control how far back in time to search
    pages_to_fetch = st.slider("Pages to fetch (50 results per page)", 1, 10, 3)

    # --- Actions ---
    if st.button("Fetch Hacker News Mentions"):
        with st.spinner(f"Searching Hacker News for '{st.session_state.brand_name}'..."):
            hn_count = bu.fetch_hn_mentions(st.session_state.brand_name, max_pages=pages_to_fetch)
            st.success(f"Added {hn_count} new Hacker News comments.")
            st.rerun()

    # --- ANALYSIS PANEL ---
    st.divider()
    all_data_df = bu.get_all_mentions_as_df(st.session_state.brand_name)
    if not all_data_df.empty:
        all_data_df = bu.clean_html(all_data_df)

    pending_df = all_data_df[all_data_df['sentiment'].isnull()]


    st.info(f"**{len(pending_df)}** Pending Analysis")

    if not pending_df.empty:
        if st.button(f"Analyze {len(pending_df)} Items"):
            progress_bar = st.progress(0, text="Analyzing mentions via Ollama...")
            total = len(pending_df)

            for i, row in enumerate(pending_df.itertuples()):
                text_to_analyze = row.text
                sentiment = bu.get_sentiment(text_to_analyze)
                topic = bu.get_topic(text_to_analyze)
                urgency = bu.get_urgency(text_to_analyze)

                if sentiment and topic and urgency:
                    bu.update_mention_analysis(row.id, sentiment, topic, urgency)

                progress_bar.progress((i + 1) / total, text=f"Analyzing item {i + 1}/{total}")

            progress_bar.empty()
            st.success("Analysis complete!")
            st.rerun()

# --- Main Page Dashboard ---
st.title(f"Brand Reputation Dashboard: {st.session_state.brand_name}")

if not all_data_df.empty:
    analyzed_df = all_data_df.dropna(subset=['sentiment']).copy()

    tab1, tab2 = st.tabs(["Main Dashboard", "Raw Data"])

    # --- Tab 1: Main Dashboard ---
    with tab1:
        st.header("Overall Brand Sentiment")

        col1, col2 = st.columns(2)

        with col1:
            sentiment_counts = analyzed_df['sentiment'].value_counts()
            if not sentiment_counts.empty:
                fig_pie = px.pie(
                    sentiment_counts,
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Sentiment Breakdown",
                    color=sentiment_counts.index,
                    color_discrete_map={'Negative': 'red', 'Positive': 'green', 'Neutral': 'blue'}
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.write("No sentiment data available.")

        with col2:
            topics_exploded = analyzed_df['topic'].str.split(',').explode().str.strip()
            topic_counts = topics_exploded.value_counts()

            if not topic_counts.empty:
                fig_bar = px.bar(
                    topic_counts,
                    x=topic_counts.index,
                    y=topic_counts.values,
                    title="Top Topics Identified",
                    labels={'x': 'Topic', 'y': 'Count'}
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.write("No topics found.")

        st.divider()
        st.header("Automated Insights")

        col_pos, col_neg, col_sug = st.columns(3)
        with col_pos:
            if st.button("Summarize Positive Feedback"):
                with st.spinner("Ollama is summarizing..."):
                    summary = bu.generate_positive_report_summary(analyzed_df)
                    st.markdown(summary)

        with col_neg:
            if st.button("Summarize Negative Feedback"):
                with st.spinner("Ollama is summarizing..."):
                    summary = bu.generate_negative_report_summary(analyzed_df)
                    st.markdown(summary)

        with col_sug:
            if st.button("Generate Suggestions"):
                with st.spinner("Ollama is brainstorming..."):
                    summary = bu.generate_report_summary(analyzed_df)
                    st.markdown(summary)

    # --- Tab 2: Raw Data ---
    with tab2:
        st.header(f"Hacker News Data for '{st.session_state.brand_name}'")
        st.dataframe(all_data_df, use_container_width=True, hide_index=True)
else:
    st.info("No data fetched yet. Use the sidebar to search Hacker News!")