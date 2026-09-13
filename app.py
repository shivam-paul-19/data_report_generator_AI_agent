import json
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import streamlit as st
from sklearn.datasets import (
    fetch_openml,
    load_iris,
    load_wine,
    load_breast_cancer,
    fetch_california_housing,
)

from agent import DataReportOrchestrator, orchestrator_worker
from information_extractor import DataInformationExtractor
from chart_generator import ChartGenerator, cleanup_chart_directory, cleanup_chart_files
from pdf_maker import make_pdf_from_report

st.set_page_config(
    page_title="Data Information & AI Report Generator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def on_download_cleanup_action(chart_dir_name: str, charts_list: Optional[list] = None):
    """Deletes generated images and chart folder from disk immediately upon download."""
    try:
        if charts_list:
            cleanup_chart_files(charts_list)
        if chart_dir_name:
            cleanup_chart_directory(chart_dir_name)
    except Exception:
        pass

st.title("📊 Data Information & AI Report Generator")
st.markdown(
    "Upload any **CSV** or **Excel (.xlsx / .xls)** dataset or pick a sample dataset (e.g. **Titanic** from sklearn) "
    "to extract structural metadata, statistical summaries, correlations, and generate a comprehensive "
    "**AI Analytical Report** powered by high-performance LangGraph orchestration."
)


@st.cache_data(show_spinner="Loading sample dataset...")
def load_sample_dataset(dataset_name: str) -> pd.DataFrame:
    """Helper function to load example datasets from scikit-learn / OpenML with caching."""
    if dataset_name == "Titanic Passengers (sklearn / OpenML)":
        try:
            return fetch_openml("titanic", version=1, as_frame=True).frame
        except Exception:
            # Fallback if offline/network issue
            iris = load_iris(as_frame=True)
            return iris.frame
    elif dataset_name == "Iris Flower Dataset (sklearn)":
        return load_iris(as_frame=True).frame
    elif dataset_name == "Wine Recognition Dataset (sklearn)":
        return load_wine(as_frame=True).frame
    elif dataset_name == "Breast Cancer Diagnostic (sklearn)":
        return load_breast_cancer(as_frame=True).frame
    elif dataset_name == "California Housing Dataset (sklearn)":
        return fetch_california_housing(as_frame=True).frame
    else:
        return load_iris(as_frame=True).frame


# Sidebar for file upload and example datasets
st.sidebar.header("📁 Data Input")

data_mode = st.sidebar.radio(
    "Select Data Source:",
    options=["Example Datasets (sklearn / OpenML)", "Upload Custom Dataset"],
    index=0,
    help="Choose whether to use built-in example datasets like Titanic or upload your own file.",
)

data_source = None
file_name = None

if data_mode == "Example Datasets (sklearn / OpenML)":
    selected_sample = st.sidebar.selectbox(
        "Choose an Example Dataset:",
        options=[
            "Titanic Passengers (sklearn / OpenML)",
            "Iris Flower Dataset (sklearn)",
            "Wine Recognition Dataset (sklearn)",
            "Breast Cancer Diagnostic (sklearn)",
            "California Housing Dataset (sklearn)",
        ],
        index=0,
        help="Select a generic example dataset to analyze.",
    )
    df_sample = load_sample_dataset(selected_sample)
    data_source = df_sample
    file_name = selected_sample.split(" (")[0] + ".csv"
else:
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV or Excel file",
        type=["csv", "xlsx", "xls", "tsv"],
        help="Upload your dataset in CSV or Excel format.",
    )
    if uploaded_file is not None:
        data_source = uploaded_file
        file_name = uploaded_file.name

# Optimal constant for inspecting top frequent values per column
TOP_N_VALUES = 10

if data_source is None:
    st.info("👆 Please select an example dataset or upload a CSV/Excel file from the sidebar to get started.")
else:
    with st.spinner(f"Extracting insights from '{file_name}'..."):
        try:
            # 1. Extract data information using DataInformationExtractor
            extractor = DataInformationExtractor(data_source)
            insights = extractor.extract_all(top_n_values=TOP_N_VALUES)

            if not insights.get("success", False) and (extractor.df is None or extractor.df.empty):
                st.error("Failed to extract data information. See details below:")
                for err in insights.get("errors", []):
                    st.error(f"- {err}")
            else:
                if insights.get("errors"):
                    for err in insights["errors"]:
                        st.warning(f"Warning: {err}")

                overview = insights.get("dataset_overview", {})
                row_count = overview.get("row_count", 0)
                col_count = overview.get("column_count", 0)
                dup_count = overview.get("duplicate_count", 0)
                total_nulls = sum(insights.get("null_count", {}).values())

                # Top Metrics Overview
                st.subheader(f"Dataset Overview: `{file_name}`")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Rows", f"{row_count:,}")
                m2.metric("Total Columns", f"{col_count:,}")
                m3.metric("Duplicate Rows", f"{dup_count:,}")
                m4.metric("Total Missing Values", f"{total_nulls:,}")

                st.divider()

                # Navigation Tabs
                tabs = st.tabs([
                    "🤖 AI Analytical Report",
                    "📊 Generated Charts",
                    "📋 Data Preview",
                    "ℹ️ Column Information",
                    "📈 Summary Statistics",
                    "⚠️ Missing Values",
                    "🔗 Correlations",
                    "📑 Grouped Aggregations",
                    "📦 Raw Insights (JSON / Dict)"
                ])

                # Tab 0: AI Generated Report via Fast LangGraph Orchestration
                with tabs[0]:
                    st.subheader("🤖 AI Executive Report Generator")
                    st.markdown(
                        "Generates a comprehensive executive analytical report with embedded visual charts "
                        "(Bar graphs, Correlation Heatmaps, Line trends, and Pie charts) powered by "
                        "**Matplotlib / Seaborn**, **LangGraph orchestration**, and **ReportLab PDF generation**."
                    )

                    report_session_key = f"report_{file_name}"
                    pdf_session_key = f"pdf_{file_name}"
                    charts_session_key = f"charts_{file_name}"

                    col_btn, col_down_pdf, col_down_md = st.columns([2, 1.5, 1.5])

                    with col_btn:
                        generate_clicked = st.button(
                            "🚀 Generate AI Analytical Report",
                            type="primary",
                            help="Click to generate charts and run LangGraph orchestration.",
                        )

                    if generate_clicked:
                        with st.spinner("📊 Deciding and rendering visual charts (matplotlib & seaborn)..."):
                            try:
                                chart_gen = ChartGenerator(output_dir=f"generated_charts_{Path(file_name).stem}")
                                dataset_charts = chart_gen.decide_and_generate_all(extractor.df)
                                st.session_state[charts_session_key] = dataset_charts
                            except Exception as chart_err:
                                dataset_charts = []
                                st.warning(f"Could not generate visual charts: {chart_err}")

                        with st.spinner("🤖 Generating executive report via LangGraph..."):
                            try:
                                orchestrator_instance = DataReportOrchestrator(graph=orchestrator_worker)
                                final_report = orchestrator_instance.generate_report(insights, charts=dataset_charts)
                                st.session_state[report_session_key] = final_report

                                # Build downloadable PDF with ReportLab
                                pdf_bytes = make_pdf_from_report(
                                    markdown_report=final_report,
                                    output_path=None,
                                    charts=dataset_charts
                                )
                                st.session_state[pdf_session_key] = pdf_bytes
                                st.success("✅ Executive Report & PDF generated successfully!")
                            except Exception as agent_err:
                                st.error(f"Error during AI report generation: {str(agent_err)}")

                    cached_report = st.session_state.get(report_session_key)
                    cached_pdf = st.session_state.get(pdf_session_key)
                    current_charts = st.session_state.get(charts_session_key, [])
                    chart_dir_to_clean = f"generated_charts_{Path(file_name).stem}"

                    if cached_report:
                        if cached_pdf:
                            with col_down_pdf:
                                st.download_button(
                                    label="📥 Download Report (.pdf)",
                                    data=cached_pdf,
                                    file_name=f"{Path(file_name).stem}_report.pdf",
                                    mime="application/pdf",
                                    type="primary",
                                    on_click=on_download_cleanup_action,
                                    args=(chart_dir_to_clean, current_charts),
                                    help="Download publication-quality PDF with charts and delete local chart files immediately."
                                )
                        with col_down_md:
                            st.download_button(
                                label="📥 Download Report (.md)",
                                data=cached_report,
                                file_name=f"{Path(file_name).stem}_report.md",
                                mime="text/markdown",
                                on_click=on_download_cleanup_action,
                                args=(chart_dir_to_clean, current_charts),
                                help="Download Markdown report and delete local chart files immediately."
                            )

                        st.markdown("---")
                        st.markdown(cached_report)
                    else:
                        st.info("Click the **'🚀 Generate AI Analytical Report'** button above to create the report.")

                # Tab 1: Generated Charts Gallery
                with tabs[1]:
                    st.subheader("📊 Visual Charts Gallery (Matplotlib & Seaborn)")
                    active_charts = st.session_state.get(charts_session_key, [])
                    if not active_charts and extractor.df is not None and not extractor.df.empty:
                        # Auto-preview charts even before full text report is clicked
                        try:
                            chart_gen = ChartGenerator(output_dir=f"generated_charts_{Path(file_name).stem}")
                            active_charts = chart_gen.decide_and_generate_all(extractor.df)
                            st.session_state[charts_session_key] = active_charts
                        except Exception as e:
                            st.warning(f"Could not render visual gallery: {e}")

                    if active_charts:
                        chart_cols = st.columns(2)
                        for idx, chart_info in enumerate(active_charts):
                            col_to_use = chart_cols[idx % 2]
                            with col_to_use:
                                st.markdown(f"#### {chart_info.get('title')}")
                                img_source = None
                                if chart_info.get("path") and Path(chart_info["path"]).exists():
                                    img_source = chart_info["path"]
                                elif chart_info.get("image_bytes"):
                                    img_source = chart_info["image_bytes"]

                                if img_source is not None:
                                    st.image(img_source, caption=chart_info.get("caption", ""), use_container_width=True)
                                else:
                                    st.info(f"Chart: {chart_info.get('title')}")
                    else:
                        st.info("No visual charts generated yet. Click 'Generate AI Analytical Report' or check dataset columns.")

                # Tab 2: Data Preview
                with tabs[2]:
                    st.subheader("Data Preview (First 50 Rows)")
                    if extractor.df is not None and not extractor.df.empty:
                        st.dataframe(extractor.df.head(50), use_container_width=True)
                    else:
                        st.write("No data available.")

                # Tab 3: Column Information
                with tabs[3]:
                    st.subheader("Column Metadata & Frequent Values")
                    df_info = insights.get("df_info", {})
                    col_names = df_info.get("col_names", [])
                    col_types = df_info.get("col_types", [])
                    col_values = df_info.get("col_values", [])

                    if col_names:
                        meta_rows = []
                        for i, name in enumerate(col_names):
                            freq_dict = col_values[i] if i < len(col_values) else {}
                            meta_rows.append({
                                "Column Name": name,
                                "Data Type": col_types[i] if i < len(col_types) else "Unknown",
                                "Top Frequent Values": json.dumps(freq_dict, ensure_ascii=False)
                            })
                        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True)
                    else:
                        st.write("No column information available.")

                # Tab 4: Summary Statistics (describe)
                with tabs[4]:
                    st.subheader("Statistical Summary (`describe`)")
                    describe_dict = insights.get("describe", {})
                    if describe_dict:
                        describe_df = pd.DataFrame(describe_dict)
                        st.dataframe(describe_df, use_container_width=True)
                    else:
                        st.info("No numeric summary statistics available.")

                # Tab 5: Missing Values
                with tabs[5]:
                    st.subheader("Missing / Null Value Counts")
                    null_counts = insights.get("null_count", {})
                    if null_counts:
                        null_df = pd.DataFrame(
                            list(null_counts.items()),
                            columns=["Column Name", "Missing Values Count"]
                        )
                        null_df["Missing Percentage (%)"] = (
                            (null_df["Missing Values Count"] / max(row_count, 1)) * 100
                        ).round(2)

                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.dataframe(null_df, use_container_width=True)
                        with c2:
                            filtered_nulls = null_df[null_df["Missing Values Count"] > 0]
                            if not filtered_nulls.empty:
                                st.bar_chart(filtered_nulls.set_index("Column Name")["Missing Values Count"])
                            else:
                                st.success("🎉 No missing values found in this dataset!")
                    else:
                        st.write("No missing value information available.")

                # Tab 6: Correlation Matrix
                with tabs[6]:
                    st.subheader("Numeric Correlations")
                    corr_dict = insights.get("correlation", {})
                    if corr_dict:
                        corr_df = pd.DataFrame(corr_dict)
                        st.dataframe(corr_df.style.background_gradient(cmap="coolwarm", vmin=-1, vmax=1), use_container_width=True)
                    else:
                        st.info("Not enough numeric columns (at least 2 required) to compute correlation matrix.")

                # Tab 7: Grouped Aggregations
                with tabs[7]:
                    st.subheader("Grouped Aggregations (Categorical vs Numeric)")
                    groupings = insights.get("groupings", [])
                    if groupings:
                        group_options = [
                            f"{g['categorical_column']} vs {g['numeric_column']}"
                            for g in groupings
                        ]
                        selected_group = st.selectbox(
                            "Select categorical & numeric column pair:",
                            options=group_options,
                        )
                        selected_idx = group_options.index(selected_group)
                        selected_records = groupings[selected_idx].get("records", [])

                        if selected_records:
                            st.dataframe(pd.DataFrame(selected_records), use_container_width=True)
                        else:
                            st.write("No records for this grouping.")
                    else:
                        st.info("No categorical and numeric column combinations available for grouped aggregations.")

                # Tab 8: Raw Dictionary / JSON Output
                with tabs[8]:
                    st.subheader("Extracted Dictionary (Raw Output)")
                    st.download_button(
                        label="📥 Download Insights as JSON",
                        data=json.dumps(insights, indent=2, default=str),
                        file_name=f"{Path(file_name).stem}_insights.json",
                        mime="application/json",
                    )
                    st.json(insights)

        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
