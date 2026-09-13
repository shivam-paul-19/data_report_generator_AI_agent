import json
import logging
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Dict, List, Optional, Union
from typing_extensions import TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()
logger = logging.getLogger(__name__)

groq_key = os.environ.get("GROQ_API_KEY", "").strip()
openai_key = (os.environ.get("OPEN_AI_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY', "")
if openai_key:
    os.environ['OPENAI_API_KEY'] = openai_key

# Smart LLM setup: prioritize high-throughput model with automatic fallback
primary_llm = None
fallback_llm = None

if openai_key:
    try:
        primary_llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=0.2)
    except Exception as e:
        logger.warning(f"Could not initialize OpenAI primary LLM: {e}")

if groq_key:
    try:
        # Available high-performance Groq models
        g_model = "openai/gpt-oss-20b"
        g_llm = ChatGroq(model=g_model, groq_api_key=groq_key, temperature=0.2)
        if primary_llm is None:
            primary_llm = g_llm
        else:
            fallback_llm = g_llm
    except Exception as e:
        logger.warning(f"Could not initialize Groq LLM: {e}")

if primary_llm is None:
    # Final fallback attempt
    primary_llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=groq_key, temperature=0.2)


def invoke_with_retry(
    callable_obj: Any,
    input_data: Any,
    fallback_obj: Optional[Any] = None,
    max_retries: int = 6,
    base_delay: float = 2.0
) -> Any:
    """
    Robust LLM invocation with automatic retry on rate limits (429),
    dynamic wait time extraction, exponential backoff with jitter,
    and automatic failover to fallback LLM.
    """
    current_callable = callable_obj
    for attempt in range(max_retries):
        try:
            return current_callable.invoke(input_data)
        except Exception as e:
            err_msg = str(e)
            is_rate_limit = (
                "429" in err_msg
                or "rate_limit" in err_msg.lower()
                or "tpm" in err_msg.lower()
                or "tokens per minute" in err_msg.lower()
            )

            if is_rate_limit:
                # Try to extract wait time from error message
                match = re.search(r"try again in ([\d\.]+)s", err_msg, re.IGNORECASE)
                if match:
                    wait_time = float(match.group(1)) + 1.5
                else:
                    wait_time = (base_delay * (2 ** attempt)) + random.uniform(0.5, 2.0)

                logger.warning(
                    f"Rate limit encountered. Waiting {wait_time:.1f}s before retry... (Attempt {attempt+1}/{max_retries})"
                )

                # If rate limited and fallback model is available, switch after first retry
                if attempt >= 1 and fallback_obj is not None and current_callable != fallback_obj:
                    logger.info("Switching to fallback LLM provider to bypass rate limits...")
                    current_callable = fallback_obj

                time.sleep(wait_time)
            else:
                if attempt == max_retries - 1:
                    if fallback_obj is not None and current_callable != fallback_obj:
                        logger.info("Final attempt: invoking fallback LLM...")
                        return fallback_obj.invoke(input_data)
                    raise e
                time.sleep(base_delay + random.uniform(0.2, 1.0))

    return current_callable.invoke(input_data)


llm = primary_llm


class State(TypedDict):
    topic: str
    charts: List[Dict[str, Any]]
    part1_architecture_health: str
    part2_analytics_strategy: str
    final_report: str


def node_architecture_and_health(state: State) -> Dict[str, Any]:
    """Pass 1: Executive Architecture, Domain Taxonomy, Data Health & Cleansing Blueprint."""
    topic_context = state.get("topic", "")
    charts = state.get("charts", [])

    charts_desc = "\n".join([
        f"- Chart Title: {c.get('title')} | Type: {c.get('type')} | Markdown Tag: `![{c.get('caption', c.get('title'))}]({c.get('path')})` | Suggested Section: {c.get('section')}"
        for c in charts if "Section 1" in c.get("section", "") or "Section 2" in c.get("section", "")
    ])
    charts_prompt = f"\n\nAvailable Generated Visual Charts to embed in Section 1 or 2:\n{charts_desc}\nEmbed these charts in their relevant subsections using the exact Markdown Tag provided above." if charts_desc else ""

    messages = [
        SystemMessage(
            content=(
                "You are an Executive Chief Data Officer and Lead Data Architect.\n"
                "Analyze the provided dataset statistics and produce the first half of the Executive Report covering Sections 1 and 2.\n\n"
                "Requirements:\n"
                "# 📊 Comprehensive Executive Data Intelligence & Analytical Report\n\n"
                "## 1. 📋 Executive Summary, Strategic Context & Architecture\n"
                "- Domain Overview: nature of data, scope, and strategic purpose.\n"
                "- Structural Metrics Table: Total Observations, Feature Dimensions, and Duplicate Counts.\n"
                "- Feature Taxonomy breakdown: Target, Continuous, Categorical, and Identifier attributes.\n"
                "- 4-5 high-level strategic takeaways summarizing the core findings and data health.\n"
                "- Embed relevant distribution / pie charts if provided.\n\n"
                "## 2. 🔍 Data Health, Missingness Mechanisms & Cleansing Blueprint\n"
                "- Column-by-column missing value audit table with exact counts, percentages, and missingness mechanism (MCAR / MAR / MNAR).\n"
                "- Evaluation of duplicate rows, data types, cardinality, and format integrity.\n"
                "- Actionable, step-by-step Cleansing & Preprocessing protocol.\n"
                "- Embed missing value bar chart if provided.\n\n"
                "Format strictly in clean GitHub markdown with exact figures from the statistics."
            )
        ),
        HumanMessage(
            content=f"Dataset Full Statistics & Metadata:\n{topic_context}{charts_prompt}\n\nGenerate Sections 1 and 2 now in markdown with embedded charts."
        )
    ]

    resp = invoke_with_retry(llm, messages, fallback_obj=fallback_llm)
    content = resp.content if hasattr(resp, "content") else str(resp)
    return {"part1_architecture_health": content}


def node_analytics_and_strategy(state: State) -> Dict[str, Any]:
    """Pass 2: Statistical Distributions, Correlations, Group Dynamics & Strategic ML Roadmap."""
    topic_context = state.get("topic", "")
    charts = state.get("charts", [])

    charts_desc = "\n".join([
        f"- Chart Title: {c.get('title')} | Type: {c.get('type')} | Markdown Tag: `![{c.get('caption', c.get('title'))}]({c.get('path')})` | Suggested Section: {c.get('section')}"
        for c in charts if "Section 3" in c.get("section", "") or "Section 4" in c.get("section", "") or "Section 5" in c.get("section", "")
    ])
    charts_prompt = f"\n\nAvailable Generated Visual Charts to embed in Section 3, 4, or 5:\n{charts_desc}\nEmbed these charts in their relevant subsections using the exact Markdown Tag provided above." if charts_desc else ""

    messages = [
        SystemMessage(
            content=(
                "You are a Principal Statistician and Chief AI Strategist.\n"
                "Analyze the provided dataset statistics and produce the second half of the Executive Report covering Sections 3, 4, and 5.\n\n"
                "Requirements:\n"
                "## 3. 📈 In-Depth Statistical Profiling, Distributions & Correlations\n"
                "- Detailed numerical breakdown (means, medians, IQR, min/max spreads, skewness).\n"
                "- Correlation Matrix dissection: highlight significant positive/negative associations, collinearity risks, and target relationships.\n"
                "- Outlier detection and feature scaling/transformation advice.\n"
                "- Embed feature correlation heatmap and trend/line charts if provided.\n\n"
                "## 4. 🔬 Cross-Tabulation & Sub-Population Dynamics\n"
                "- Detailed evaluation of Grouped Aggregations (categorical splits vs numeric metrics: mean, median, min, max, count).\n"
                "- Subgroup comparison tables displaying key disparities, demographic patterns, and cohort behaviors.\n"
                "- Embed grouped bar charts if provided.\n\n"
                "## 5. 💡 Strategic Conclusions, Machine Learning Roadmap & Action Matrix\n"
                "- Conclusive findings and business/operational takeaways.\n"
                "- Feature Engineering Playbook (derived indicators, binning, encoding).\n"
                "- Predictive Modeling Strategy (recommended algorithms, validation strategy, metrics).\n"
                "- Risk & Caveats Matrix followed by a Priority Action Matrix (High / Medium / Low).\n\n"
                "Format strictly in clean GitHub markdown with exact figures and structured tables."
            )
        ),
        HumanMessage(
            content=f"Dataset Full Statistics & Metadata:\n{topic_context}{charts_prompt}\n\nGenerate Sections 3, 4, and 5 now in markdown with embedded charts."
        )
    ]

    resp = invoke_with_retry(llm, messages, fallback_obj=fallback_llm)
    content = resp.content if hasattr(resp, "content") else str(resp)
    return {"part2_analytics_strategy": content}


def node_synthesizer(state: State) -> Dict[str, Any]:
    """Synthesizer: Unifies the comprehensive sections and guarantees all generated charts are embedded."""
    p1 = state.get("part1_architecture_health", "")
    p2 = state.get("part2_analytics_strategy", "")
    charts = state.get("charts", [])
    full_report = f"{p1}\n\n---\n\n{p2}"

    # Verify that each generated chart is embedded in the report; if missing, append gracefully
    missing_chart_tags = []
    for c in charts:
        img_path = str(c.get("path", ""))
        if img_path and img_path not in full_report and Path(img_path).name not in full_report:
            caption = c.get("caption", c.get("title", "Analytical Visualization"))
            missing_chart_tags.append(f"### 📊 {c.get('title', 'Analytical Chart')}\n![{caption}]({img_path})\n\n*{caption}*\n")

    if missing_chart_tags:
        full_report += "\n\n---\n\n## 📊 Analytical Visualizations & Chart Catalog\n\n" + "\n\n".join(missing_chart_tags)

    return {"final_report": full_report}


# Build LangGraph StateGraph
orchestrator_worker_builder = StateGraph(State)

orchestrator_worker_builder.add_node("ArchitectureHealthWorker", node_architecture_and_health)
orchestrator_worker_builder.add_node("AnalyticsStrategyWorker", node_analytics_and_strategy)
orchestrator_worker_builder.add_node("Synthesizer", node_synthesizer)

orchestrator_worker_builder.add_edge(START, "ArchitectureHealthWorker")
orchestrator_worker_builder.add_edge("ArchitectureHealthWorker", "AnalyticsStrategyWorker")
orchestrator_worker_builder.add_edge("AnalyticsStrategyWorker", "Synthesizer")
orchestrator_worker_builder.add_edge("Synthesizer", END)

orchestrator_worker = orchestrator_worker_builder.compile()


class DataReportOrchestrator:
    """
    Class to orchestrate report generation using the compiled LangGraph orchestrator_worker.
    Can be imported and invoked from app.py or other modules.
    """

    def __init__(self, graph=orchestrator_worker):
        self.graph = graph

    @staticmethod
    def format_insights_to_prompt(insights: Union[Dict[str, Any], str]) -> str:
        """
        Formats insights into an information-dense, token-efficient prompt.
        """
        if isinstance(insights, str):
            return insights

        overview = insights.get("dataset_overview", {})
        df_info = insights.get("df_info", {})
        describe = insights.get("describe", {})
        null_count = insights.get("null_count", {})
        correlation = insights.get("correlation", {})
        groupings = insights.get("groupings", [])
        row_count = overview.get("row_count", 1) or 1

        parts = [
            f"### Dataset High-Level Overview\n"
            f"- Total Rows: {overview.get('row_count', 'N/A')}\n"
            f"- Total Columns: {overview.get('column_count', 'N/A')}\n"
            f"- Exact Duplicate Rows: {overview.get('duplicate_count', 'N/A')}\n"
        ]

        col_names = df_info.get("col_names", [])
        col_types = df_info.get("col_types", [])
        col_values = df_info.get("col_values", [])
        if col_names:
            cols_info = []
            for i, col in enumerate(col_names):
                c_type = col_types[i] if i < len(col_types) else "unknown"
                sample_vals = col_values[i] if i < len(col_values) else {}
                null_val = null_count.get(col, 0)
                null_pct = round((null_val / row_count) * 100, 1)
                top_items = list(sample_vals.items())[:4]
                dist_str = ", ".join([f"{k}: {v}" for k, v in top_items]) if top_items else "N/A"
                cols_info.append(f"- `{col}` ({c_type}) | Nulls: {null_val} ({null_pct}%) | Top: [{dist_str}]")
            parts.append("### Complete Column Schema & Distributions\n" + "\n".join(cols_info))

        if null_count:
            parts.append(f"### Missing Values\n{json.dumps(null_count, indent=2)}")

        if describe:
            parts.append(f"### Descriptive Statistics (describe)\n{json.dumps(describe, default=str, indent=2)}")

        if correlation:
            parts.append(f"### Correlation Matrix\n{json.dumps(correlation, default=str, indent=2)}")

        if groupings:
            parts.append(f"### Grouped Aggregations (Key Cohorts)\n{json.dumps(groupings[:10], default=str, indent=2)}")

        return "\n\n".join(parts)

    def generate_report(
        self,
        data_insights: Union[Dict[str, Any], str],
        charts: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        topic_text = self.format_insights_to_prompt(data_insights)
        initial_state: State = {
            "topic": topic_text,
            "charts": charts or [],
            "part1_architecture_health": "",
            "part2_analytics_strategy": "",
            "final_report": "",
        }
        result = self.graph.invoke(initial_state)
        return result.get("final_report", "")