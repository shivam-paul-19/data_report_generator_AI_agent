import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend suitable for server/agent environments
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Set default aesthetic theme using seaborn and matplotlib
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "figure.titlesize": 14,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.autolayout": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


class ChartGenerator:
    """
    Automated and intelligent chart generation tool using matplotlib and seaborn.
    Decides and renders necessary Bar Graphs, Correlation Heatmaps, Line Graphs,
    and Pie Charts based on dataset properties and agent guidance.
    """

    def __init__(self, output_dir: str = "generated_charts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        filename: str = "correlation_heatmap.png",
        annot: bool = True,
        max_cols: int = 15
    ) -> Optional[str]:
        """
        Generate a correlation heatmap for numeric features using Seaborn.
        """
        try:
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.shape[1] < 2:
                return None

            # If there are too many columns, pick the top columns by variance
            if numeric_df.shape[1] > max_cols:
                variances = numeric_df.var().sort_values(ascending=False)
                top_cols = variances.head(max_cols).index.tolist()
                numeric_df = numeric_df[top_cols]

            corr = numeric_df.corr()

            fig_width = max(8, min(14, numeric_df.shape[1] * 0.8))
            fig_height = max(6, min(12, numeric_df.shape[1] * 0.7))

            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            mask = np.triu(np.ones_like(corr, dtype=bool))
            
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            sns.heatmap(
                corr,
                mask=mask,
                cmap=cmap,
                vmax=1.0,
                vmin=-1.0,
                center=0,
                square=True,
                linewidths=0.75,
                cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
                annot=annot if numeric_df.shape[1] <= 12 else False,
                fmt=".2f",
                ax=ax,
            )
            ax.set_title("Correlation Heatmap (Feature Associations)", fontsize=13, weight="bold", pad=15)
            
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate correlation heatmap: {e}")
            plt.close("all")
            return None

    def create_missing_values_bar(
        self,
        df: pd.DataFrame,
        filename: str = "missing_values_bar.png"
    ) -> Optional[str]:
        """
        Generate a missing values bar chart using Seaborn if missing values exist.
        """
        try:
            null_counts = df.isna().sum()
            missing = null_counts[null_counts > 0].sort_values(ascending=False)
            if missing.empty:
                return None

            missing_df = pd.DataFrame({
                "Feature": missing.index,
                "MissingCount": missing.values,
                "MissingPct": (missing.values / len(df)) * 100
            })

            fig, ax = plt.subplots(figsize=(max(8, len(missing_df) * 0.7), 5))
            bars = sns.barplot(
                data=missing_df,
                x="Feature",
                y="MissingCount",
                hue="Feature",
                palette="Reds_r",
                legend=False,
                ax=ax
            )
            
            # Add percentage labels above bars
            for bar, pct in zip(bars.patches, missing_df["MissingPct"]):
                height = bar.get_height()
                ax.annotate(
                    f"{int(height)} ({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    fontweight="semibold"
                )

            ax.set_title("Missing Values Audit (Counts & Percentage)", fontsize=13, weight="bold", pad=12)
            ax.set_xlabel("Columns / Features", fontsize=10, weight="semibold")
            ax.set_ylabel("Null Count", fontsize=10, weight="semibold")
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")

            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate missing values bar chart: {e}")
            plt.close("all")
            return None

    def create_pie_chart(
        self,
        df: pd.DataFrame,
        column: str,
        filename: Optional[str] = None,
        max_slices: int = 6
    ) -> Optional[str]:
        """
        Generate a pie chart using Matplotlib for categorical columns with low cardinality.
        """
        try:
            if column not in df.columns:
                return None

            val_counts = df[column].dropna().value_counts()
            if len(val_counts) < 2 or len(val_counts) > max_slices:
                return None

            if filename is None:
                safe_col = "".join(c if c.isalnum() else "_" for c in str(column))
                filename = f"pie_chart_{safe_col}.png"

            fig, ax = plt.subplots(figsize=(6, 6))
            colors = sns.color_palette("pastel", len(val_counts))
            
            wedges, texts, autotexts = ax.pie(
                val_counts.values,
                labels=[str(l) for l in val_counts.index],
                autopct="%1.1f%%",
                startangle=140,
                colors=colors,
                wedgeprops=dict(width=0.8, edgecolor="white", linewidth=1.5),
                textprops=dict(color="black", fontsize=9)
            )
            for autotext in autotexts:
                autotext.set_color("black")
                autotext.set_fontsize(9)
                autotext.set_weight("bold")

            ax.set_title(f"Composition Breakdown: {column}", fontsize=13, weight="bold", pad=12)

            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate pie chart for {column}: {e}")
            plt.close("all")
            return None

    def create_bar_chart(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: Optional[str] = None,
        filename: Optional[str] = None,
        top_n: int = 10,
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a bar chart for categorical frequency or grouped numeric aggregation.
        """
        try:
            if x_col not in df.columns:
                return None

            safe_x = "".join(c if c.isalnum() else "_" for c in str(x_col))
            safe_y = "".join(c if c.isalnum() else "_" for c in str(y_col)) if y_col else "freq"
            if filename is None:
                filename = f"bar_chart_{safe_x}_{safe_y}.png"

            fig, ax = plt.subplots(figsize=(8, 5))

            if y_col and y_col in df.columns and pd.api.types.is_numeric_dtype(df[y_col]):
                # Grouped mean aggregation
                grouped = df.groupby(x_col, observed=False)[y_col].mean().dropna().sort_values(ascending=False).head(top_n)
                plot_data = pd.DataFrame({"Category": [str(i) for i in grouped.index], "MeanValue": grouped.values})
                sns.barplot(data=plot_data, x="Category", y="MeanValue", hue="Category", palette="Blues_r", legend=False, ax=ax)
                ax.set_ylabel(f"Average {y_col}", fontsize=10, weight="semibold")
                default_title = f"Average {y_col} across {x_col}"
            else:
                # Frequency distribution
                val_counts = df[x_col].value_counts().head(top_n)
                plot_data = pd.DataFrame({"Category": [str(i) for i in val_counts.index], "Count": val_counts.values})
                sns.barplot(data=plot_data, x="Category", y="Count", hue="Category", palette="viridis", legend=False, ax=ax)
                ax.set_ylabel("Count / Frequency", fontsize=10, weight="semibold")
                default_title = f"Top {top_n} Categories: {x_col}"

            ax.set_title(title or default_title, fontsize=12, weight="bold", pad=12)
            ax.set_xlabel(str(x_col), fontsize=10, weight="semibold")
            plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate bar chart for {x_col}: {e}")
            plt.close("all")
            return None

    def create_line_chart(
        self,
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        filename: Optional[str] = None,
        title: Optional[str] = None
    ) -> Optional[str]:
        """
        Generate a line chart for continuous trends, ordered series, or sequential distributions.
        """
        try:
            if x_col not in df.columns or y_col not in df.columns:
                return None

            clean_df = df[[x_col, y_col]].dropna()
            if clean_df.empty:
                return None

            safe_x = "".join(c if c.isalnum() else "_" for c in str(x_col))
            safe_y = "".join(c if c.isalnum() else "_" for c in str(y_col))
            if filename is None:
                filename = f"line_chart_{safe_x}_{safe_y}.png"

            fig, ax = plt.subplots(figsize=(8, 5))

            # If categorical or few unique values, compute mean trend
            if clean_df[x_col].nunique() < 50:
                trend = clean_df.groupby(x_col, observed=False)[y_col].mean().reset_index()
                sns.lineplot(data=trend, x=x_col, y=y_col, marker="o", color="#1f77b4", linewidth=2.2, ax=ax)
            else:
                # Continuous line or rolling trend
                sorted_df = clean_df.sort_values(by=x_col).head(200)
                sns.lineplot(data=sorted_df, x=x_col, y=y_col, color="#1f77b4", linewidth=1.8, ax=ax)

            default_title = f"Trend & Relationship: {y_col} vs {x_col}"
            ax.set_title(title or default_title, fontsize=12, weight="bold", pad=12)
            ax.set_xlabel(str(x_col), fontsize=10, weight="semibold")
            ax.set_ylabel(str(y_col), fontsize=10, weight="semibold")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to generate line chart for {x_col} vs {y_col}: {e}")
            plt.close("all")
            return None

    def _is_id_column(self, col_name: str) -> bool:
        """Check if column name indicates an identifier or index column."""
        lower = str(col_name).lower()
        return any(term in lower for term in ["seqn", "id", "index", "unnamed", "row_id", "record_id"])

    def decide_and_generate_all(
        self,
        df: pd.DataFrame,
        max_charts: int = 6
    ) -> List[Dict[str, Any]]:
        """
        Intelligently analyzes the DataFrame and decides which charts (Heatmap, Bar graphs,
        Line graphs, Pie charts) are most necessary and informative.
        
        Returns a list of dictionaries with chart metadata and file paths.
        """
        generated_charts: List[Dict[str, Any]] = []
        if df is None or df.empty:
            return generated_charts

        # 1. Missing Values Bar Chart (if nulls exist)
        if df.isna().sum().sum() > 0:
            missing_chart = self.create_missing_values_bar(df, filename="chart_1_missing_values.png")
            if missing_chart:
                generated_charts.append({
                    "id": "missing_values_chart",
                    "type": "bar_chart",
                    "title": "Missing Values Audit",
                    "path": missing_chart,
                    "section": "Section 2: Data Health & Cleansing Blueprint",
                    "caption": "Audit of missing values across dataset features showing total counts and percentages."
                })

        # 2. Filter out pure ID columns from analytic features
        all_num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        num_cols = [c for c in all_num_cols if not self._is_id_column(c)]
        if not num_cols:
            num_cols = all_num_cols

        # Correlation Heatmap (for numeric features)
        if len(num_cols) >= 2:
            heatmap_path = self.create_correlation_heatmap(df[num_cols], filename="chart_2_correlation_heatmap.png")
            if heatmap_path:
                generated_charts.append({
                    "id": "correlation_heatmap",
                    "type": "heatmap",
                    "title": "Feature Correlation Heatmap",
                    "path": heatmap_path,
                    "section": "Section 3: In-Depth Statistical Profiling, Distributions & Correlations",
                    "caption": "Pearson correlation coefficients highlighting multicollinearity and feature interactions."
                })

        # 3. Categorical breakdown (Pie Chart for low cardinality / Target column, Bar Chart for moderate cardinality)
        all_cat_cols = df.select_dtypes(include=["object", "category", "string", "bool"]).columns.tolist()
        cat_cols = [c for c in all_cat_cols if not self._is_id_column(c)]
        # Also check discrete numeric columns (e.g. binary/multiclass target like MetabolicSyndrome 0/1)
        discrete_num_cols = [c for c in num_cols if 2 <= df[c].nunique() <= 6]
        all_cat_candidates = cat_cols + discrete_num_cols

        pie_created = False
        for col in all_cat_candidates:
            nunique = df[col].nunique()
            if 2 <= nunique <= 5 and not pie_created:
                safe_col = "".join(c if c.isalnum() else "_" for c in str(col))
                pie_path = self.create_pie_chart(df, column=col, filename=f"chart_3_pie_{safe_col}.png")
                if pie_path:
                    generated_charts.append({
                        "id": f"pie_{safe_col}",
                        "type": "pie_chart",
                        "title": f"Distribution of {col}",
                        "path": pie_path,
                        "section": "Section 1: Executive Summary, Strategic Context & Architecture",
                        "caption": f"Proportional composition and class balance breakdown for `{col}`."
                    })
                    pie_created = True
                    break

        # 4. Grouped Comparison Bar Chart (Categorical vs Meaningful Numeric feature)
        bar_created = False
        for cat_col in cat_cols:
            if 2 <= df[cat_col].nunique() <= 15:
                # Find most relevant numeric column (non-ID)
                if num_cols:
                    # Pick numeric col with high variance or first non-ID numeric
                    target_num = num_cols[-1] if len(num_cols) > 1 else num_cols[0]
                    safe_cat = "".join(c if c.isalnum() else "_" for c in str(cat_col))
                    safe_num = "".join(c if c.isalnum() else "_" for c in str(target_num))
                    bar_path = self.create_bar_chart(
                        df,
                        x_col=cat_col,
                        y_col=target_num,
                        filename=f"chart_4_group_bar_{safe_cat}_{safe_num}.png"
                    )
                    if bar_path:
                        generated_charts.append({
                            "id": f"bar_{safe_cat}_{safe_num}",
                            "type": "bar_chart",
                            "title": f"Mean {target_num} by {cat_col}",
                            "path": bar_path,
                            "section": "Section 4: Cross-Tabulation & Sub-Population Dynamics",
                            "caption": f"Comparative breakdown of mean `{target_num}` across `{cat_col}` cohorts."
                        })
                        bar_created = True
                        break

        # 5. Line Graph / Continuous distribution or bivariate trend
        if len(num_cols) >= 2 and len(generated_charts) < max_charts:
            x_col = num_cols[0]
            y_col = num_cols[1]
            safe_x = "".join(c if c.isalnum() else "_" for c in str(x_col))
            safe_y = "".join(c if c.isalnum() else "_" for c in str(y_col))
            line_path = self.create_line_chart(
                df,
                x_col=x_col,
                y_col=y_col,
                filename=f"chart_5_line_{safe_x}_{safe_y}.png"
            )
            if line_path:
                generated_charts.append({
                    "id": f"line_{safe_x}_{safe_y}",
                    "type": "line_chart",
                    "title": f"Trend Analysis: {y_col} vs {x_col}",
                    "path": line_path,
                    "section": "Section 3: In-Depth Statistical Profiling, Distributions & Correlations",
                    "caption": f"Bivariate trend trajectory evaluating interaction between `{x_col}` and `{y_col}`."
                })

        # Pre-cache image bytes into memory so charts can be displayed / converted to PDF
        # even after the physical files are removed upon download
        for c in generated_charts:
            p = Path(c.get("path", ""))
            if p.exists():
                try:
                    c["image_bytes"] = p.read_bytes()
                except Exception:
                    pass

        return generated_charts

    def cleanup_all(self) -> None:
        """Deletes all generated chart files in the output directory."""
        cleanup_chart_directory(self.output_dir)


def cleanup_chart_directory(dir_path: Union[str, Path]) -> None:
    """Safely deletes all image files and the chart folder from disk."""
    try:
        p = Path(dir_path)
        if p.exists() and p.is_dir():
            for f in p.glob("*.png"):
                try:
                    f.unlink(missing_ok=True)
                except Exception:
                    pass
            try:
                p.rmdir()
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Error cleaning up chart directory {dir_path}: {e}")


def cleanup_chart_files(charts: Optional[List[Dict[str, Any]]] = None) -> None:
    """Deletes specific chart image files from disk immediately."""
    if not charts:
        return
    for c in charts:
        p_str = c.get("path")
        if p_str:
            try:
                p = Path(p_str)
                if p.exists():
                    p.unlink(missing_ok=True)
            except Exception as e:
                logger.warning(f"Error deleting chart file {p_str}: {e}")

