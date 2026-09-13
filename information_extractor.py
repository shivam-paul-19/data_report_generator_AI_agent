import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

# Set up logger for information extractor
logger = logging.getLogger(__name__)


class DataInformationExtractor:
    """
    A generic and scalable extractor for extracting statistical insights,
    structural metadata, correlations, and grouped aggregations from CSV or Excel datasets.
    """

    def __init__(
        self,
        data_source: Union[str, Path, pd.DataFrame, Any],
        sheet_name: Optional[Union[str, int]] = 0,
        **read_kwargs: Any
    ):
        """
        Initialize the DataInformationExtractor.

        Parameters
        ----------
        data_source : str, Path, pd.DataFrame, or file-like buffer
            The input data source. Can be a path to a .csv or .xlsx file,
            an in-memory file buffer, or an existing pandas DataFrame.
        sheet_name : str or int, optional
            Sheet name or index to read when loading Excel files. Default is 0.
        **read_kwargs : Any
            Additional keyword arguments passed to pd.read_csv or pd.read_excel.
        """
        self.data_source = data_source
        self.sheet_name = sheet_name
        self.read_kwargs = read_kwargs
        self.df: Optional[pd.DataFrame] = None
        self.errors: List[str] = []

        self._load_data()

    def _load_data(self) -> None:
        """
        Load data into a pandas DataFrame from a CSV or Excel file, buffer, or existing DataFrame.
        """
        try:
            if isinstance(self.data_source, pd.DataFrame):
                self.df = self.data_source.copy()
                return

            if isinstance(self.data_source, (str, Path)):
                file_path = Path(self.data_source)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: '{file_path}'")

                suffix = file_path.suffix.lower()
                if suffix in [".csv", ".txt", ".tsv"]:
                    sep = "\t" if suffix == ".tsv" else self.read_kwargs.pop("sep", ",")
                    self.df = pd.read_csv(file_path, sep=sep, **self.read_kwargs)
                elif suffix in [".xlsx", ".xls", ".xlsm", ".xlsb"]:
                    self.df = pd.read_excel(file_path, sheet_name=self.sheet_name, **self.read_kwargs)
                else:
                    # Fallback attempt: try csv first, then excel
                    try:
                        self.df = pd.read_csv(file_path, **self.read_kwargs)
                    except Exception:
                        self.df = pd.read_excel(file_path, sheet_name=self.sheet_name, **self.read_kwargs)
                return

            # If it's a file-like object / buffer (e.g., BytesIO, StringIO, UploadedFile)
            if hasattr(self.data_source, "read"):
                file_name = getattr(self.data_source, "name", "").lower()
                if file_name.endswith((".xlsx", ".xls", ".xlsm", ".xlsb")):
                    self.df = pd.read_excel(self.data_source, sheet_name=self.sheet_name, **self.read_kwargs)
                else:
                    try:
                        self.df = pd.read_csv(self.data_source, **self.read_kwargs)
                    except Exception:
                        # Reset pointer if seekable and try excel
                        if hasattr(self.data_source, "seek"):
                            self.data_source.seek(0)
                        self.df = pd.read_excel(self.data_source, sheet_name=self.sheet_name, **self.read_kwargs)
                return

            raise TypeError(
                f"Unsupported data source type: {type(self.data_source)}. "
                "Expected file path (str/Path), file-like object, or pd.DataFrame."
            )

        except Exception as e:
            err_msg = f"Failed to load dataset: {str(e)}"
            logger.error(err_msg)
            self.errors.append(err_msg)
            self.df = pd.DataFrame()

    def get_df_info(self, top_n_values: int = 10) -> Dict[str, Any]:
        """
        Extract column names, data types, and top frequent values for each column.

        Parameters
        ----------
        top_n_values : int, optional
            Number of top frequent values to return for each column. Default is 10.

        Returns
        -------
        dict
            Dictionary containing col_names, col_types, and col_values.
        """
        if self.df is None or self.df.empty:
            return {"col_names": [], "col_types": [], "col_values": []}

        try:
            col_name = []
            col_type = []
            col_values = []

            for col in self.df.columns:
                col_name.append(str(col))
                col_type.append(str(self.df[col].dtype))
                col_values.append(self.df[col].value_counts().head(top_n_values).to_dict())

            return {
                "col_names": col_name,
                "col_types": col_type,
                "col_values": col_values,
            }
        except Exception as e:
            err_msg = f"Error in get_df_info: {str(e)}"
            logger.error(err_msg)
            self.errors.append(err_msg)
            return {"col_names": [], "col_types": [], "col_values": [], "error": err_msg}

    def get_describe(self, include_all: bool = False) -> Dict[str, Any]:
        """
        Extract descriptive statistical summary of the DataFrame.

        Parameters
        ----------
        include_all : bool, optional
            Whether to include all columns or numeric only. Default is False (numeric only).

        Returns
        -------
        dict
            Summary statistics dictionary.
        """
        if self.df is None or self.df.empty:
            return {}

        try:
            if include_all:
                return self.df.describe(include="all").to_dict()
            return self.df.describe().to_dict()
        except Exception as e:
            err_msg = f"Error in get_describe: {str(e)}"
            logger.error(err_msg)
            self.errors.append(err_msg)
            return {"error": err_msg}

    def get_null_counts(self) -> Dict[str, int]:
        """
        Extract missing (null/NaN) values count for each column.

        Returns
        -------
        dict
            Dictionary mapping column names to missing value counts.
        """
        if self.df is None or self.df.empty:
            return {}

        try:
            return {str(col): int(count) for col, count in self.df.isna().sum().to_dict().items()}
        except Exception as e:
            err_msg = f"Error in get_null_counts: {str(e)}"
            logger.error(err_msg)
            self.errors.append(err_msg)
            return {"error": err_msg}

    def get_correlations(self) -> Dict[str, Dict[str, float]]:
        """
        Extract correlation matrix for numeric columns.

        Returns
        -------
        dict
            Correlation matrix dictionary.
        """
        if self.df is None or self.df.empty:
            return {}

        try:
            numeric_df = self.df.select_dtypes(include=["number"])
            if numeric_df.shape[1] < 2:
                return {}
            return numeric_df.corr(numeric_only=True).to_dict()
        except Exception as e:
            err_msg = f"Error in get_correlations: {str(e)}"
            logger.error(err_msg)
            self.errors.append(err_msg)
            return {"error": err_msg}

    def get_groupings(
        self,
        aggregations: Optional[List[str]] = None,
        max_cardinality: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Extract aggregated statistics (mean, median, min, max, count) grouped
        by non-numeric / categorical columns against numeric columns.

        Parameters
        ----------
        aggregations : list of str, optional
            List of aggregation functions to apply. Defaults to ['mean', 'median', 'min', 'max', 'count'].
        max_cardinality : int, optional
            Maximum unique categories per column to include in grouping analysis. Default is 30.

        Returns
        -------
        list of dict
            List of grouped aggregation records for each (categorical, numeric) column pair.
        """
        if self.df is None or self.df.empty:
            return []

        if aggregations is None:
            aggregations = ["mean", "median", "min", "max", "count"]

        try:
            # Select categorical/non-numeric columns with meaningful cardinality
            non_num_columns = [
                col for col in self.df.select_dtypes(
                    include=["object", "category", "string", "bool"]
                ).columns
                if self.df[col].nunique() <= max_cardinality
            ]
            num_columns = self.df.select_dtypes(include=["number"]).columns.tolist()

            results = []
            for non_num_col in non_num_columns:
                for num_col in num_columns:
                    try:
                        grouped_df = (
                            self.df.groupby(non_num_col, observed=False)[num_col]
                            .agg(aggregations)
                            .reset_index()
                        )
                        results.append({
                            "categorical_column": str(non_num_col),
                            "numeric_column": str(num_col),
                            "records": grouped_df.to_dict(orient="records"),
                        })
                    except Exception as inner_err:
                        logger.warning(
                            f"Could not compute grouping for {non_num_col} vs {num_col}: {inner_err}"
                        )

            return results
        except Exception as e:
            err_msg = f"Error in get_groupings: {str(e)}"
            logger.error(err_msg)
            self.errors.append(err_msg)
            return [{"error": err_msg}]

    def get_dataset_overview(self) -> Dict[str, Any]:
        """
        Extract high-level overview such as row count, column count, and duplicate count.

        Returns
        -------
        dict
            Dictionary containing row_count, column_count, and duplicate_count.
        """
        if self.df is None or self.df.empty:
            return {"row_count": 0, "column_count": 0, "duplicate_count": 0}

        try:
            return {
                "row_count": int(self.df.shape[0]),
                "column_count": int(self.df.shape[1]),
                "duplicate_count": int(self.df.duplicated().sum()),
            }
        except Exception as e:
            err_msg = f"Error in get_dataset_overview: {str(e)}"
            logger.error(err_msg)
            self.errors.append(err_msg)
            return {"error": err_msg}

    def extract_all(self, top_n_values: int = 10) -> Dict[str, Any]:
        """
        Extract all insights into a single comprehensive dictionary.

        Parameters
        ----------
        top_n_values : int, optional
            Number of top frequent values to return for each column. Default is 10.

        Returns
        -------
        dict
            Dictionary containing all extracted insights and metadata.
        """
        if self.errors and (self.df is None or self.df.empty):
            return {
                "success": False,
                "errors": self.errors,
                "dataset_overview": {},
                "df_info": {},
                "describe": {},
                "null_count": {},
                "correlation": {},
                "groupings": [],
            }

        return {
            "success": len(self.errors) == 0,
            "errors": self.errors,
            "dataset_overview": self.get_dataset_overview(),
            "df_info": self.get_df_info(top_n_values=top_n_values),
            "describe": self.get_describe(),
            "null_count": self.get_null_counts(),
            "correlation": self.get_correlations(),
            "groupings": self.get_groupings(),
        }

    def generate_report(
        self,
        top_n_values: int = 10,
        orchestrator: Optional[Any] = None,
        generate_charts: bool = True,
        output_pdf_path: Optional[str] = None,
        **kwargs: Any
    ) -> Union[str, Tuple[str, Optional[str]]]:
        """
        Extract all dataset insights, generate visual charts using Matplotlib & Seaborn,
        and invoke the LangGraph orchestrator_worker workflow to generate a comprehensive AI analytical report.
        Optionally compiles the final report and charts into a PDF using pdf_maker.py.

        Parameters
        ----------
        top_n_values : int, optional
            Number of top frequent values to extract per column. Default is 10.
        orchestrator : DataReportOrchestrator or StateGraph or Any, optional
            Custom orchestrator or LangGraph graph instance. If None, uses default DataReportOrchestrator(orchestrator_worker).
        generate_charts : bool, optional
            Whether to generate bar graphs, correlation heatmap, line graphs, and pie charts. Default is True.
        output_pdf_path : str, optional
            If provided, saves the report and charts as a PDF file.

        Returns
        -------
        str or (str, str)
            The generated markdown analytical report (and optional PDF path if requested).
        """
        insights = self.extract_all(top_n_values=top_n_values)
        charts = []
        if generate_charts and self.df is not None and not self.df.empty:
            try:
                from chart_generator import ChartGenerator
                chart_gen = ChartGenerator()
                charts = chart_gen.decide_and_generate_all(self.df)
            except Exception as chart_err:
                logger.warning(f"Could not generate charts: {chart_err}")

        final_md = ""
        if orchestrator is None:
            try:
                from agent import DataReportOrchestrator, orchestrator_worker
                orchestrator_instance = DataReportOrchestrator(graph=orchestrator_worker)
                final_md = orchestrator_instance.generate_report(insights, charts=charts)
            except Exception as e:
                err_msg = f"Failed to invoke orchestrator_worker: {str(e)}"
                logger.error(err_msg)
                final_md = f"### Error Generating Report\n\n{err_msg}"
        elif hasattr(orchestrator, "generate_report"):
            final_md = orchestrator.generate_report(insights, charts=charts)
        elif hasattr(orchestrator, "invoke"):
            from agent import DataReportOrchestrator
            orch = DataReportOrchestrator(graph=orchestrator)
            final_md = orch.generate_report(insights, charts=charts)
        else:
            raise TypeError(f"Invalid orchestrator type: {type(orchestrator)}. Expected DataReportOrchestrator or LangGraph graph.")

        pdf_path = None
        if output_pdf_path:
            try:
                from pdf_maker import make_pdf_from_report
                pdf_path = make_pdf_from_report(final_md, output_path=output_pdf_path, charts=charts)
            except Exception as pdf_err:
                logger.error(f"Failed to generate PDF: {pdf_err}")

        if output_pdf_path:
            return final_md, pdf_path
        return final_md

    def invoke_orchestrator(
        self,
        top_n_values: int = 10,
        orchestrator: Optional[Any] = None,
        **kwargs: Any
    ) -> str:
        """
        Scalable method to directly invoke the orchestrator_worker on this dataset.
        """
        return self.generate_report(top_n_values=top_n_values, orchestrator=orchestrator, **kwargs)


def extract_information(
    data_source: Union[str, Path, pd.DataFrame, Any],
    top_n_values: int = 10,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Convenience function to extract all data insights from a CSV/Excel file or DataFrame.

    Parameters
    ----------
    data_source : str, Path, pd.DataFrame, or file-like buffer
        The input data source (.csv or .xlsx file path, buffer, or DataFrame).
    top_n_values : int, optional
        Number of top frequent values to extract per column. Default is 10.
    **kwargs : Any
        Additional keyword arguments passed to the DataInformationExtractor.

    Returns
    -------
    dict
        Comprehensive dictionary of extracted data insights.
    """
    try:
        extractor = DataInformationExtractor(data_source=data_source, **kwargs)
        return extractor.extract_all(top_n_values=top_n_values)
    except Exception as e:
        err_msg = f"Unexpected error during information extraction: {str(e)}"
        logger.error(err_msg)
        return {
            "success": False,
            "errors": [err_msg],
            "dataset_overview": {},
            "df_info": {},
            "describe": {},
            "null_count": {},
            "correlation": {},
            "groupings": [],
        }


if __name__ == "__main__":
    import json
    from sklearn.datasets import fetch_openml

    print("--- Extracting insights from Titanic dataset (sklearn/OpenML) ---")
    try:
        titanic_df = fetch_openml("titanic", version=1, as_frame=True).frame
    except Exception:
        from sklearn.datasets import load_iris
        titanic_df = load_iris(as_frame=True).frame

    extractor = DataInformationExtractor(titanic_df)
    insights = extractor.extract_all()
    print(f"Extraction Success: {insights['success']}")
    print(f"Dataset Overview: {insights['dataset_overview']}")
    print(f"Columns: {insights['df_info']['col_names']}")
    print(f"Number of Groupings Computed: {len(insights['groupings'])}")
    print("\nNull Counts:")
    print(json.dumps(insights["null_count"], indent=2))