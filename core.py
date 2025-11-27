# core.py
"""
Core data-quality and EDA utilities for the Data Quality & EDA Assistant.

This file does NOT know anything about MCP. It's just a small Python API
for:
- Listing datasets (only CSV files)
- Generating a profiling report (Markdown)
- Detecting data issues and computing a "data quality score"
- Opening a saved report snippet
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Optional: anomaly detection (IsolationForest). We degrade gracefully if not installed.
try:
    from sklearn.ensemble import IsolationForest  # type: ignore
except Exception:  # pragma: no cover
    IsolationForest = None  # type: ignore


# -------------------------
# Paths & basic config
# -------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
REPORTS_DIR = BASE_DIR / "reports"

# Make sure the directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetInfo:
    """Simple structured info about a dataset on disk."""
    name: str           # base name without extension (e.g., "titanic")
    filename: str       # full file name (e.g., "titanic.csv")
    path: str           # full path on disk
    format: str         # always "csv" in this simple project
    n_rows: int         # -1 if we failed to read
    n_columns: int      # -1 if we failed to read
    columns: List[str]  # column names or error message


# -------------------------
# Internal helpers
# -------------------------

def _load_dataset_path(name: str) -> Path:
    """
    Resolve a dataset name to a CSV file in DATA_DIR.

    Supports:
    - "titanic"
    - "titanic.csv"
    """
    name_stripped = name.strip()
    name_lower = name_stripped.lower()

    # If user included an extension, try that exact file first
    direct_candidate = DATA_DIR / name_stripped
    if direct_candidate.exists():
        return direct_candidate

    # Try adding .csv
    candidate = DATA_DIR / f"{name_stripped}.csv"
    if candidate.exists():
        return candidate

    # As a last resort, scan by stem
    for path in DATA_DIR.iterdir():
        if path.is_file() and path.suffix.lower() == ".csv":
            if path.stem.lower() == name_lower:
                return path

    raise FileNotFoundError(
        f"Could not find dataset '{name}'. "
        f"Looked in {DATA_DIR}. Make sure a CSV file is placed there."
    )


def _read_dataset(path: Path) -> pd.DataFrame:
    """Read CSV into a pandas DataFrame."""
    if path.suffix.lower() != ".csv":
        raise ValueError(f"Unsupported file extension: {path.suffix}. Only .csv is supported.")
    df = pd.read_csv(path)
    return df


def _markdown_block_from_df(df: pd.DataFrame, max_rows: int = 10) -> str:
    """
    Render a DataFrame as a preformatted text block (cheap & dependency-free).
    We avoid df.to_markdown() to skip extra dependencies (tabulate).
    """
    preview = df.head(max_rows)
    return "```text\n" + preview.to_string() + "\n```"


# -------------------------
# Public API functions
# -------------------------

def list_datasets() -> List[DatasetInfo]:
    """
    List available CSV files in DATA_DIR with basic info.

    This reads each dataset fully (fine for small/medium files) to get row/column counts.
    """
    datasets: List[DatasetInfo] = []

    for path in sorted(DATA_DIR.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".csv":
            continue  # CSV only

        try:
            df = _read_dataset(path)
            n_rows, n_cols = df.shape
            info = DatasetInfo(
                name=path.stem,
                filename=path.name,
                path=str(path),
                format="csv",
                n_rows=int(n_rows),
                n_columns=int(n_cols),
                columns=list(df.columns),
            )
        except Exception as e:  # pragma: no cover
            info = DatasetInfo(
                name=path.stem,
                filename=path.name,
                path=str(path),
                format="csv",
                n_rows=-1,
                n_columns=-1,
                columns=[f"Error reading file: {e}"],
            )

        datasets.append(info)

    return datasets


def profile_dataset(name: str) -> Dict[str, Any]:
    """
    Compute simple EDA summary for a dataset and write a Markdown report
    into REPORTS_DIR.

    Returns a JSON-serialisable dict with:
    - basic summary
    - numeric summary (describe)
    - correlations
    - path to the Markdown report
    """
    path = _load_dataset_path(name)
    df = _read_dataset(path)

    n_rows, n_cols = df.shape

    # Column-level summary
    columns_summary: List[Dict[str, Any]] = []
    for col in df.columns:
        s = df[col]
        columns_summary.append(
            {
                "name": col,
                "dtype": str(s.dtype),
                "non_null": int(s.notna().sum()),
                "missing_count": int(s.isna().sum()),
                "missing_pct": float(s.isna().mean() * 100.0),
                "n_unique": int(s.nunique(dropna=True)),
            }
        )

    # Numeric summary
    num_df = df.select_dtypes(include="number")
    if not num_df.empty:
        numeric_summary_df = num_df.describe().T.round(3)
        numeric_summary_records = (
            numeric_summary_df.reset_index()
            .rename(columns={"index": "column"})
            .to_dict(orient="records")
        )
    else:
        numeric_summary_df = None
        numeric_summary_records = []

    # Correlation matrix
    if not num_df.empty and num_df.shape[1] >= 2:
        corr_df = num_df.corr(numeric_only=True).round(3)
        corr_dict = {col: {k: float(v) for k, v in row.items()} for col, row in corr_df.to_dict().items()}
    else:
        corr_df = None
        corr_dict = {}

    basic_summary: Dict[str, Any] = {
        "dataset_name": path.stem,
        "filename": path.name,
        "path": str(path),
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "columns": columns_summary,
    }

    # Build Markdown report
    md_lines: List[str] = []
    md_lines.append(f"# Dataset profile: `{path.stem}`\n")
    md_lines.append("## Overview\n")
    md_lines.append(f"- File: `{path.name}`")
    md_lines.append(f"- Rows: **{n_rows}**")
    md_lines.append(f"- Columns: **{n_cols}**\n")

    md_lines.append("## Column summary\n")
    md_lines.append("| Column | Dtype | Non-null | Missing % | Unique |\n"
                    "|--------|-------|----------|-----------|--------|")
    for col in columns_summary:
        md_lines.append(
            f"| {col['name']} | {col['dtype']} | {col['non_null']} | "
            f"{col['missing_pct']:.1f}% | {col['n_unique']} |"
        )
    md_lines.append("")

    md_lines.append("## Numeric summary (`pandas.describe()`)\n")
    if numeric_summary_df is not None:
        md_lines.append(_markdown_block_from_df(numeric_summary_df))
    else:
        md_lines.append("_No numeric columns found._\n")

    md_lines.append("## Correlations (numeric columns)\n")
    if corr_df is not None:
        md_lines.append(_markdown_block_from_df(corr_df))
    else:
        md_lines.append("_Not enough numeric columns for correlations._\n")

    md_report = "\n".join(md_lines)
    report_path = REPORTS_DIR / f"{path.stem}_profile.md"
    report_path.write_text(md_report, encoding="utf-8")

    return {
        "summary": basic_summary,
        "numeric_summary": numeric_summary_records,
        "correlations": corr_dict,
        "report_path": str(report_path),
    }


def detect_data_issues(name: str) -> Dict[str, Any]:
    """
    Run a simple data-quality check on a dataset:
    - Missing values
    - Duplicate IDs (id / *_id)
    - Outliers (IQR rule)
    - Skewness
    - Very strong correlation with target (leakage risk)
    - Optional anomaly detection with IsolationForest
    - Aggregate "data quality score" (0–100)
    """
    path = _load_dataset_path(name)
    df = _read_dataset(path)
    n_rows, n_cols = df.shape

    issues: List[Dict[str, Any]] = []

    # Missingness
    missing_pct = df.isna().mean() * 100.0
    for col, pct in missing_pct.items():
        if pct >= 30.0:
            severity = "high"
        elif pct >= 10.0:
            severity = "medium"
        else:
            continue  # ignore small missingness

        issues.append(
            {
                "type": "missing_values",
                "column": col,
                "severity": severity,
                "details": f"{pct:.1f}% of values are missing in column '{col}'.",
            }
        )

    # Duplicate IDs (heuristic: columns named id / *_id)
    candidate_id_cols = [
        c
        for c in df.columns
        if c.lower() == "id" or c.lower().endswith("_id")
    ]
    duplicate_info: List[Dict[str, Any]] = []
    for col in candidate_id_cols:
        dup_count = int(df[col].duplicated().sum())
        null_count = int(df[col].isna().sum())
        if dup_count > 0 or null_count > 0:
            severity = "high" if dup_count > 0 else "medium"
            issues.append(
                {
                    "type": "duplicate_ids",
                    "column": col,
                    "severity": severity,
                    "details": f"Column '{col}' has {dup_count} duplicate values and {null_count} nulls.",
                }
            )
        duplicate_info.append(
            {
                "column": col,
                "duplicate_count": dup_count,
                "null_count": null_count,
            }
        )

    # Outliers and skewness (numeric columns)
    num_df = df.select_dtypes(include="number")
    outlier_summary: List[Dict[str, Any]] = []
    skew_summary: List[Dict[str, Any]] = []

    if not num_df.empty and n_rows > 0:
        q1 = num_df.quantile(0.25)
        q3 = num_df.quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        for col in num_df.columns:
            col_series = num_df[col]
            mask_outliers = (col_series < lower[col]) | (col_series > upper[col])
            outlier_count = int(mask_outliers.sum())
            outlier_pct = float(outlier_count / n_rows * 100.0)

            skew_value = float(col_series.skew())

            outlier_summary.append(
                {
                    "column": col,
                    "outlier_count": outlier_count,
                    "outlier_pct": outlier_pct,
                }
            )
            skew_summary.append(
                {
                    "column": col,
                    "skew": skew_value,
                }
            )

            if outlier_pct >= 5.0:
                issues.append(
                    {
                        "type": "outliers",
                        "column": col,
                        "severity": "medium",
                        "details": f"{outlier_pct:.1f}% of values in '{col}' are IQR outliers.",
                    }
                )
            elif outlier_pct > 0.0:
                issues.append(
                    {
                        "type": "outliers",
                        "column": col,
                        "severity": "low",
                        "details": f"{outlier_pct:.1f}% of values in '{col}' are IQR outliers (low level).",
                    }
                )

            if abs(skew_value) >= 1.0:
                severity = "medium" if abs(skew_value) < 2.0 else "high"
                issues.append(
                    {
                        "type": "skewness",
                        "column": col,
                        "severity": severity,
                        "details": f"Column '{col}' has skewness {skew_value:.2f}, indicating a skewed distribution.",
                    }
                )

    # Heuristic target & leakage risk
    target_col: Optional[str] = None
    for cand in df.columns:
        lower_name = cand.lower()
        if lower_name in ("target", "label", "y", "survived"):
            target_col = cand
            break

    if target_col is None:
        # Fallback: any small-cardinality column (2–3 unique values)
        for cand in df.columns:
            nunique = int(df[cand].nunique(dropna=True))
            if 2 <= nunique <= 3:
                target_col = cand
                break

    leakage_warnings: List[Dict[str, Any]] = []
    if target_col is not None and not num_df.empty:
        target_series = df[target_col]
        if not pd.api.types.is_numeric_dtype(target_series):
            target_encoded = target_series.astype("category").cat.codes
        else:
            target_encoded = target_series

        x_target = target_encoded.fillna(0).to_numpy()

        for col in num_df.columns:
            if col == target_col:
                continue
            x = num_df[col].fillna(num_df[col].median()).to_numpy()
            if x.shape[0] != x_target.shape[0]:
                continue
            if np.all(x == x[0]):  # constant column
                continue

            corr_matrix = np.corrcoef(x, x_target)
            corr = float(corr_matrix[0, 1])
            if np.isnan(corr):
                continue

            if abs(corr) >= 0.95:
                msg = (
                    f"Column '{col}' has very strong correlation ({corr:.2f}) "
                    f"with target '{target_col}'. This might indicate leakage."
                )
                issues.append(
                    {
                        "type": "leakage_risk",
                        "column": col,
                        "severity": "high",
                        "details": msg,
                    }
                )
                leakage_warnings.append(
                    {
                        "column": col,
                        "target": target_col,
                        "corr_with_target": corr,
                    }
                )

    # Optional anomaly detection
    anomaly_summary: Optional[Dict[str, Any]] = None
    if IsolationForest is not None and not num_df.empty and n_rows >= 20:
        try:
            X = num_df.fillna(num_df.median())
            model = IsolationForest(
                contamination=0.01,
                random_state=42,
            )
            preds = model.fit_predict(X)
            anomaly_indices = np.where(preds == -1)[0]
            anomaly_fraction = float(len(anomaly_indices) / n_rows)

            anomaly_summary = {
                "n_anomalies": int(len(anomaly_indices)),
                "anomaly_fraction": anomaly_fraction,
            }

            if anomaly_fraction >= 0.02:
                issues.append(
                    {
                        "type": "anomaly_detection",
                        "column": None,
                        "severity": "medium",
                        "details": (
                            f"IsolationForest flagged {len(anomaly_indices)} rows "
                            f"as anomalies ({anomaly_fraction * 100:.1f}% of rows)."
                        ),
                    }
                )
        except Exception as e:  # pragma: no cover
            anomaly_summary = {"error": f"IsolationForest failed: {e}"}
    elif IsolationForest is None:
        anomaly_summary = {
            "note": "scikit-learn is not installed; skipping anomaly detection."
        }

    # Simple data quality score (0–100)
    score = 100

    # Penalty for missingness
    for col, pct in missing_pct.items():
        if pct >= 30.0:
            score -= 5
        elif pct >= 10.0:
            score -= 2

    for issue in issues:
        t = issue["type"]
        severity = issue["severity"]

        if t == "duplicate_ids":
            score -= 10
        elif t == "outliers" and severity == "medium":
            score -= 2
        elif t == "skewness" and severity == "high":
            score -= 2
        elif t == "leakage_risk":
            score -= 5
        elif t == "anomaly_detection":
            score -= 3

    score = max(0, min(100, score))

    return {
        "dataset": path.stem,
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "issues": issues,
        "missingness": {k: float(v) for k, v in missing_pct.round(2).to_dict().items()},
        "candidate_id_columns": candidate_id_cols,
        "duplicate_id_summary": duplicate_info,
        "outlier_summary": outlier_summary,
        "skew_summary": skew_summary,
        "leakage_warnings": leakage_warnings,
        "anomaly_summary": anomaly_summary,
        "data_quality_score": int(score),
    }


def open_report(name: str, max_chars: int = 2000) -> Dict[str, Any]:
    """
    Open a previously generated profiling report and return a short snippet.

    If the report doesn't exist yet, we'll generate it first.
    """
    # Try direct name_profile.md
    report_path = REPORTS_DIR / f"{name.strip()}_profile.md"
    if not report_path.exists():
        # Try inferring dataset path and regenerating profile
        path = _load_dataset_path(name)
        result = profile_dataset(path.stem)
        report_path = Path(result["report_path"])

    text = report_path.read_text(encoding="utf-8")
    snippet = text[:max_chars]

    return {
        "report_path": str(report_path),
        "snippet": snippet,
    }
