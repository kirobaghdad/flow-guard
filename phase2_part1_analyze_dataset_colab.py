# %% [markdown]
# # Phase 2 Part 1: Analyze Your Dataset
#
# Colab-ready exploratory analysis for Team 12's IoT-23 v2 project:
# malicious network flow classification.
#
# This notebook source follows the project brief requirements:
# visualize the dataset, inspect histograms, review random samples,
# look for outliers, and compare against a simple ZeroR-style baseline.
#
# Official dataset sources:
# - IoT-23 v2 index: https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset-v2/
# - Stratosphere IoT-23 page: https://www.stratosphereips.org/datasets-iot23
#
# If a fresh Colab runtime is missing packages, run:
# `!pip install -q requests pandas matplotlib seaborn scikit-learn`

# %%
from __future__ import annotations

import io
import math
import os
import re
import warnings
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.figsize"] = (12, 6)


# %% [markdown]
# ## 1. Setup And Dataset Manifest
#
# The default subset intentionally mixes benign and malware scenarios while
# staying light enough for Colab.

# %%
BASE_DATASET_URL = "https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset-v2/"
CACHE_DIR = (
    Path("/content/iot23v2_cache")
    if Path("/content").exists()
    else Path("iot23v2_cache")
)
RANDOM_STATE = 42
MAX_PLOT_ROWS = 10_000
MAX_PCA_ROWS = 5_000
TOP_HIST_FEATURES = 12
TOP_BOXPLOT_FEATURES = 6
TOP_CORR_FEATURES = 16
TOP_DETAILED_LABELS = 12
TOP_CATEGORY_VALUES = 10
NUMERIC_EXCLUDE_COLUMNS = {"ts"}
MIN_MULTICLASS_CLASS_SIZE = 50
MAX_SVM_TUNING_ROWS = 50_000
MAX_SVM_FINAL_TRAIN_ROWS = 60_000

SCENARIO_MANIFEST = [
    {
        "scenario_id": "CTU-Honeypot-Capture-4-1",
        "capture_type": "benign",
        "device_or_malware": "Philips Hue Bridge",
        "source_file": "2018-10-25-14-06-32-192.168.1.132-zeek-conn.log.labeled",
    },
    {
        "scenario_id": "CTU-Honeypot-Capture-5-1",
        "capture_type": "benign",
        "device_or_malware": "Amazon Echo Dot",
        "source_file": "2018-09-21-11-40-22-192.168.2.3-zeek-conn-log.labeled",
    },
    {
        "scenario_id": "CTU-Honeypot-Capture-7-2",
        "capture_type": "benign",
        "device_or_malware": "Somfy Smart Door Lock And Gateway",
        "source_file": "2019-07-03-16-41-09-192.168.1.158-zeek-conn-log.labeled",
    },
    {
        "scenario_id": "CTU-IoT-Malware-Capture-3-1",
        "capture_type": "malware",
        "device_or_malware": "Muhstik",
        "source_file": "2018-05-19-20-57-19-192.168.2.5-zeek-conn-log.labeled",
    },
    {
        "scenario_id": "CTU-IoT-Malware-Capture-8-1",
        "capture_type": "malware",
        "device_or_malware": "Hakai",
        "source_file": "2018-07-31-15-15-09-192.168.100.113-zeek-conn-log.labeled",
    },
    {
        "scenario_id": "CTU-IoT-Malware-Capture-20-1",
        "capture_type": "malware",
        "device_or_malware": "Torii",
        "source_file": "2018-10-02-13-12-30-192.168.100.103-zeek-conn-log.labeled",
    },
    {
        "scenario_id": "CTU-IoT-Malware-Capture-34-1",
        "capture_type": "malware",
        "device_or_malware": "Mirai",
        "source_file": "2018-12-21-15-50-14-192.168.1.195-zeek-conn-log.labeled",
    },
    {
        "scenario_id": "CTU-IoT-Malware-Capture-42-1",
        "capture_type": "malware",
        "device_or_malware": "Trojan",
        "source_file": "2019-01-10-14-34-38-192.168.1.197-zeek-conn-log.labeled",
    },
]

manifest_df = pd.DataFrame(SCENARIO_MANIFEST)
print("===== DATASET MANIFEST =====")
display(manifest_df)
print(f"Cache directory: {CACHE_DIR}")


# %% [markdown]
# ## 2. Download And Parse
#
# Each scenario directory is resolved dynamically to its single
# `*-zeek-conn-log.labeled` file. The Zeek log parser reads the `#fields`
# header instead of assuming CSV columns.

# %%
def decode_zeek_value(raw_value: str) -> str:
    return raw_value.encode("utf-8").decode("unicode_escape")


def fetch_text(url: str, timeout: int = 60) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def build_labeled_file_url(
    scenario_id: str,
    source_file: str | None = None,
    base_url: str = BASE_DATASET_URL,
) -> str:
    if source_file:
        return urljoin(base_url, f"{scenario_id}/{source_file}")

    raise ValueError(
        f"No source filename was provided for {scenario_id}, so the labeled file URL "
        "cannot be constructed directly."
    )


def find_local_labeled_file(
    scenario_id: str,
    cache_dir: Path,
    preferred_filename: str | None = None,
) -> Path | None:
    scenario_dir = cache_dir / scenario_id

    if preferred_filename:
        preferred_path = scenario_dir / preferred_filename
        if preferred_path.exists():
            return preferred_path

    matches = sorted(scenario_dir.glob("*-zeek-conn*.labeled"))
    if matches:
        return matches[0]

    return None


def download_text_with_cache(url: str, cache_path: Path) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    text = fetch_text(url)
    cache_path.write_text(text, encoding="utf-8")
    return text


def parse_zeek_log(raw_text: str) -> pd.DataFrame:
    separator = "\t"
    unset_field = "-"
    empty_field = "(empty)"
    fields: list[str] | None = None
    mixed_separator_pattern = r"\t+|\s{3,}"

    data_lines: list[str] = []

    for line in raw_text.splitlines():
        if not line:
            continue

        if line.startswith("#separator "):
            separator = decode_zeek_value(line.split(" ", 1)[1].strip())
            continue

        if line.startswith("#unset_field "):
            unset_field = line.split(" ", 1)[1].strip()
            continue

        if line.startswith("#empty_field "):
            empty_field = line.split(" ", 1)[1].strip()
            continue

        if line.startswith("#"):
            directive_parts = line.split(None, 1)
            directive = directive_parts[0]
            payload = directive_parts[1] if len(directive_parts) > 1 else ""

            if directive == "#separator":
                separator = decode_zeek_value(payload.strip())
            elif directive == "#unset_field":
                unset_field = payload.strip()
            elif directive == "#empty_field":
                empty_field = payload.strip()
            elif directive == "#fields":
                fields = re.split(mixed_separator_pattern, payload.strip())
            continue

        data_lines.append(line)

    if not fields:
        raise ValueError("Could not find a '#fields' header in the Zeek log.")

    if not data_lines:
        return pd.DataFrame(columns=fields)

    parsed = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        sep=mixed_separator_pattern,
        names=fields,
        na_values=[unset_field, empty_field],
        keep_default_na=True,
        engine="python",
    )
    return parsed


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [
        re.sub(r"[^0-9a-zA-Z_]+", "_", str(column)).strip("_").lower()
        for column in cleaned.columns
    ]
    return cleaned


def convert_numeric_like_columns(
    df: pd.DataFrame,
    exclude: Iterable[str] | None = None,
    threshold: float = 0.90,
) -> pd.DataFrame:
    cleaned = df.copy()
    exclude = set(exclude or [])

    for column in cleaned.columns:
        if column in exclude:
            continue

        if cleaned[column].dtype != "object":
            continue

        string_values = cleaned[column].astype(str).str.replace(",", "", regex=False).str.strip()
        numeric_values = pd.to_numeric(string_values, errors="coerce")

        if numeric_values.notna().mean() >= threshold:
            cleaned[column] = numeric_values

    return cleaned


def first_present_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def normalize_dataset_frame(df: pd.DataFrame, scenario_meta: dict[str, str]) -> pd.DataFrame:
    normalized = clean_column_names(df)
    normalized = convert_numeric_like_columns(
        normalized,
        exclude={"uid", "id_orig_h", "id_resp_h", "proto", "service", "conn_state", "history"},
    )

    if "ts" in normalized.columns:
        normalized["ts_datetime"] = pd.to_datetime(normalized["ts"], unit="s", errors="coerce")

    label_column = first_present_column(normalized, ["label"])
    detailed_label_column = first_present_column(normalized, ["detailed_label", "detailedlabel"])

    if label_column is None:
        fallback_label = scenario_meta["capture_type"].strip().lower()
        label_text = pd.Series([fallback_label] * len(normalized), index=normalized.index, dtype="object")
        detailed_text = label_text.copy()
    else:
        label_text = normalized[label_column].fillna("unknown").astype(str).str.strip()
        if detailed_label_column is None:
            detailed_text = label_text.copy()
        else:
            detailed_text = normalized[detailed_label_column].fillna(label_text).astype(str).str.strip()

    detailed_text = detailed_text.replace({"": np.nan, "nan": np.nan}).fillna(label_text)

    normalized["label"] = label_text
    normalized["detailed_label"] = detailed_text
    normalized["detailed_label_clean"] = normalized["detailed_label"].astype(str).str.strip()
    normalized["binary_label"] = np.where(
        normalized["label"].str.lower().eq("benign"),
        "benign",
        "malicious",
    )
    normalized["scenario_id"] = scenario_meta["scenario_id"]
    normalized["capture_type"] = scenario_meta["capture_type"]
    normalized["device_or_malware"] = scenario_meta["device_or_malware"]

    return normalized


def load_iot23_subset(
    scenario_manifest: list[dict[str, str]],
    cache_dir: Path = CACHE_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    summary_rows: list[dict[str, object]] = []
    errors: list[str] = []

    for scenario_meta in scenario_manifest:
        scenario_id = scenario_meta["scenario_id"]
        preferred_filename = scenario_meta.get("source_file")
        print(f"Loading {scenario_id} ...")

        try:
            local_cache_path = find_local_labeled_file(
                scenario_id=scenario_id,
                cache_dir=cache_dir,
                preferred_filename=preferred_filename,
            )

            if local_cache_path is not None:
                filename = local_cache_path.name
                raw_text = local_cache_path.read_text(encoding="utf-8")
                source_mode = "local_cache"
            else:
                file_url = build_labeled_file_url(
                    scenario_id=scenario_id,
                    source_file=preferred_filename,
                )
                filename = Path(file_url).name
                cache_path = cache_dir / scenario_id / filename
                raw_text = download_text_with_cache(file_url, cache_path)
                source_mode = "download"

            parsed = parse_zeek_log(raw_text)
            normalized = normalize_dataset_frame(parsed, scenario_meta)
            frames.append(normalized)

            summary_rows.append(
                {
                    "scenario_id": scenario_id,
                    "capture_type": scenario_meta["capture_type"],
                    "device_or_malware": scenario_meta["device_or_malware"],
                    "rows": len(normalized),
                    "columns": normalized.shape[1],
                    "source_file": filename,
                    "load_mode": source_mode,
                }
            )
        except Exception as exc:
            errors.append(f"{scenario_id}: {exc}")

    if errors:
        raise RuntimeError(
            "Failed to download or parse one or more scenarios:\n- " + "\n- ".join(errors)
        )

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    summary = pd.DataFrame(summary_rows)
    return combined, summary


dataset, load_summary = load_iot23_subset(SCENARIO_MANIFEST)

print("\n===== DOWNLOAD SUMMARY =====")
display(load_summary)


# %% [markdown]
# ## 3. Dataset Overview
# 
# We inspect the combined dataframe before moving into plotting and baseline
# analysis.

# %%
def show_basic_overview(df: pd.DataFrame) -> None:
    print("===== DATASET OVERVIEW =====")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Approximate memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"Scenarios included: {df['scenario_id'].nunique() if 'scenario_id' in df.columns else 0}")

    print("\n===== FIRST 5 ROWS =====")
    display(df.head())

    print("\n===== RANDOM SAMPLE ROWS =====")
    display(df.sample(min(5, len(df)), random_state=RANDOM_STATE))

    print("\n===== DATA TYPES =====")
    dtype_table = pd.DataFrame({"dtype": df.dtypes.astype(str)}).sort_values("dtype")
    display(dtype_table)

    duplicate_rows = int(df.duplicated().sum())
    constant_columns = [
        column for column in df.columns if df[column].nunique(dropna=False) <= 1
    ]

    print(f"\nDuplicate rows: {duplicate_rows:,}")
    print(f"Constant columns: {len(constant_columns)}")
    if constant_columns:
        print(constant_columns[:20])


show_basic_overview(dataset)


# %% [markdown]
# ## 4. Missing-Value Analysis

# %%
def show_missing_values(df: pd.DataFrame) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]

    print("===== MISSING VALUES =====")
    if missing.empty:
        print("No missing values found.")
        return

    missing_table = pd.DataFrame(
        {
            "missing_count": missing,
            "missing_percent": ((missing / len(df)) * 100).round(2),
        }
    )
    display(missing_table.head(25))


show_missing_values(dataset)


# %% [markdown]
# ## 5. Target Analysis
# 
# `binary_label` is the main target. `detailed_label_clean` is included as a
# richer subtype view.

# %%
def plot_distribution_table_and_bar(
    series: pd.Series,
    title: str,
    top_n: int | None = None,
    palette: str = "viridis",
) -> None:
    cleaned = series.fillna("missing").astype(str)
    counts = cleaned.value_counts()

    if top_n is not None and len(counts) > top_n:
        top_counts = counts.head(top_n)
        other_count = counts.iloc[top_n:].sum()
        counts = pd.concat([top_counts, pd.Series({"other": other_count})])

    percent = ((counts / len(cleaned)) * 100).round(2)

    print(f"===== {title.upper()} =====")
    display(pd.DataFrame({"count": counts, "percent": percent}))

    plt.figure(figsize=(12, 5))
    ax = sns.barplot(x=counts.index.astype(str), y=counts.values, palette=palette)
    plt.title(title)
    plt.xlabel(series.name or "category")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=45, ha="right")

    for idx, value in enumerate(counts.values):
        ax.text(idx, value, f"{value:,}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.show()


plot_distribution_table_and_bar(
    dataset["binary_label"],
    title="Binary Label Distribution",
    top_n=None,
    palette="viridis",
)

plot_distribution_table_and_bar(
    dataset["detailed_label_clean"],
    title=f"Top {TOP_DETAILED_LABELS} Detailed Labels",
    top_n=TOP_DETAILED_LABELS,
    palette="magma",
)


# %% [markdown]
# ## 6. Domain-Specific Categorical Plots
# 
# These plots help us understand how protocol, service, and connection states
# differ between benign and malicious flows.

# %%
def plot_categorical_feature_by_label(
    df: pd.DataFrame,
    feature: str,
    label_col: str = "binary_label",
    top_n: int = TOP_CATEGORY_VALUES,
) -> None:
    if feature not in df.columns:
        print(f"Skipping '{feature}' because it is not present in the dataset.")
        return

    working = df[[feature, label_col]].copy()
    working[feature] = working[feature].fillna("missing").astype(str)
    top_values = working[feature].value_counts().head(top_n).index
    working = working[working[feature].isin(top_values)]

    count_table = pd.crosstab(working[feature], working[label_col]).sort_values(
        by=working[label_col].unique().tolist(),
        ascending=False,
    )

    print(f"===== {feature.upper()} BY {label_col.upper()} =====")
    display(count_table)

    ax = count_table.plot(kind="bar", stacked=True, figsize=(12, 6), colormap="Set2")
    ax.set_title(f"{feature} distribution by {label_col}")
    ax.set_xlabel(feature)
    ax.set_ylabel("Flow count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


for categorical_feature in ["proto", "service", "conn_state"]:
    plot_categorical_feature_by_label(dataset, categorical_feature)


# %% [markdown]
# ## 7. Numeric EDA
# 
# We compute summary statistics, histograms, outlier tables, class-wise
# boxplots, a correlation heatmap, and a PCA projection.

# %%
def get_numeric_features(
    df: pd.DataFrame,
    exclude: Iterable[str] | None = None,
) -> list[str]:
    exclude = set(exclude or [])
    exclude.update(NUMERIC_EXCLUDE_COLUMNS)

    numeric_frame = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan)
    filtered_columns: list[str] = []

    for column in numeric_frame.columns:
        if column in exclude:
            continue

        series = numeric_frame[column]
        if series.notna().sum() < 2:
            continue

        if series.nunique(dropna=True) <= 1:
            continue

        filtered_columns.append(column)

    return filtered_columns


def select_top_variance_features(
    df: pd.DataFrame,
    numeric_features: list[str],
    top_k: int,
) -> list[str]:
    if not numeric_features:
        return []

    numeric_frame = df[numeric_features].replace([np.inf, -np.inf], np.nan)
    variances = numeric_frame.var(numeric_only=True).sort_values(ascending=False)
    return variances.head(min(top_k, len(variances))).index.tolist()


def show_numeric_summary(df: pd.DataFrame, numeric_features: list[str]) -> None:
    if not numeric_features:
        print("No numeric features found, so summary statistics are skipped.")
        return

    numeric_frame = df[numeric_features].replace([np.inf, -np.inf], np.nan)
    summary = numeric_frame.describe().T
    summary["missing_percent"] = (numeric_frame.isna().mean() * 100).round(2)
    summary["zero_percent"] = (numeric_frame.eq(0).mean() * 100).round(2)
    summary["skewness"] = numeric_frame.skew(numeric_only=True).round(2)

    print("===== NUMERIC FEATURE SUMMARY =====")
    display(summary.sort_values("std", ascending=False).head(20))


def plot_numeric_histograms(df: pd.DataFrame, numeric_features: list[str]) -> None:
    if not numeric_features:
        print("No numeric features found, so histogram plotting is skipped.")
        return

    top_features = select_top_variance_features(df, numeric_features, TOP_HIST_FEATURES)
    sample_df = (
        df[top_features]
        .replace([np.inf, -np.inf], np.nan)
        .sample(min(MAX_PLOT_ROWS, len(df)), random_state=RANDOM_STATE)
    )

    print("===== HISTOGRAMS FOR HIGH-VARIANCE NUMERIC FEATURES =====")
    sample_df.hist(bins=30, figsize=(18, 12), edgecolor="black")
    plt.suptitle("Feature Histograms", y=1.02)
    plt.tight_layout()
    plt.show()


def compute_iqr_outliers(df: pd.DataFrame, numeric_features: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    numeric_frame = df[numeric_features].replace([np.inf, -np.inf], np.nan)

    for column in numeric_features:
        series = numeric_frame[column].dropna()

        if series.nunique() < 2:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            continue

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        outlier_mask = (series < low) | (series > high)

        rows.append(
            {
                "feature": column,
                "outlier_count": int(outlier_mask.sum()),
                "outlier_percent": round(outlier_mask.mean() * 100, 2),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["feature", "outlier_count", "outlier_percent"])

    return pd.DataFrame(rows).sort_values("outlier_percent", ascending=False)


def plot_boxplots_by_label(
    df: pd.DataFrame,
    label_col: str,
    numeric_features: list[str],
) -> None:
    unique_labels = df[label_col].fillna("missing").astype(str).nunique()
    if unique_labels > 6 or not numeric_features:
        print("Skipping feature-vs-label boxplots because they would be too crowded.")
        return

    top_features = select_top_variance_features(df, numeric_features, TOP_BOXPLOT_FEATURES)
    plot_df = (
        df[[label_col] + top_features]
        .replace([np.inf, -np.inf], np.nan)
        .sample(min(MAX_PLOT_ROWS, len(df)), random_state=RANDOM_STATE)
    )

    print("===== FEATURE DISTRIBUTIONS BY CLASS =====")
    cols = 2
    rows = math.ceil(len(top_features) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, 5 * rows))
    axes = np.array(axes).reshape(-1)

    for idx, feature in enumerate(top_features):
        sns.boxplot(
            data=plot_df,
            x=label_col,
            y=feature,
            ax=axes[idx],
            palette="Set2",
        )
        axes[idx].tick_params(axis="x", rotation=45)
        axes[idx].set_title(f"{feature} by {label_col}")

    for idx in range(len(top_features), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, numeric_features: list[str]) -> None:
    if len(numeric_features) < 2:
        print("Not enough numeric features for a correlation heatmap.")
        return

    top_features = select_top_variance_features(df, numeric_features, TOP_CORR_FEATURES)
    corr = (
        df[top_features]
        .replace([np.inf, -np.inf], np.nan)
        .corr()
    )

    print("===== CORRELATION HEATMAP =====")
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=True)
    plt.title("Correlation Heatmap (Top Variance Features)")
    plt.tight_layout()
    plt.show()


def plot_pca_projection(
    df: pd.DataFrame,
    label_col: str,
    numeric_features: list[str],
) -> None:
    if len(numeric_features) < 2:
        print("Not enough numeric features for PCA visualization.")
        return

    top_features = select_top_variance_features(df, numeric_features, min(12, len(numeric_features)))
    plot_df = df[[label_col] + top_features].replace([np.inf, -np.inf], np.nan).copy()
    usable_features: list[str] = []

    for feature in top_features:
        series = plot_df[feature]
        if series.notna().sum() < 2:
            continue

        median_value = series.median()
        if pd.isna(median_value):
            continue

        plot_df[feature] = series.fillna(median_value)
        if plot_df[feature].nunique(dropna=True) > 1:
            usable_features.append(feature)

    if len(usable_features) < 2:
        print("Not enough usable numeric features remain for PCA after preprocessing.")
        return

    plot_df = plot_df.sample(min(MAX_PCA_ROWS, len(plot_df)), random_state=RANDOM_STATE)

    x = plot_df[usable_features]
    y = plot_df[label_col].fillna("missing").astype(str)
    x_scaled = StandardScaler().fit_transform(x)

    pca_model = PCA(n_components=2, random_state=RANDOM_STATE)
    x_pca = pca_model.fit_transform(x_scaled)

    pca_df = pd.DataFrame(
        {
            "pc1": x_pca[:, 0],
            "pc2": x_pca[:, 1],
            label_col: y.values,
        }
    )

    print("===== PCA PROJECTION =====")
    plt.figure(figsize=(10, 7))
    sns.scatterplot(
        data=pca_df,
        x="pc1",
        y="pc2",
        hue=label_col,
        alpha=0.7,
        s=40,
        palette="tab10",
    )
    plt.title(
        "2D PCA Projection\n"
        f"Explained variance: {pca_model.explained_variance_ratio_.sum() * 100:.2f}%"
    )
    plt.tight_layout()
    plt.show()


numeric_features = get_numeric_features(dataset, exclude={"binary_label"})

show_numeric_summary(dataset, numeric_features)
plot_numeric_histograms(dataset, numeric_features)

outlier_table = compute_iqr_outliers(dataset, numeric_features)
print("===== OUTLIER SUMMARY (IQR METHOD) =====")
if outlier_table.empty:
    print("No numeric outlier summary could be computed.")
else:
    display(outlier_table.head(15))

plot_boxplots_by_label(dataset, "binary_label", numeric_features)
plot_correlation_heatmap(dataset, numeric_features)
plot_pca_projection(dataset, "binary_label", numeric_features)


# %% [markdown]
# ## 8. ZeroR-Style Baseline
# 
# This baseline predicts the majority class for every sample to put later model
# results into context.

# %%
def run_majority_class_baseline(df: pd.DataFrame, target_col: str = "binary_label") -> None:
    y = df[target_col].fillna("missing_label").astype(str)
    x_dummy = np.zeros((len(df), 1))
    class_counts = y.value_counts()
    stratify = y if class_counts.min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x_dummy,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(x_train, y_train)
    y_pred = baseline.predict(x_test)
    predicted_class = pd.Series(y_pred).iloc[0]

    print("===== BASELINE: MAJORITY CLASS (ZeroR-Style) =====")
    print(f"Predicted class for every sample: {predicted_class}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    print(f"Macro F1: {f1_score(y_test, y_pred, average='macro', zero_division=0):.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    labels = class_counts.index.tolist()
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Baseline Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


run_majority_class_baseline(dataset, "binary_label")


# %% [markdown]
# ## 9. Report-Ready Notes
# 
# These bullets are meant to be copied into the report discussion and refined
# later with your actual observations.

# %%
def print_report_ready_notes(df: pd.DataFrame, numeric_features: list[str]) -> None:
    target_counts = df["binary_label"].fillna("missing").astype(str).value_counts(normalize=True) * 100
    detailed_counts = df["detailed_label_clean"].fillna("missing").astype(str).value_counts()
    duplicates = int(df.duplicated().sum())
    missing_columns = int((df.isna().sum() > 0).sum())

    print("===== REPORT-READY NOTES =====")
    print(f"- Dataset size: {len(df):,} rows x {df.shape[1]:,} columns")
    print(f"- Scenarios used: {df['scenario_id'].nunique()}")
    print(f"- Numeric features available for EDA: {len(numeric_features)}")
    print(f"- Columns with missing values: {missing_columns}")
    print(f"- Duplicate rows: {duplicates:,}")
    print(f"- Majority binary class: '{target_counts.index[0]}' ({target_counts.iloc[0]:.2f}% of the data)")
    print("- Accuracy alone should not be used as the main evaluation metric if the classes are imbalanced.")
    print("- Balanced accuracy, macro F1, and the confusion matrix should be emphasized in later model comparisons.")

    print("\nTop detailed labels:")
    for label, count in detailed_counts.head(8).items():
        print(f"- {label}: {count:,} flows")

    print("\nPer-scenario row counts:")
    scenario_counts = df["scenario_id"].value_counts()
    for scenario_id, count in scenario_counts.items():
        print(f"- {scenario_id}: {count:,} rows")


print_report_ready_notes(dataset, numeric_features)


# %% [markdown]
# ## 10. Phase 2 Part 2: Multiclass Linear SVM
# 
# We now switch from binary classification to multiclass malware
# classification. Instead of predicting `benign` vs `malicious`, we keep only
# malicious flows and predict the finer attack category stored in
# `detailed_label_clean`.
# 
# To keep the task stable:
# - we remove benign rows,
# - we group very rare attack labels into `other_malware`,
# - we train one linear SVM model with simple preprocessing,
# - and we report multiclass metrics on a held-out test set.
#
# We use a linear SVM instead of a kernel SVM because this dataset is large and
# the features become high-dimensional after one-hot encoding.

# %%
def get_modeling_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    blocked_columns = {
        "label",
        "detailed_label",
        "detailed_label_clean",
        "binary_label",
        "scenario_id",
        "capture_type",
        "device_or_malware",
        "uid",
        "id_orig_h",
        "id_resp_h",
        "ts",
        "ts_datetime",
    }

    numeric_cols = [
        column
        for column in get_numeric_features(df)
        if column not in blocked_columns
    ]

    candidate_categorical_cols = ["proto", "service", "conn_state"]
    categorical_cols = [
        column
        for column in candidate_categorical_cols
        if column in df.columns and column not in blocked_columns
    ]

    feature_cols = numeric_cols + categorical_cols
    return feature_cols, numeric_cols, categorical_cols


def prepare_multiclass_malware_dataset(
    df: pd.DataFrame,
    min_class_size: int = MIN_MULTICLASS_CLASS_SIZE,
) -> tuple[pd.DataFrame, pd.Series]:
    malware_df = df[df["binary_label"] == "malicious"].copy()
    malware_df["attack_category"] = (
        malware_df["detailed_label_clean"]
        .fillna("unknown_attack")
        .astype(str)
        .str.strip()
    )

    class_counts = malware_df["attack_category"].value_counts()
    rare_labels = class_counts[class_counts < min_class_size].index

    if len(rare_labels) > 0:
        malware_df.loc[
            malware_df["attack_category"].isin(rare_labels),
            "attack_category",
        ] = "other_malware"

    final_counts = malware_df["attack_category"].value_counts()
    if final_counts.min() < 4:
        raise ValueError(
            "At least one multiclass label has fewer than 4 samples after grouping, "
            "which is too small for the train/validation/test split."
        )

    return malware_df, final_counts


def build_multiclass_linear_svm_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    c_value: float,
    class_weight: str | None,
) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    model = LinearSVC(
        C=c_value,
        class_weight=class_weight,
        max_iter=2000,
        tol=1e-3,
        dual="auto",
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_multiclass_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def run_multiclass_linear_svm_experiment(df: pd.DataFrame) -> None:
    feature_cols, numeric_cols, categorical_cols = get_modeling_columns(df)
    malware_df, class_counts = prepare_multiclass_malware_dataset(df)

    print("===== MULTICLASS LINEAR SVM EXPERIMENT =====")
    print(f"Feature columns used: {len(feature_cols)}")
    print(f"Numeric columns: {numeric_cols}")
    print(f"Categorical columns: {categorical_cols}")

    print("\n===== MULTICLASS TARGET DISTRIBUTION =====")
    multiclass_distribution = pd.DataFrame(
        {
            "count": class_counts,
            "percent": ((class_counts / class_counts.sum()) * 100).round(2),
        }
    )
    display(multiclass_distribution)

    x = malware_df[feature_cols]
    y = malware_df["attack_category"].astype(str)

    x_train_val, x_test, y_train_val, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_val,
        y_train_val,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    print(f"\nTrain size: {len(x_train):,}")
    print(f"Validation size: {len(x_valid):,}")
    print(f"Test size: {len(x_test):,}")

    if len(x_train) > MAX_SVM_TUNING_ROWS:
        x_tune, _, y_tune, _ = train_test_split(
            x_train,
            y_train,
            train_size=MAX_SVM_TUNING_ROWS,
            random_state=RANDOM_STATE,
            stratify=y_train,
        )
    else:
        x_tune, y_tune = x_train, y_train

    print(f"Tuning subset size: {len(x_tune):,}")

    if len(x_train_val) > MAX_SVM_FINAL_TRAIN_ROWS:
        x_train_final, _, y_train_final, _ = train_test_split(
            x_train_val,
            y_train_val,
            train_size=MAX_SVM_FINAL_TRAIN_ROWS,
            random_state=RANDOM_STATE,
            stratify=y_train_val,
        )
    else:
        x_train_final, y_train_final = x_train_val, y_train_val

    print(f"Final training size: {len(x_train_final):,}")

    hyperparameter_grid = [
        {"C": 0.1, "class_weight": None},
        {"C": 1.0, "class_weight": "balanced"},
    ]

    validation_rows: list[dict[str, float | str]] = []
    best_params: dict[str, float | str | None] | None = None
    best_score = -np.inf

    for params in hyperparameter_grid:
        pipeline = build_multiclass_linear_svm_pipeline(
            numeric_cols=numeric_cols,
            categorical_cols=categorical_cols,
            c_value=float(params["C"]),
            class_weight=params["class_weight"],
        )

        pipeline.fit(x_tune, y_tune)
        valid_pred = pipeline.predict(x_valid)
        metrics = evaluate_multiclass_predictions(y_valid, valid_pred)

        validation_rows.append(
            {
                "C": float(params["C"]),
                "class_weight": str(params["class_weight"]),
                **{key: round(value, 4) for key, value in metrics.items()},
            }
        )

        selection_score = metrics["macro_f1"]
        if selection_score > best_score:
            best_score = selection_score
            best_params = params

    validation_results = pd.DataFrame(validation_rows).sort_values(
        by=["macro_f1", "balanced_accuracy", "accuracy"],
        ascending=False,
    )

    print("\n===== VALIDATION RESULTS =====")
    display(validation_results)

    if best_params is None:
        raise RuntimeError("No multiclass linear SVM model was successfully trained.")

    print("\nBest validation setting:")
    print(best_params)

    final_pipeline = build_multiclass_linear_svm_pipeline(
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        c_value=float(best_params["C"]),
        class_weight=best_params["class_weight"],
    )
    final_pipeline.fit(x_train_final, y_train_final)

    test_pred = final_pipeline.predict(x_test)
    test_metrics = evaluate_multiclass_predictions(y_test, test_pred)

    print("\n===== TEST RESULTS =====")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value:.4f}")

    labels = class_counts.index.tolist()

    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, labels=labels, zero_division=0))

    cm = confusion_matrix(y_test, test_pred, labels=labels)

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Multiclass Linear SVM Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    result_summary = pd.DataFrame(
        {
            "metric": list(test_metrics.keys()),
            "value": [round(value, 4) for value in test_metrics.values()],
        }
    )

    print("\n===== TEST METRIC TABLE =====")
    display(result_summary)


run_multiclass_linear_svm_experiment(dataset)
