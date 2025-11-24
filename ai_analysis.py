"""
AI-powered analysis assistant using Google Gemini (FREE).

This module interprets natural-language prompts and turns them into a
*declarative* analysis specification that is then executed by safe,
pre-defined Python routines.

Capabilities:
- General LLM-style chat (greetings, questions about physics, etc.).
- Descriptive statistics of the loaded dataset.
- 1D histograms, scatter and 2D histograms.
- Automatic application of numeric cuts.
- Optional overlay of two datasets (e.g. signal vs background).
- Simple significance estimates such as S/sqrt(B) or S/sqrt(S+B).

The entry point used by the Streamlit app is `analyze_with_ai`.
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Gemini client setup
# ---------------------------------------------------------------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client: Optional[genai.Client] = None

if GEMINI_API_KEY:
    try:
        gemini_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        gemini_client = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_with_ai(
    user_prompt: str,
    df: pd.DataFrame,
    secondary_df: Optional[pd.DataFrame] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Main entry point for AI-assisted analysis.

    Parameters
    ----------
    user_prompt:
        Natural-language request from the user.
    df:
        Primary dataset as a pandas DataFrame.
    secondary_df:
        Optional secondary dataset (e.g. background if primary is signal).
    meta:
        Optional metadata dictionary, e.g.
        {
            "primary_name": "...",
            "secondary_name": "...",
            "primary_role": "signal" | "background" | "unlabeled",
            "secondary_role": "signal" | "background" | "unlabeled",
        }

    Returns
    -------
    dict
        {
            "success": bool,
            "explanation": str,
            "figure": matplotlib.figure.Figure or None,
            "results": dict or None,
            "specification": dict (the AI-generated JSON spec)
        }
    """
    if df is None or len(df) == 0:
        return {
            "success": False,
            "error": "No data loaded. Please upload or select a dataset first.",
        }

    if gemini_client is None:
        return {
            "success": False,
            "error": (
                "Gemini client is not configured. Set GEMINI_API_KEY in the "
                "environment to enable AI analysis."
            ),
        }

    # --------------------------------------
    # 1. Build compact descriptors for LLM
    # --------------------------------------
    def _summarize_df(frame: pd.DataFrame, name: str) -> str:
        cols = frame.columns.tolist()
        dtypes = {c: str(frame[c].dtype) for c in cols}
        n_rows, n_cols = frame.shape
        return (
            f"Dataset '{name}': {n_rows} rows, {n_cols} columns. "
            f"Column types: {dtypes}"
        )

    primary_name = (meta or {}).get("primary_name", "primary")
    secondary_name = (meta or {}).get("secondary_name", "secondary")

    primary_role = (meta or {}).get("primary_role", "unlabeled")
    secondary_role = (meta or {}).get("secondary_role", "unlabeled")

    primary_desc = _summarize_df(df, primary_name)

    if secondary_df is not None:
        secondary_desc = _summarize_df(secondary_df, secondary_name)
        combo_desc = (
            primary_desc
            + f"  Role: {primary_role}.  "
            + secondary_desc
            + f"  Role: {secondary_role}."
        )
    else:
        secondary_desc = "No secondary dataset provided."
        combo_desc = primary_desc + f"  Role: {primary_role}.  " + secondary_desc

    # --------------------------------------
    # 2. System prompt for structured JSON
    # --------------------------------------
    system_prompt = f"""
You are an expert assistant for particle-physics data analysis.

You receive one mandatory *primary* dataset and an optional *secondary* dataset
(e.g. signal vs background). Your job is to either:

(a) Have a normal conversation (greetings, conceptual questions, time/date), or
(b) Produce a concrete analysis plan that can be carried out in Python.

DATASETS:
- {combo_desc}

ROLES (if provided):
- A dataset with role "signal" should be treated as S.
- A dataset with role "background" should be treated as B.

You MUST respond with a SINGLE JSON object and NOTHING else.
The JSON MUST be valid (no comments, no trailing commas).

Required top-level keys in the JSON object:

{{
  "explanation": "Natural-language reply to the user.",
  "operation": "chat | describe | statistics | plot | overlay_plot | significance",
  "plot_type": "none | histogram | scatter | 2dhist | line | overlay_histogram",
  "column_x": null or "name of x column",
  "column_y": null or "name of y column (for scatter/2dhist/line)",
  "bins": 50,
  "range_min": null,
  "range_max": null,
  "xlabel": null or "label for x-axis",
  "ylabel": null or "label for y-axis",
  "title": null or "plot title",
  "statistics": [],
  "filters": null or {{
      "column_name": {{
          "min": float or null,
          "max": float or null
      }}
  }},
  "use_secondary": false,
  "normalize": "none | unit_area",
  "significance": {{
      "enabled": false,
      "metric": "S_over_sqrtB" | "S_over_sqrtSplusB",
      "variable": null or "column name to cut on",
      "region": "window | greater | less",
      "min": null,
      "max": null
  }}
}}

BEHAVIOUR RULES:

1. Pure chat:
   - If the user only greets you or asks a general question (not about the data),
     set "operation": "chat", "plot_type": "none".
   - Put your conversational answer entirely in "explanation".
   - Leave "statistics" empty and "significance.enabled" false.

2. Simple description / column info:
   - If the user asks for "what columns exist", "describe the dataset", or
     similar, set "operation": "describe", "plot_type": "none".
   - The Python side will handle the description using the loaded dataset.

3. Plots on a single dataset:
   - For requests like "plot invariant mass", "make a histogram of pt", etc.,
     use "operation": "plot".
   - Use "plot_type": "histogram" for 1D distributions.
   - Use "plot_type": "scatter" or "2dhist" for 2D visualisations.
   - Fill "column_x", and "column_y" when needed.
   - If a numeric range is mentioned (e.g. 70–200 GeV), set "range_min" and
     "range_max" appropriately.

4. Overlay plots (signal vs background):
   - If the user says "overlay", "compare", or "plot signal and background
     together", set:
       "operation": "overlay_plot",
       "plot_type": "overlay_histogram",
       "use_secondary": true.
   - Assume the primary dataset is used first; the secondary dataset is the
     comparison sample.

5. Statistics:
   - If the user requests numeric summaries (mean, std, etc.), populate
     "statistics" with any of:
       ["mean", "std", "median", "min", "max", "count", "sum"].
   - Use "column_x" as the variable to summarise.

6. Significance calculations:
   - If the user asks for "significance", "S/sqrt(B)", "sensitivity", etc.,
     set "operation": "significance" AND
       "significance.enabled": true.
   - Choose "metric" as either "S_over_sqrtB" or "S_over_sqrtSplusB".
   - Choose "variable" and a "region" among:
       - "window": events with min <= variable <= max
       - "greater": events with variable >= min
       - "less":    events with variable <= max
   - If only one dataset is available OR roles are not signal/background,
     still return a JSON object but explain the limitation in "explanation"
     and keep "significance.enabled": false.

Remember: return ONLY the JSON object, no markdown fences, no extra text.
"""

    # --------------------------------------
    # 3. Call Gemini to get the JSON spec
    # --------------------------------------
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=system_prompt + "\n\nUser request: " + user_prompt)],
                )
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    except Exception as e:
        return {
            "success": False,
            "error": f"Gemini API error: {e}",
        }

    # Simple JSON responses usually land in .text
    raw_text = getattr(response, "text", None)
    if not raw_text and getattr(response, "candidates", None):
        # Fallback: try to extract from first candidate
        try:
            raw_text = response.candidates[0].content.parts[0].text
        except Exception:
            raw_text = None

    if not raw_text:
        return {
            "success": False,
            "error": "Gemini did not return any text.",
        }

    try:
        spec = json.loads(raw_text)
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to parse JSON from Gemini: {e}",
            "raw_response": raw_text,
        }

    # --------------------------------------
    # 4. Execute the declarative spec safely
    # --------------------------------------
    try:
        result = execute_safe_analysis(
            df=df,
            spec=spec,
            secondary_df=secondary_df,
            meta=meta,
        )
        result["specification"] = spec
        return result
    except Exception as exec_error:
        return {
            "success": False,
            "error": f"Error executing analysis: {exec_error}",
            "specification": spec,
            "explanation": spec.get("explanation", ""),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _apply_filters(df: pd.DataFrame, filters: Optional[Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """Apply simple numeric min/max filters described in the JSON spec."""
    if not filters:
        return df

    filtered = df.copy()
    for col, f in filters.items():
        if col not in filtered.columns:
            continue
        if f is None:
            continue
        min_val = f.get("min", None)
        max_val = f.get("max", None)
        if min_val is not None:
            filtered = filtered[filtered[col] >= min_val]
        if max_val is not None:
            filtered = filtered[filtered[col] <= max_val]
    return filtered


def _compute_statistics(series: pd.Series, names: List[str]) -> Dict[str, float]:
    """Compute basic statistics for a 1D series."""
    out: Dict[str, float] = {}
    if series.empty:
        return out

    if "mean" in names:
        out["mean"] = float(series.mean())
    if "std" in names:
        out["std"] = float(series.std())
    if "median" in names:
        out["median"] = float(series.median())
    if "min" in names:
        out["min"] = float(series.min())
    if "max" in names:
        out["max"] = float(series.max())
    if "count" in names:
        out["count"] = int(series.count())
    if "sum" in names:
        out["sum"] = float(series.sum())
    return out


def _compute_significance(
    sig_counts: int,
    bkg_counts: int,
    metric: str,
) -> float:
    """Compute simple significance estimates."""
    if bkg_counts <= 0 and metric == "S_over_sqrtB":
        return float("nan")
    if sig_counts + bkg_counts <= 0 and metric == "S_over_sqrtSplusB":
        return float("nan")

    if metric == "S_over_sqrtB":
        return sig_counts / math.sqrt(bkg_counts) if bkg_counts > 0 else float("nan")
    elif metric == "S_over_sqrtSplusB":
        return sig_counts / math.sqrt(sig_counts + bkg_counts) if (sig_counts + bkg_counts) > 0 else float("nan")
    else:
        return float("nan")


def execute_safe_analysis(
    df: pd.DataFrame,
    spec: Dict[str, Any],
    secondary_df: Optional[pd.DataFrame] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Safely execute analysis based on the declarative specification.

    Only uses controlled operations (pandas / numpy / matplotlib).
    No arbitrary code execution from the LLM is ever run.

    Returns
    -------
    dict with keys:
        - success
        - explanation
        - figure (or None)
        - results (dict with stats / significance, etc.)
    """
    operation = spec.get("operation", "plot")
    plot_type = spec.get("plot_type", "none")

    column_x = spec.get("column_x")
    column_y = spec.get("column_y")

    bins = spec.get("bins", 50)
    range_min = spec.get("range_min")
    range_max = spec.get("range_max")
    xlabel = spec.get("xlabel")
    ylabel = spec.get("ylabel")
    title = spec.get("title")
    stats_req = spec.get("statistics") or []
    filters = spec.get("filters")
    use_secondary = bool(spec.get("use_secondary", False))
    normalize = spec.get("normalize", "none")

    sig_spec = spec.get("significance") or {}
    sig_enabled = bool(sig_spec.get("enabled", False))

    explanation = spec.get("explanation", "")

    # Base result object
    result: Dict[str, Any] = {
        "success": True,
        "explanation": explanation,
        "figure": None,
        "results": {},
    }

    # 1. Chat-only: nothing else to do
    if operation == "chat" or (plot_type == "none" and not stats_req and not sig_enabled):
        return result

    # 2. Prepare filtered data
    df_primary = _apply_filters(df, filters)

    df_secondary = None
    if secondary_df is not None:
        df_secondary = _apply_filters(secondary_df, filters)

    # 3. Statistics on primary dataset
    if stats_req and column_x and column_x in df_primary.columns:
        series = df_primary[column_x].dropna()
        result["results"].update(_compute_statistics(series, stats_req))

    # 4. Plotting
    fig = None
    if plot_type != "none":
        fig, ax = plt.subplots(figsize=(6, 4))

        # Histogram on primary (and optional secondary overlay)
        if plot_type in ("histogram", "overlay_histogram"):
            if column_x is None or column_x not in df_primary.columns:
                raise ValueError(f"Column {column_x!r} not found in primary dataset.")

            data_primary = df_primary[column_x].dropna().values

            hist_range = None
            if range_min is not None and range_max is not None:
                hist_range = (range_min, range_max)

            # Normalise if requested
            density = normalize == "unit_area"

            ax.hist(
                data_primary,
                bins=bins,
                range=hist_range,
                histtype="stepfilled",
                alpha=0.6,
                density=density,
                label=(meta or {}).get("primary_name", "primary"),
            )

            # Optional second dataset overlay
            if plot_type == "overlay_histogram" and use_secondary and df_secondary is not None:
                if column_x not in df_secondary.columns:
                    raise ValueError(f"Column {column_x!r} not found in secondary dataset.")
                data_secondary = df_secondary[column_x].dropna().values
                ax.hist(
                    data_secondary,
                    bins=bins,
                    range=hist_range,
                    histtype="step",
                    linewidth=1.8,
                    density=density,
                    label=(meta or {}).get("secondary_name", "secondary"),
                )
                ax.legend()

        elif plot_type == "scatter":
            if not column_x or not column_y:
                raise ValueError("Both column_x and column_y are required for scatter plots.")
            if column_x not in df_primary.columns or column_y not in df_primary.columns:
                raise ValueError("Requested columns not found in primary dataset.")
            ax.scatter(
                df_primary[column_x],
                df_primary[column_y],
                s=8,
                alpha=0.5,
            )

        elif plot_type == "2dhist":
            if not column_x or not column_y:
                raise ValueError("Both column_x and column_y are required for 2D histograms.")
            if column_x not in df_primary.columns or column_y not in df_primary.columns:
                raise ValueError("Requested columns not found in primary dataset.")
            ax.hist2d(
                df_primary[column_x],
                df_primary[column_y],
                bins=bins if isinstance(bins, int) else 50,
            )
            plt.colorbar(ax.collections[0], ax=ax)

        elif plot_type == "line":
            if not column_x:
                raise ValueError("column_x is required for line plots.")
            if column_x not in df_primary.columns:
                raise ValueError(f"Column {column_x!r} not found in primary dataset.")
            series = df_primary[column_x].sort_values().reset_index(drop=True)
            ax.plot(series.index, series.values)

        # Axis labels and title
        if xlabel:
            ax.set_xlabel(xlabel)
        elif column_x:
            ax.set_xlabel(column_x)

        if ylabel:
            ax.set_ylabel(ylabel)
        elif plot_type in ("histogram", "overlay_histogram"):
            ax.set_ylabel("Events")

        if title:
            ax.set_title(title)

        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        result["figure"] = fig

    # 5. Significance using two datasets
    if sig_enabled and df_secondary is not None:
        metric = sig_spec.get("metric", "S_over_sqrtB")
        var = sig_spec.get("variable") or column_x
        region = sig_spec.get("region", "window")
        s_min = sig_spec.get("min")
        s_max = sig_spec.get("max")

        if var is None:
            raise ValueError("Significance requested but no variable specified.")

        if var not in df_primary.columns or var not in df_secondary.columns:
            raise ValueError(f"Column {var!r} not found in both datasets.")

        # Build masks
        def _mask(frame: pd.DataFrame) -> pd.Series:
            series = frame[var]
            if region == "window":
                if s_min is None or s_max is None:
                    raise ValueError("Window region requires both min and max.")
                return (series >= s_min) & (series <= s_max)
            elif region == "greater":
                if s_min is None:
                    raise ValueError("Greater region requires min.")
                return series >= s_min
            elif region == "less":
                if s_max is None:
                    raise ValueError("Less region requires max.")
                return series <= s_max
            else:
                raise ValueError(f"Unknown region type: {region!r}")

        mask_sig = _mask(df_primary)
        mask_bkg = _mask(df_secondary)

        # Decide which dataset is S and which is B from meta roles
        primary_role = (meta or {}).get("primary_role", "unlabeled")
        secondary_role = (meta or {}).get("secondary_role", "unlabeled")

        # Default: if roles not given, treat primary as signal, secondary as background
        if primary_role == "signal" and secondary_role == "background":
            S = int(mask_sig.sum())
            B = int(mask_bkg.sum())
        elif primary_role == "background" and secondary_role == "signal":
            S = int(mask_bkg.sum())
            B = int(mask_sig.sum())
        else:
            # Fallback: primary = signal, secondary = background
            S = int(mask_sig.sum())
            B = int(mask_bkg.sum())

        Z = _compute_significance(S, B, metric)

        result["results"].update(
            {
                "significance_metric": metric,
                "variable": var,
                "region": region,
                "S": S,
                "B": B,
                "Z": Z,
            }
        )

    return result

