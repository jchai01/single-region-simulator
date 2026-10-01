"""
LandFuture DSS -- Scenario Viewer

Visualizes and compares NSGA-II, greedy, and random-baseline reallocation
scenarios: an interactive disruption-vs-carbon-gain chart (the same
comparison as the paper's Table 2, made interactive), plus a per-parcel
map for any selected scenario, plus a side-by-side compare mode between
any two scenarios (e.g. greedy vs. NSGA-II at the same disruption level).

DATA CONTRACT this app expects (see generate_synthetic_data.py for the
schema, or nsga2_scenario_search.py's export_scenario_details() / the
equivalent to be added to baseline_comparison.py for the real exporters):
  scenario_details: one row per (method, scenario_id, parcel_id) --
    columns: method, scenario_id, disruption_fraction,
    carbon_delta_vs_baseline_t, parcel_id, original_class,
    assigned_class, changed
  parcel geometry: a GeoDataFrame with a parcel_id column joinable
    against scenario_details, plus a geometry column.

Swap DETAILS_PATH / GEOMETRY_PATH below to point at real exported files
once available; everything else is schema-driven and needs no changes.
"""
import os
import glob
import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px

st.set_page_config(layout="wide", page_title="LandFuture DSS Scenario Viewer")


# Real exports, once both scripts have been run against the real region:
#   - nsga2_scenario_search.py's export_scenario_details() writes
#     data/scenario_details_seed{1,2,3,4}.parquet (one file per seed)
#   - baseline_comparison.py's export_scenario_details() writes
#     data/scenario_details_baselines.parquet (greedy + random together)
# Falls back to the synthetic demo files if none of the real ones exist,
# so the app still runs standalone for development/testing.
REAL_NSGA2_GLOB = "data/scenario_details_seed*.parquet"
REAL_BASELINES_PATH = "data/scenario_details_baselines.parquet"
# run extract_region_geometry.py once first
REAL_GEOMETRY_PATH = "data/region_geometry.gpkg"
# guard against accidentally pointing this at the full national file
MAX_SANE_PARCEL_COUNT = 5000

SYNTHETIC_DETAILS_PATH = "data/synthetic_scenario_details.parquet"
SYNTHETIC_GEOMETRY_PATH = "data/synthetic_parcels.gpkg"

CLASS_COLORS = {
    "tillage": "#D4A24C",
    "grassland": "#6FA86F",
    "forestry": "#2E5E3E",
    "peatland": "#8B5E3C",
}
METHOD_COLORS = {"nsga2": "#534AB7", "greedy": "#993C1D", "random": "#888780"}


@st.cache_data
def load_details():
    """Loads real exports if present (all NSGA-II seed files plus the
    baselines file, concatenated into one long-format table), otherwise
    falls back to the synthetic demo data. Returns (dataframe, is_real)."""
    nsga2_files = sorted(glob.glob(REAL_NSGA2_GLOB))
    has_baselines = os.path.exists(REAL_BASELINES_PATH)

    if nsga2_files or has_baselines:
        parts = [pd.read_parquet(f) for f in nsga2_files]
        if has_baselines:
            parts.append(pd.read_parquet(REAL_BASELINES_PATH))
        return pd.concat(parts, ignore_index=True), True

    return pd.read_parquet(SYNTHETIC_DETAILS_PATH), False


@st.cache_data
def load_geometry():
    """Real parcel geometry if present, else the synthetic grid.

    Guards against REAL_GEOMETRY_PATH accidentally pointing at the full
    national parcel file (~1.54M parcels) instead of the small, pre-
    extracted region file (run extract_region_geometry.py to produce
    it) -- that mistake is what made the app hang/crash, especially in
    split view, since every non-region parcel got rendered as a
    "not reallocatable" layer covering nearly the whole country.
    """
    if os.path.exists(REAL_GEOMETRY_PATH):
        gdf = gpd.read_file(REAL_GEOMETRY_PATH)
        if "parcel_id" not in gdf.columns:
            gdf["parcel_id"] = gdf.index
        if len(gdf) > MAX_SANE_PARCEL_COUNT:
            st.error(
                f"{REAL_GEOMETRY_PATH} has {len(gdf):,} parcels -- this looks like "
                "the full national file, not a region extract, and will hang the "
                "app (especially in split view). Run extract_region_geometry.py "
                "first to produce a small region-only file, then rerun this app."
            )
            st.stop()
        return gdf, True
    return gpd.read_file(SYNTHETIC_GEOMETRY_PATH), False


@st.cache_data
def summary_table(details):
    """One row per (method, scenario_id): aggregate disruption/carbon,
    collapsing the per-parcel rows -- what the comparison chart plots."""
    return (
        details.groupby(["method", "scenario_id"])
        .agg(disruption_fraction=("disruption_fraction", "first"),
             carbon_delta_vs_baseline_t=("carbon_delta_vs_baseline_t", "first"))
        .reset_index()
    )


NOT_REALLOCATABLE_COLOR = "#B9B7AE"


def render_scenario_map(ax, geometry, details, method, scenario_id, color_by):
    """
    geometry includes EVERY parcel in the region, but scenario_rows only
    covers parcels that survived filter_reallocatable() (immutable
    infrastructure and likely-commonage parcels are deliberately excluded
    from the reallocation target set upstream -- see generate_scenarios.py).
    The left-merge below leaves NaN for those excluded parcels' columns,
    which is real, meaningful information (not an error to paper over) --
    shown here as its own "not reallocatable" category rather than
    silently dropped or crashing on NaN.
    """
    scenario_rows = details[(details["method"] == method) & (
        details["scenario_id"] == scenario_id)]
    merged = geometry.merge(scenario_rows, on="parcel_id", how="left")
    not_in_scenario = merged["assigned_class"].isna()

    if color_by == "assigned_class":
        for cls, color in CLASS_COLORS.items():
            subset = merged[merged["assigned_class"] == cls]
            if len(subset):
                subset.plot(ax=ax, color=color, edgecolor="white",
                            linewidth=0.3, label=cls)
        excluded = merged[not_in_scenario]
        if len(excluded):
            excluded.plot(ax=ax, color=NOT_REALLOCATABLE_COLOR, edgecolor="white",
                          linewidth=0.3, label="not reallocatable")
        ax.legend(loc="upper left", bbox_to_anchor=(
            1.0, 1.0), fontsize=8, frameon=False)
    else:  # changed / unchanged
        # Explicit == comparisons, not `~merged["changed"]`: the "changed"
        # column is NaN (a float), not a clean bool, for excluded parcels,
        # and `~` on a float raises TypeError. == True / == False both
        # correctly evaluate to False for NaN, so excluded parcels fall
        # into neither bucket here and are handled as their own case below.
        unchanged = merged[merged["changed"] == False]
        changed = merged[merged["changed"] == True]
        excluded = merged[not_in_scenario]
        if len(unchanged):
            unchanged.plot(ax=ax, color="#E5E3DC",
                           edgecolor="white", linewidth=0.3, label="unchanged")
        if len(changed):
            changed.plot(ax=ax, color="#C0392B", edgecolor="white",
                         linewidth=0.3, label="changed")
        if len(excluded):
            excluded.plot(ax=ax, color=NOT_REALLOCATABLE_COLOR, edgecolor="white",
                          linewidth=0.3, label="not reallocatable")
        ax.legend(loc="upper left", bbox_to_anchor=(
            1.0, 1.0), fontsize=8, frameon=False)

    ax.set_axis_off()
    row = scenario_rows.iloc[0]
    ax.set_title(
        f"{method} -- disruption {row['disruption_fraction']:.1f}, "
        f"carbon {row['carbon_delta_vs_baseline_t']:,.0f} t C",
        fontsize=10,
    )


# ---------------------------------------------------------------------
st.title("LandFuture DSS -- Scenario Viewer")

details, details_are_real = load_details()
geometry, geometry_is_real = load_geometry()
summary = summary_table(details)

if not (details_are_real and geometry_is_real):
    st.info(
        "Showing synthetic demo data -- real exported files "
        f"({REAL_NSGA2_GLOB}, {REAL_BASELINES_PATH}, {REAL_GEOMETRY_PATH}) "
        "were not found in data/.",
        icon="ℹ️",
    )

st.subheader("Disruption vs. carbon gain -- all methods")
fig = px.scatter(
    summary, x="disruption_fraction", y="carbon_delta_vs_baseline_t",
    color="method", color_discrete_map=METHOD_COLORS,
    hover_data=["scenario_id"],
    labels={"disruption_fraction": "Disruption fraction",
            "carbon_delta_vs_baseline_t": "Carbon gain (t C)"},
)
fig.update_traces(marker=dict(size=10))
fig.update_layout(height=450, legend_title_text="Method")
st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("Scenario map")

color_by = st.radio("Color parcels by", [
                    "assigned_class", "changed"], horizontal=True)
compare_mode = st.checkbox("Compare two scenarios side by side")

methods_available = sorted(summary["method"].unique())


def scenario_picker(label, key_prefix):
    method = st.selectbox(f"{label} -- method",
                          methods_available, key=f"{key_prefix}_method")
    options = summary[summary["method"] ==
                      method].sort_values("disruption_fraction")
    choice = st.selectbox(
        f"{label} -- scenario",
        options["scenario_id"],
        format_func=lambda sid: (
            f"disruption {options.loc[options.scenario_id == sid, 'disruption_fraction'].iloc[0]:.1f}, "
            f"carbon {options.loc[options.scenario_id == sid, 'carbon_delta_vs_baseline_t'].iloc[0]:,.0f} t C"
        ),
        key=f"{key_prefix}_scenario",
    )
    return method, choice


if compare_mode:
    col1, col2 = st.columns(2)
    with col1:
        method_a, scenario_a = scenario_picker("Scenario A", "a")
    with col2:
        method_b, scenario_b = scenario_picker("Scenario B", "b")

    map_col1, map_col2 = st.columns(2)
    with map_col1:
        fig_a, ax_a = plt.subplots(figsize=(5, 5))
        render_scenario_map(ax_a, geometry, details,
                            method_a, scenario_a, color_by)
        st.pyplot(fig_a)
    with map_col2:
        fig_b, ax_b = plt.subplots(figsize=(5, 5))
        render_scenario_map(ax_b, geometry, details,
                            method_b, scenario_b, color_by)
        st.pyplot(fig_b)
else:
    method_a, scenario_a = scenario_picker("Scenario", "single")
    fig_a, ax_a = plt.subplots(figsize=(7, 7))
    render_scenario_map(ax_a, geometry, details,
                        method_a, scenario_a, color_by)
    st.pyplot(fig_a)
