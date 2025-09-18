# app.py
import streamlit as st
import pandas as pd
import yaml
import io
import re
import plotly.express as px
import os

st.set_page_config(page_title="Benchmarking (Category Summary)", layout="wide")

# ------------------------- helpers -------------------------
NUM_RE = re.compile(r"[-+]?\d*\.?\d+")


def parse_numeric(cell, pick="max"):
    if pd.isna(cell):
        return None
    s = str(cell)
    nums = NUM_RE.findall(s)
    nums = [float(n) for n in nums] if nums else []
    if not nums:
        return None
    if pick == "min":
        return min(nums)
    if pick == "avg":
        return sum(nums) / len(nums)
    return max(nums)


def parse_boolean(cell):
    if pd.isna(cell):
        return 0
    s = str(cell).strip().lower()
    if s in ("yes", "y", "true", "1"):
        return 1
    if s in ("no", "n", "false", "0"):
        return 0
    return 0  # unknown -> treat as false/0


def scale_dict(d, invert=False):
    vals = {k: v for k, v in d.items() if v is not None}
    if not vals:
        return {k: 0.0 for k in d.keys()}
    vmin = min(vals.values())
    vmax = max(vals.values())
    if abs(vmax - vmin) < 1e-12:
        return {k: (50.0 if d[k] is not None else 0.0) for k in d.keys()}
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = 0.0
        else:
            s = (v - vmin) / (vmax - vmin) * 100.0
            out[k] = 100.0 - s if invert else s
    return out


def load_yaml(cfg_file):
    if cfg_file is None:
        return {}
    try:
        raw = cfg_file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        return yaml.safe_load(raw) or {}
    except Exception as e:
        st.error(f"Failed to parse YAML config: {e}")
        return {}


# ------------------------- scoring -------------------------
def compute_scores(df, cfg):
    if "Criterion" not in df.columns or "Category" not in df.columns:
        raise ValueError("Specs sheet must contain 'Category' and 'Criterion' columns.")

    df = df.copy()
    df["Criterion"] = df["Criterion"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    products = [c for c in df.columns if c not in ("Category", "Criterion")]
    missing_rules = set()

    weighted = pd.DataFrame(index=df["Criterion"].values, columns=products)

    for _, row in df.iterrows():
        crit = row["Criterion"]
        cfg_rule = cfg.get(crit)
        if not cfg_rule:
            missing_rules.add(crit)
            continue

        ctype = cfg_rule.get("type", "numeric")
        weight = float(cfg_rule.get("weight", 1.0))
        pick = cfg_rule.get("pick", "max")

        if ctype in ("numeric", "numeric_inverse"):
            parsed = {p: parse_numeric(row[p], pick=pick) for p in products}
            invert = (ctype == "numeric_inverse") or bool(cfg_rule.get("invert", False))
            scaled = scale_dict(parsed, invert=invert)
            for p in products:
                weighted.at[crit, p] = scaled.get(p, 0.0) * weight

        elif ctype == "boolean":
            parsed = {p: parse_boolean(row[p]) for p in products}
            for p in products:
                weighted.at[crit, p] = parsed.get(p, 0.0) * 100.0 * weight

        elif ctype == "categorical":
            mapping = cfg_rule.get("mapping", {})
            parsed = {p: float(mapping.get(str(row[p]).strip(), 0.0)) for p in products}
            scaled = scale_dict(parsed, invert=False)
            for p in products:
                weighted.at[crit, p] = scaled.get(p, 0.0) * weight

        else:
            parsed = {p: parse_numeric(row[p], pick=pick) for p in products}
            scaled = scale_dict(parsed, invert=False)
            for p in products:
                weighted.at[crit, p] = scaled.get(p, 0.0) * weight

    weighted = weighted.fillna(0.0).astype(float)
    crit_to_cat = dict(zip(df["Criterion"], df["Category"]))
    weighted.index.name = "Criterion"
    weighted = weighted.reset_index()
    weighted["Category"] = weighted["Criterion"].map(crit_to_cat)

    products = [c for c in weighted.columns if c not in ("Criterion", "Category")]
    cat_group = weighted.groupby("Category")[products].sum()

    cat_scores_0_10 = cat_group.copy().astype(float)
    for cat in cat_scores_0_10.index:
        row = cat_scores_0_10.loc[cat]
        vmax = row.max()
        if vmax == 0:
            cat_scores_0_10.loc[cat] = 0
        else:
            cat_scores_0_10.loc[cat] = (row / vmax * 10).round().astype(int)

    cat_scores_0_10.loc["TOTAL"] = cat_scores_0_10.sum(numeric_only=True)
    totals = cat_scores_0_10.loc["TOTAL"]
    cat_scores_0_10.loc["RANK"] = totals.rank(ascending=False, method="dense").astype(int)

    detailed = weighted.set_index("Criterion")
    detailed_cols = ["Category"] + [c for c in detailed.columns if c != "Category"]
    detailed = detailed[detailed_cols]

    return detailed, cat_scores_0_10, sorted(list(missing_rules))


# ------------------------- Streamlit UI -------------------------
st.title("Product Benchmarking")
st.markdown(
    "Upload a Specs Excel (sheet named `Specs` with columns: Category, Criterion, then product columns). "
    "Optionally upload a scoring_config.yaml to control scoring logic."
)

# Sidebar projects section
st.sidebar.subheader("Projects")
templates_dir = "templates"
if os.path.exists(templates_dir):
    project_folders = [f for f in os.listdir(templates_dir) if os.path.isdir(os.path.join(templates_dir, f))]
    for project in project_folders:
        st.sidebar.markdown(f"### {project}")
        project_path = os.path.join(templates_dir, project)
        project_files = [f for f in os.listdir(project_path) if os.path.isfile(os.path.join(project_path, f))]
        for file_name in project_files:
            file_path = os.path.join(project_path, file_name)
            with open(file_path, "rb") as f:
                st.sidebar.download_button(
                    label=f"Download {file_name}",
                    data=f,
                    file_name=file_name
                )

spec_file = st.file_uploader("Specs Excel (.xlsx)", type=["xlsx"])
cfg_file = st.file_uploader("scoring_config.yaml (optional)", type=["yaml", "yml"])

if not spec_file:
    st.info("Please upload Specs Excel to proceed.")
    st.stop()

try:
    specs = pd.read_excel(spec_file, sheet_name="Specs")
except Exception as e:
    st.error(f"Failed to read 'Specs' sheet: {e}")
    st.stop()

if "Category" not in specs.columns or "Criterion" not in specs.columns:
    st.error("Specs sheet must contain 'Category' and 'Criterion' columns.")
    st.stop()

# Preview table - always show unfiltered data
st.subheader("Preview Specs (first rows)")
st.dataframe(specs.head())  # Use the original specs DataFrame for the preview

# Load YAML config
cfg = load_yaml(cfg_file) if cfg_file else {}
if cfg_file:
    with st.expander("Uploaded Scoring Config"):
        st.json(cfg)

# Initialize session states
if 'detailed' not in st.session_state:
    st.session_state.detailed = None
if 'category_summary' not in st.session_state:
    st.session_state.category_summary = None

# Run Scoring
if st.button("Run Scoring"):
    with st.spinner("Processing data..."):
        try:
            detailed, category_summary, missing = compute_scores(specs, cfg)
            # Store the computed data in session state
            st.session_state.detailed = detailed
            st.session_state.category_summary = category_summary
            st.success("Processing complete!")
        except Exception as e:
            st.error(f"Error computing scores: {e}")
            st.stop()

# Add category filter and display results if data exists
if st.session_state.detailed is not None:
    # Add category filter in the side panel
    st.sidebar.subheader("Filter by Category")
    categories = sorted(st.session_state.detailed["Category"].unique().tolist())
    
    # Create multiselect for categories using the session state directly
    selected_cats = st.sidebar.multiselect(
        "Select Categories",
        options=categories,
        default=categories,  # Default to all categories
        help="Select categories to filter the results. If none selected, all categories will be shown."
    )

    # Initialize filtered dataframes with all data
    filtered_detailed = st.session_state.detailed.copy()
    filtered_summary = st.session_state.category_summary.copy()

    # Apply filter only if specific categories are selected
    if selected_cats:
        filtered_detailed = filtered_detailed[filtered_detailed["Category"].isin(selected_cats)]
        # Keep 'TOTAL' and 'RANK' rows when filtering
        filter_cats = selected_cats + ['TOTAL', 'RANK']
        filtered_summary = filtered_summary.loc[filtered_summary.index.isin(filter_cats)]

    # Display the tabs with filtered data
    tab1, tab2 = st.tabs(["Detailed Scores", "Category Summary"])
    with tab1:
        st.subheader("Criterion-level weighted scores")
        st.dataframe(filtered_detailed)
    with tab2:
        st.subheader("Category summary")
        st.dataframe(filtered_summary)

        # Show plot in the Category Summary tab
        if "TOTAL" in filtered_summary.index:
            totals = filtered_summary.loc["TOTAL"]
            fig = px.bar(x=totals.index, y=totals.values,
                        labels={"x": "Product", "y": "TOTAL score"},
                        title="TOTAL scores per product")
            st.plotly_chart(fig, use_container_width=True)

    # Show missing rules warning
    if hasattr(st.session_state, 'missing_rules') and st.session_state.missing_rules:
        st.warning("Missing rules in YAML for: " + ", ".join(st.session_state.missing_rules))

    # Add download button for results
    out = io.BytesIO()
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        specs.to_excel(writer, sheet_name="Specs", index=False)
        filtered_detailed.to_excel(writer, sheet_name="DetailedWeighted", index=True)
        filtered_summary.to_excel(writer, sheet_name="CategorySummary", index=True)
    out.seek(0)
    st.download_button("Download results Excel", data=out.read(), file_name="benchmark_results.xlsx")
