import streamlit as st
import pandas as pd
import yaml
import io
import re
from sklearn.preprocessing import MinMaxScaler

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
        return sum(nums)/len(nums)
    return max(nums)

def parse_boolean(cell):
    if pd.isna(cell):
        return 0
    s = str(cell).strip().lower()
    if s in ("yes","y","true","1"):
        return 1
    if s in ("no","n","false","0"):
        return 0
    return 0

def scale_dict(d, invert=False):
    # d: product->value (numeric or None) -> returns product->0..100 scaled
    vals = {k:v for k,v in d.items() if v is not None}
    if not vals:
        return {k:0.0 for k in d.keys()}
    vmin = min(vals.values())
    vmax = max(vals.values())
    if abs(vmax - vmin) < 1e-12:
        return {k:50.0 if d[k] is not None else 0.0 for k in d.keys()}
    out = {}
    for k,v in d.items():
        if v is None:
            out[k]=0.0
        else:
            s = (v - vmin)/(vmax - vmin) * 100.0
            out[k] = 100.0 - s if invert else s
    return out

def load_yaml(cfg_file):
    if cfg_file is None:
        return {}
    try:
        import yaml
        return yaml.safe_load(cfg_file)
    except Exception as e:
        st.error(f"Failed to parse YAML config: {e}")
        return {}

# ------------------------- scoring -------------------------
def compute_scores(df, cfg):
    # df: DataFrame with columns Category, Criterion, <products...>
    df["Criterion"] = df["Criterion"].astype(str).str.strip()
    df["Category"] = df["Category"].astype(str).str.strip()
    products = [c for c in df.columns if c not in ("Category","Criterion")]
    crit_rows = []
    missing_rules = set()

    # store weighted scores (criterion x products)
    weighted = pd.DataFrame(index=df["Criterion"].values, columns=products)

    for _, row in df.iterrows():
        crit = row["Criterion"]
        cfg_rule = cfg.get(crit)
        if not cfg_rule:
            missing_rules.add(crit)
            # treat unknown as boolean with weight 0 (ignored)
            continue
        ctype = cfg_rule.get("type","numeric")
        weight = float(cfg_rule.get("weight",1.0))
        pick = cfg_rule.get("pick","max")

        if ctype in ("numeric","numeric_inverse"):
            parsed = {p: parse_numeric(row[p], pick=pick) for p in products}
            invert = (ctype=="numeric_inverse") or bool(cfg_rule.get("invert",False))
            scaled = scale_dict(parsed, invert=invert)  # 0..100
            for p in products:
                weighted.at[crit,p] = scaled.get(p,0.0) * weight

        elif ctype=="boolean":
            parsed = {p: parse_boolean(row[p]) for p in products}
            for p in products:
                weighted.at[crit,p] = parsed.get(p,0.0) * 100.0 * weight

        elif ctype=="categorical":
            mapping = cfg_rule.get("mapping",{})
            parsed = {p: float(mapping.get(str(row[p]).strip(), 0.0)) for p in products}
            # if mapping given likely already numeric scores; normalize 0..100
            scaled = scale_dict(parsed, invert=False)
            for p in products:
                weighted.at[crit,p] = scaled.get(p,0.0) * weight

        else:
            # fallback numeric
            parsed = {p: parse_numeric(row[p], pick=pick) for p in products}
            scaled = scale_dict(parsed, invert=False)
            for p in products:
                weighted.at[crit,p] = scaled.get(p,0.0) * weight

    weighted = weighted.fillna(0.0).astype(float)

    # attach category mapping
    crit_to_cat = dict(zip(df["Criterion"], df["Category"]))
    weighted.index.name = "Criterion"
    weighted = weighted.reset_index()

    # group by category: sum weighted scores per category per product
    weighted["Category"] = weighted["Criterion"].map(crit_to_cat)
    products = [c for c in weighted.columns if c not in ("Criterion","Category")]
    cat_group = weighted.groupby("Category")[products].sum()

    # normalize each category to 0..10 integer scale (so summary shows small integers)
    cat_scores_0_10 = cat_group.copy().astype(float)
    for cat in cat_scores_0_10.index:
        row = cat_scores_0_10.loc[cat]
        vmax = row.max()
        if vmax == 0:
            cat_scores_0_10.loc[cat] = 0
        else:
            cat_scores_0_10.loc[cat] = (row / vmax * 10).round().astype(int)

    # compute total as sum of category integers
    cat_scores_0_10.loc["TOTAL"] = cat_scores_0_10.sum(numeric_only=True)
    # compute rank by TOTAL descending
    totals = cat_scores_0_10.loc["TOTAL"]
    ranks = totals.rank(ascending=False, method="dense").astype(int)
    cat_scores_0_10.loc["RANK"] = ranks

    # prepare detailed weighted (criterion x product) as DataFrame with Category column
    detailed = weighted.set_index("Criterion")
    # reorder columns: Category first then products
    detailed_cols = ["Category"] + [c for c in detailed.columns if c!="Category"]
    detailed = detailed[detailed_cols]

    return detailed, cat_scores_0_10, sorted(list(missing_rules))

# ------------------------- Streamlit UI -------------------------
st.title("Product Benchmarking — Category Summary Output")

st.markdown("Upload a Specs Excel (sheet 'Specs' with columns: Category, Criterion, then product columns). Upload a YAML scoring_config to control logic.")

spec_file = st.file_uploader("Specs Excel (.xlsx)", type=["xlsx"])
cfg_file = st.file_uploader("scoring_config.yaml (optional)", type=["yaml","yml"])

if not spec_file:
    st.info("Please upload Specs Excel to proceed. Use the provided template if needed.")
    st.stop()

try:
    specs = pd.read_excel(spec_file, sheet_name="Specs")
except Exception as e:
    st.error(f"Failed to read 'Specs' sheet: {e}")
    st.stop()

st.subheader("Preview Specs (first rows)")
st.dataframe(specs.head())

cfg = load_yaml(cfg_file) if cfg_file else load_yaml(None)
detailed, category_summary, missing = compute_scores(specs, cfg)

st.subheader("Criterion-level weighted scores (raw, 0..100 * weight)")
st.dataframe(detailed)

st.subheader("Category summary (0..10 per category scale, TOTAL and RANK)")
st.dataframe(category_summary)

if missing:
    st.warning("Missing rules in YAML for: " + ", ".join(missing))

# allow download
out = io.BytesIO()
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    specs.to_excel(writer, sheet_name="Specs", index=False)
    detailed.to_excel(writer, sheet_name="DetailedWeighted", index=True)
    category_summary.to_excel(writer, sheet_name="CategorySummary", index=True)
out.seek(0)
st.download_button("Download results Excel", data=out.read(), file_name="benchmark_results.xlsx")
