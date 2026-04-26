"""
ui/app.py

Streamlit front-end for NeuralLens.

Run with:
    streamlit run ui/app.py
"""

# stdlib
import sys
import os

# Ensure project root is on the path when run from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# third-party
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# local
from pipeline.optimization_loop import run_full_pipeline

load_dotenv()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="NeuralLens 🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("NeuralLens 🧠")
    st.caption("Neuromarketing for every business — not just the Fortune 500")
    st.divider()

    industry = st.selectbox(
        "Industry",
        [
            "general",
            "ecommerce",
            "saas",
            "restaurant",
            "healthcare",
            "real estate",
            "finance",
            "education",
            "non-profit",
        ],
        index=0,
    )

    with st.expander("What is the Neural Engagement Score?"):
        st.markdown(
            """
The **Neural Engagement Score (NES)** predicts how strongly a human brain
engages with a webpage, on a 0–100 scale.

It is computed from fMRI-style brain-region activations produced by
Meta's TRIBE v2 model, weighted by how each region contributes to
consumer decision-making:

| Weight | Region | Signal |
|--------|--------|--------|
| +0.25 | Striatum | Desire / reward |
| +0.20 | Amygdala | Emotional pull |
| +0.15 | Hippocampus | Memory encoding |
| −0.20 | dlPFC | Cognitive load |
| +0.10 | IPS | Attention salience |
| +0.10 | mPFC | Self-relevance |
| ±0.00 | Insula | Distrust signal |
"""
        )

    with st.expander("Brain region guide"):
        st.markdown(
            """
- **Amygdala** — emotional arousal; high = content evokes feeling
- **Striatum** — reward/desire; high = visitor wants what you're selling
- **Hippocampus** — memory; high = page will be remembered
- **dlPFC** — cognitive effort; HIGH = page is too complex (bad)
- **IPS** — visual attention; high = CTA is noticed
- **mPFC** — self-relevance; high = "this is for me"
- **Insula** — visceral discomfort; HIGH = something feels wrong (bad)
"""
        )

    st.divider()
    st.markdown("**Agentverse agent**")
    st.code("agent address — fill in after running agent", language="text")

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------

st.title("NeuralLens 🧠")
st.subheader("Neuromarketing for every business — not just the Fortune 500")
st.markdown(
    "Paste any URL below to run Meta's TRIBE v2 brain simulation model on it "
    "and receive three neuroscience-backed optimisation suggestions."
)

url_input = st.text_input(
    "Page URL",
    placeholder="https://yourwebsite.com",
    help="Must include http:// or https://",
)

run_button = st.button("Analyze with Brain Simulation", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Pipeline execution
# ---------------------------------------------------------------------------

if run_button and url_input:
    with st.spinner("Running TRIBE v2 brain simulation..."):
        try:
            result = run_full_pipeline(url_input.strip(), industry)
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")
            st.stop()

    # -----------------------------------------------------------------------
    # Top metrics
    # -----------------------------------------------------------------------

    col_before, col_after, col_profile = st.columns(3)
    with col_before:
        st.metric("NES Before", f"{result['nes_before']}/100")
    with col_after:
        st.metric(
            "NES After",
            f"{result['nes_after']}/100",
            delta=f"{result['delta']:+.1f}",
        )
    with col_profile:
        st.metric("Profile (after)", result["profile_after"])

    st.progress(int(result["nes_after"]))

    st.divider()

    # -----------------------------------------------------------------------
    # Heatmap overlays
    # -----------------------------------------------------------------------

    img_col_a, img_col_b = st.columns(2)
    with img_col_a:
        st.image(
            result["overlay_before_url"],
            caption="Before — brain activation heatmap overlaid on original screenshot",
            use_column_width=True,
        )
    with img_col_b:
        st.image(
            result["overlay_after_url"],
            caption="After — brain activation heatmap following Gemma optimisations",
            use_column_width=True,
        )

    st.divider()

    # -----------------------------------------------------------------------
    # Emotion circumplex (Russell's model)
    # -----------------------------------------------------------------------

    st.subheader("Emotion Circumplex")
    fig = go.Figure()

    # Quadrant labels
    for x, y, label in [
        (0.7, 0.7, "Excited"), (-0.7, 0.7, "Tense"),
        (0.7, -0.7, "Relaxed"), (-0.7, -0.7, "Sad"),
    ]:
        fig.add_annotation(x=x, y=y, text=label, showarrow=False,
                           font=dict(size=11, color="grey"))

    # Quadrant dividers
    fig.add_hline(y=0, line_dash="dot", line_color="lightgrey")
    fig.add_vline(x=0, line_dash="dot", line_color="lightgrey")

    fig.add_trace(go.Scatter(
        x=[result["valence_before"]],
        y=[result["arousal_before"]],
        mode="markers+text",
        marker=dict(color="red", size=14),
        text=["Before"],
        textposition="top center",
        name="Before",
    ))
    fig.add_trace(go.Scatter(
        x=[result["valence_after"]],
        y=[result["arousal_after"]],
        mode="markers+text",
        marker=dict(color="green", size=14),
        text=["After"],
        textposition="top center",
        name="After",
    ))

    fig.update_layout(
        xaxis=dict(title="Valence (negative ← → positive)", range=[-1.1, 1.1]),
        yaxis=dict(title="Arousal (calm ← → excited)", range=[-1.1, 1.1]),
        height=400,
        margin=dict(l=40, r=40, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # -----------------------------------------------------------------------
    # ROI breakdown
    # -----------------------------------------------------------------------

    st.subheader("Brain Region Breakdown")
    roi_regions = ["amygdala", "striatum", "hippocampus", "dlPFC", "IPS", "mPFC", "insula"]
    roi_cols = st.columns(7)
    for col, region in zip(roi_cols, roi_regions):
        before_val = result["roi_before"].get(region, 0)
        after_val = result["roi_after"].get(region, 0)
        delta_val = round(after_val - before_val, 1)
        col.metric(region, f"{after_val}", delta=f"{delta_val:+.1f}")

    st.divider()

    # -----------------------------------------------------------------------
    # Gemma suggested changes
    # -----------------------------------------------------------------------

    st.subheader("Optimisation Changes")
    st.caption(result.get("summary", ""))

    for i, change in enumerate(result.get("changes", []), start=1):
        with st.expander(
            f"Change {i}: [{change.get('target_region', '?')}] "
            f"{change.get('selector', '')} — {change.get('reason', '')}"
        ):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Before**")
                st.code(change.get("old_value", ""), language="html")
            with c2:
                st.markdown("**After**")
                st.code(change.get("new_value", ""), language="html")
            st.caption(
                f"Property: `{change.get('property')}` | "
                f"Target region: `{change.get('target_region')}`"
            )

    st.divider()

    # -----------------------------------------------------------------------
    # Issues detected
    # -----------------------------------------------------------------------

    if result.get("issues"):
        st.subheader("Issues Detected")
        for issue in result["issues"]:
            st.warning(issue)

elif run_button and not url_input:
    st.warning("Please enter a URL before clicking Analyze.")
