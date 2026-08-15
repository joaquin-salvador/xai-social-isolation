import streamlit as st
from model import load_artifacts, DEFAULT_CATEGORY_LABELS, DEFAULT_DISPLAY_NAMES, DEFAULT_LIKERT_FEATURES
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="XAI for Post-COVID Social Isolation Prediction",
    page_icon="assets/computer.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS
st.markdown("""
<style>
/* Colored metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #7c3aed11, #a855f711);
    border: 1px solid #7c3aed33;
    border-radius: 10px;
    padding: 15px;
}

/* Sidebar styling — light background with legible dark text */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f3effc, #e5dcf5);
}
[data-testid="stSidebar"] * {
    color: #1a1a2e !important;
}
[data-testid="stSidebar"] .stMarkdown a {
    color: #7c3aed !important;
}

/* Table text wrapping — prevent truncation */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th {
    white-space: normal !important;
    word-wrap: break-word !important;
    overflow-wrap: break-word !important;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 20px;
}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 600;
    color: #7c3aed;
}

/* Info box accent */
.stAlert {
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
# In unsafe HTML we trust

def main():
    try:
        (model, surrogate, X_train, X_test, X_explain, y_train, y_test,
         shap_values_test, shap_expected_value, feature_info,
         precomputed_preds, precomputed_cfs,
         precomputed_cfs_limited,
         precomputed_cfs_kdtree,
         precomputed_cfs_genetic) = load_artifacts()
    except FileNotFoundError as e:
        st.error(
            "Could not load model artifacts. Please run the notebook first "
            "to generate them, then place the output files in the `data/` folder.\n\n"
            f"Error: {e}"
        )
        st.markdown(
            "**Expected folder structure:**\n"
            "```\n"
            "finalfinal_webapp/\n"
            "  data/\n"
            "    models/tabpfn.joblib\n"
            "    models/surrogate.joblib\n"
            "    explainers/shap_values_test.pkl\n"
            "    explainers/shap_expected_value.pkl\n"
            "    explainers/feature_info.pkl\n"
            "    explainers/counterfactual_results.pkl\n"
            "    explainers/counterfactual_results_limited.pkl\n"
            "    artifacts/X_train.csv\n"
            "    artifacts/X_test.csv\n"
            "    artifacts/y_train.csv\n"
            "    artifacts/y_test.csv\n"
            "    artifacts/X_explain.csv\n"
            "    artifacts/df_encoded.csv\n"
            "    artifacts/test_predictions.pkl\n"
            "```"
        )
        st.stop()

    features = feature_info["features"]
    class_names = feature_info["class_names"]
    category_labels = feature_info.get("category_labels", DEFAULT_CATEGORY_LABELS)
    display_names = feature_info.get("display_names", DEFAULT_DISPLAY_NAMES)
    likert_features = feature_info.get("likert_features", DEFAULT_LIKERT_FEATURES)

    # Sidebar
    st.sidebar.image(
        "assets/computer.png"
    )
    st.sidebar.title("XAI for Post-COVID Social Isolation Prediction")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigate to:", [
        "📖 About",
        "📈 EDA",
        "📊 Model Overview",
        "🔍 SHAP Explanations",
        "🔧 What-If Analysis",
        "🔄 Counterfactual Explorer",
        "📚 Bibliography",
    ])

    st.sidebar.markdown("---")
    st.sidebar.markdown("**📋 About the Project**")
    st.sidebar.markdown(f"- Primary model: `TabPFN`")
    if surrogate is not None:
        st.sidebar.markdown(f"- Surrogate: `{type(surrogate).__name__}` (TreeSHAP + What-If)")
    st.sidebar.markdown(f"- Features: **{len(features)}**")
    st.sidebar.markdown(f"- Test samples: **{len(X_test)}**")
    st.sidebar.markdown(f"- SHAP explained: **{len(X_explain)}** samples")
    if precomputed_cfs_kdtree:
        st.sidebar.markdown(
            f"- KDTree CFs: **{len(precomputed_cfs_kdtree)}** samples  \n"
            "  _(real training-set examples)_"
        )
    if precomputed_cfs_genetic:
        st.sidebar.markdown(
            f"- Genetic CFs: **{len(precomputed_cfs_genetic)}** samples"
        )
    if precomputed_cfs:
        st.sidebar.markdown(f"- Random CFs: **{len(precomputed_cfs)}** samples")
    if precomputed_cfs_limited:
        immutable = feature_info.get("immutable_features", [])
        excluded = ", ".join(immutable) if immutable else "demographics"
        st.sidebar.markdown(
            f"- Demographics-Excluded CFs: **{len(precomputed_cfs_limited)}** samples"
            f"  \n  _(keeps {excluded} fixed)_"
        )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**📏 Lubben Social Network Scale (LSNS-6)**")
    st.sidebar.markdown("🟢 Score ≥ 12: **Not Socially Isolated**")
    st.sidebar.markdown("🔴 Score < 12: **Socially Isolated**")

    st.sidebar.markdown("---")
    st.sidebar.markdown("All images sourced from [irasutoya.com](https://www.irasutoya.com/)")

    # Navigation
    if page == "📖 About":
        from model import render_about
        render_about()
    elif page == "📊 Model Overview":
        from model import render_overview
        render_overview(model, X_test, y_test, class_names, precomputed_preds)
    elif page == "📈 EDA":
        from eda import render_eda, _load_encoded_dataset
        df_encoded = _load_encoded_dataset()
        target = feature_info.get("target", "Socially_Isolated")
        render_eda(df_encoded, features, target, display_names,
                   category_labels, likert_features, class_names)
    elif page == "🔍 SHAP Explanations":
        from shap_page import render_shap
        render_shap(model, X_explain, shap_values_test, shap_expected_value,
                    features, class_names, display_names, category_labels,
                    likert_features, precomputed_preds)
    elif page == "🔧 What-If Analysis":
        from counterfactual import render_whatif
        render_whatif(model, surrogate, X_train, X_explain, shap_values_test,
                      shap_expected_value, features, class_names,
                      display_names, category_labels, likert_features,
                      precomputed_preds)
    elif page == "🔄 Counterfactual Explorer":
        from counterfactual import render_counterfactuals
        render_counterfactuals(model, X_train, X_test, y_train, y_test,
                               features, class_names, feature_info,
                               display_names, category_labels, likert_features,
                               precomputed_cfs, precomputed_cfs_limited,
                               precomputed_cfs_kdtree, precomputed_cfs_genetic,
                               precomputed_preds)
    elif page == "📚 Bibliography":
        from model import render_bibliography
        render_bibliography()

if __name__ == "__main__":
    main()