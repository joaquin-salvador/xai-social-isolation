import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import base64
import textwrap

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
EXPLAINERS_DIR = os.path.join(DATA_DIR, "explainers")
ARTIFACTS_DIR = os.path.join(DATA_DIR, "artifacts")

# Feature metadata defaults
DEFAULT_CATEGORY_LABELS = {
    "Sex_Male": {0: "Female",
                 1: "Male"},
    "Is_Married": {0: "Unmarried",
                   1: "Married"},
    "Has_Child": {0: "No Children",
                  1: "Has Children"},
    "Income": {0: "Missing",
               1: "<2.0M JPY",
               2: "2.0-3.9M JPY",
               3: "4.0-5.9M JPY",
               4: "6.0-7.9M JPY",
               5: ">=8.0M JPY"},
    "Income_Missing": {0: "Income Reported",
                       1: "Income Missing"},
    "Job_Employed": {0: "No",
                     1: "Yes"},
    "Job_Homemaker": {0: "No",
                      1: "Yes"},
    "Job_Student": {0: "No",
                    1: "Yes"},
    "Job_Unemployed": {0: "No",
                       1: "Yes"},
    "Job_Other": {0: "No",
                  1: "Yes"},
}

DEFAULT_DISPLAY_NAMES = {
    "Sex_Male": "Sex",
    "Age": "Age",
    "Is_Married": "Marital Status",
    "Has_Child": "Has Children",
    "Income": "Household Income",
    "Income_Missing": "Income Missing",
    "Job_Employed": "Employed",
    "Job_Homemaker": "Homemaker",
    "Job_Student": "Student",
    "Job_Unemployed": "Unemployed",
    "Job_Other": "Other Job",
    "Activity": "Social/Physical Activity",
    "Exercise": "Exercise Frequency",
    "Healthy_Diet": "Healthy Diet",
    "Healthy_Sleep": "Healthy Sleep",
    "Interaction_Offline": "Offline Social Interaction",
    "Interaction_Online": "Online Social Interaction",
    "Altruistic": "Altruistic Behavior",
    "Frustration": "Frustration Level",
    "Optimism": "Optimism Level",
    "Covid_Anxiety": "COVID Anxiety",
    "Covid_Sleepless": "COVID-related Sleeplessness",
    "Deterioration_Economy": "Economic Deterioration",
    "Deterioration_Interact": "Social Interaction Deterioration",
    "Difficulty_Living": "Difficulty Living",
    "Difficulty_Work": "Difficulty Working",
}

DEFAULT_LIKERT_FEATURES = [
    "Activity",
    "Exercise",
    "Healthy_Diet",
    "Healthy_Sleep",
    "Interaction_Offline",
    "Interaction_Online",
    "Altruistic",
    "Frustration",
    "Optimism",
    "Covid_Anxiety",
    "Covid_Sleepless",
    "Deterioration_Economy",
    "Deterioration_Interact",
    "Difficulty_Living",
    "Difficulty_Work"
]

BINARY_FEATURES = {
    "Sex_Male",
    "Is_Married",
    "Has_Child",
    "Income_Missing",
    "Job_Employed",
    "Job_Homemaker",
    "Job_Student",
    "Job_Unemployed",
    "Job_Other",
}

JOB_FEATURES = ["Job_Employed",
                "Job_Homemaker",
                "Job_Student",
                "Job_Unemployed",
                "Job_Other"]
JOB_OPTIONS = {
    "Job_Employed": "Employed",
    "Job_Homemaker": "Homemaker",
    "Job_Student": "Student",
    "Job_Unemployed": "Unemployed",
    "Job_Other": "Other",
}

# Lazy imports
def _import_shap():
    import shap
    return shap

def _import_matplotlib():
    import matplotlib.pyplot as plt
    return plt

def _import_plotly():
    import plotly.express as px
    import plotly.figure_factory as ff
    return px, ff

# Helpers
def get_display_name(feat, display_names):
    return display_names.get(feat, feat.replace("_", " "))

def format_feature_value(feat, value, category_labels, likert_features):
    if feat in category_labels:
        int_val = int(round(value))
        return category_labels[feat].get(int_val, str(int_val))
    elif feat in likert_features:
        return f"{value:.1f} / 7"
    elif feat == "Age":
        return f"{value:.0f} years"
    return f"{value:.2f}"

def build_person_label(idx, row, class_names, precomputed_preds=None):
    """Build a human-readable label for a person selector dropdown."""
    age = row.get("Age", None)
    sex = row.get("Sex_Male", None)
    age_str = f"{int(age)}yo" if age is not None and not np.isnan(age) else ""
    sex_str = ("M" if int(sex) == 1 else "F") if sex is not None and not np.isnan(sex) else ""
    pred_str = ""
    if precomputed_preds is not None:
        pred_str = f" — {class_names[int(precomputed_preds['y_pred'][idx])]}"
    demo = ", ".join(p for p in [sex_str, age_str] if p)
    return f"Person {idx + 1} ({demo}{pred_str})"

def get_prediction(model, X, idx, precomputed_preds=None):
    """Get prediction and probabilities for a sample, using precomputed if available."""
    if precomputed_preds is not None and idx < len(precomputed_preds["y_pred"]):
        pred = int(precomputed_preds["y_pred"][idx])
        prob_high = float(precomputed_preds["y_prob"][idx])
        return pred, np.array([1.0 - prob_high, prob_high])
    pred = model.predict(X.iloc[[idx]])[0]
    prob = model.predict_proba(X.iloc[[idx]])[0]
    return int(pred), prob

def build_person_options(X, class_names, precomputed_preds=None):
    """Build person index list and label dict for a selectbox."""
    options = list(range(len(X)))
    labels = {i: build_person_label(i, X.iloc[i], class_names, precomputed_preds)
              for i in options}
    return options, labels

def build_profile_df(X_row, features, display_names, category_labels,
                     likert_features, include_raw=False):
    """Build a profile DataFrame for display."""
    rows = []
    for feat in features:
        val = X_row[feat]
        entry = {
            "Feature": get_display_name(feat, display_names),
            "Value": format_feature_value(feat, val, category_labels, likert_features),
        }
        if include_raw:
            entry["Raw"] = f"{val:.2f}"
        rows.append(entry)
    return pd.DataFrame(rows)

# Data Loading
@st.cache_resource
def load_artifacts():
    """Load all saved model and explainer artifacts."""
    import joblib

    # Use surrogate model as the primary model (CPU-friendly!!!!!!!!!)
    model_path = os.path.join(MODELS_DIR, "surrogate.joblib")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"surrogate.joblib not found in {MODELS_DIR}. "
            "Make sure you exported it from your notebook."
        )

    model = joblib.load(model_path)

    # Keep surrogate reference
    surrogate = model

    X_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(ARTIFACTS_DIR, "y_test.csv")).squeeze()

    with open(os.path.join(EXPLAINERS_DIR, "shap_values_test.pkl"), "rb") as f:
        shap_values_test = pickle.load(f)

    # Normalize: ensure list-of-2D format [class0, class1]
    if isinstance(shap_values_test, np.ndarray) and shap_values_test.ndim == 3:
        shap_values_test = [shap_values_test[:, :, i]
                            for i in range(shap_values_test.shape[2])]

    with open(os.path.join(EXPLAINERS_DIR, "shap_expected_value.pkl"), "rb") as f:
        shap_expected_value = np.array(pickle.load(f)).flatten()

    x_explain_path = os.path.join(ARTIFACTS_DIR, "X_explain.csv")
    if os.path.exists(x_explain_path):
        X_explain = pd.read_csv(x_explain_path)
    else:
        n_explained = np.array(shap_values_test[0]).shape[0]
        X_explain = X_test.iloc[:n_explained].copy()

    with open(os.path.join(EXPLAINERS_DIR, "feature_info.pkl"), "rb") as f:
        feature_info = pickle.load(f)

    preds_path = os.path.join(ARTIFACTS_DIR, "test_predictions.pkl")
    precomputed_preds = None
    if os.path.exists(preds_path):
        with open(preds_path, "rb") as f:
            precomputed_preds = pickle.load(f)

    def _load_cf_dict(filenames):
        for cf_name in filenames:
            cf_path = os.path.join(EXPLAINERS_DIR, cf_name)
            if not os.path.exists(cf_path):
                continue
            with open(cf_path, "rb") as f:
                loaded = pickle.load(f)
            if isinstance(loaded, dict):
                return loaded
            if isinstance(loaded, list):
                return {item["sample_idx"]: item["counterfactuals"] for item in loaded}
        return None

    precomputed_cfs = _load_cf_dict(
        ["counterfactual_results.pkl", "dice_results.pkl"]
    )
    precomputed_cfs_limited = _load_cf_dict(
        ["counterfactual_results_limited.pkl", "dice_results_limited.pkl"]
    )
    precomputed_cfs_kdtree = _load_cf_dict(
        ["counterfactual_results_kdtree.pkl", "dice_results_kdtree.pkl"]
    )
    precomputed_cfs_genetic = _load_cf_dict(
        ["counterfactual_results_genetic.pkl", "dice_results_genetic.pkl"]
    )

    return (model, surrogate, X_train, X_test, X_explain, y_train, y_test,
            shap_values_test, shap_expected_value, feature_info,
            precomputed_preds, precomputed_cfs, precomputed_cfs_limited,
            precomputed_cfs_kdtree, precomputed_cfs_genetic)


# Architecture Diagram with hover/tap tooltips
ARCHITECTURE_REGIONS = [
    {
        "key": "dataset",
        "coords": (1809, 81, 1878, 142),  # x1, y1, x2, y2
        "title": "Dataset + Data Preprocessing",
        "body": (
            "<ul>"
            "<li>Longitudinal panel data reshaped from <em>wide</em> to "
            "<em>long</em> format across the four survey phases.</li>"
            "<li>Categorical features one-hot encoded (Job type) or binary "
            "encoded (Sex, Marital Status, Has Children).</li>"
            "<li>Missing household income flagged with a dedicated "
            "<code>Income_Missing</code> indicator.</li>"
            "<li>LSNS-6 scores binarized at the cutoff of <strong>12</strong> "
            "&rarr; <em>Not Isolated</em> (&ge;&nbsp;12) vs. "
            "<em>Isolated</em> (&lt;&nbsp;12).</li>"
            "<li>Phases 1&ndash;3 used for training (n&nbsp;=&nbsp;7,977); "
            "Phase&nbsp;4 (2024) held out for testing (n&nbsp;=&nbsp;2,659).</li>"
            "</ul>"
        ),
    },
    {
        "key": "model",
        "coords": (1738, 400, 1805, 443),
        "title": "Model + Explainability Layer",
        "body": (
            "<p><strong>Primary Model: TabPFN</strong></p>"
            "<ul>"
            "<li>TabPFN is a <em>Prior-Data Fitted Network</em> &mdash; a "
            "meta-learned transformer that performs Bayesian inference in a "
            "single forward pass.</li>"
            "<li>It requires no hyperparameter tuning and is especially "
            "strong on tabular data with moderate sample sizes.</li>"
            "<li>The model takes <strong>26 features</strong> "
            "(demographics, lifestyle, COVID impact) and outputs class "
            "probabilities for Not Isolated / Isolated.</li>"
            "</ul>"
            "<p><strong>XGBoost Surrogate</strong></p>"
            "<ul>"
            "<li>An XGBoost classifier (500 trees, depth 6, lr=0.05, "
            "subsample=0.8, colsample_bytree=0.8) is trained to mimic "
            "TabPFN's predictions, with sample weights emphasising "
            "high-confidence TabPFN predictions.</li>"
            "<li>The surrogate enables <strong>exact TreeSHAP</strong> in "
            "polynomial time &mdash; not feasible directly on TabPFN.</li>"
            "<li>It also powers the <strong>What-If Analysis</strong> for "
            "instant interactive predictions.</li>"
            "</ul>"
        ),
    },
    {
        "key": "shap",
        "coords": (1002, 861, 1056, 896),
        "title": "SHAP (SHapley Additive exPlanations)",
        "body": (
            "<p>SHAP produces <strong>both global and local feature "
            "importance</strong> from the same set of values. Averaging "
            "the absolute SHAP values across the test set gives "
            "population-level importance; the per-sample SHAP vector "
            "explains an individual prediction.</p>"
            "<p>The values are grounded in cooperative game theory &mdash; "
            "they are the unique attribution that satisfies efficiency, "
            "symmetry, dummy, and additivity, so contributions sum exactly "
            "to the model output minus its expected value.</p>"
            "<p>We compute <strong>TreeSHAP</strong> on the XGBoost "
            "surrogate, which gives <em>exact</em> SHAP values in polynomial "
            "time &mdash; substantially faster than KernelSHAP on this "
            "dataset.</p>"
        ),
    },
    {
        "key": "dice",
        "coords": (346, 347, 394, 387),
        "title": "DiCE (Diverse Counterfactual Explanations)",
        "body": (
            "<p>DiCE generates <strong>diverse, actionable</strong> "
            "counterfactuals &mdash; multiple alternative scenarios that "
            "would flip the model's prediction, while differing from each "
            "other (<em>diversity</em>) and staying close to the original "
            "instance (<em>proximity</em>).</p>"
            "<p>Crucially, DiCE supports first-class constraints via "
            "<code>features_to_vary</code> and <code>permitted_range</code>, "
            "which let us <em>forbid</em> changes to immutable attributes "
            "such as Age and Sex. This is what makes the "
            "<strong>Demographics-Excluded</strong> counterfactual set "
            "meaningful &mdash; a recommendation a person can actually act "
            "on, not one that requires becoming younger or changing "
            "biological sex.</p>"
            "<p>DiCE is model-agnostic, so the same explainer wraps TabPFN "
            "directly &mdash; no surrogate needed for counterfactual "
            "search.</p>"
        ),
    },
]

def _render_architecture_diagram():
    """Render architecture.png with hover/tap tooltips at each named section."""
    img_path = os.path.join(BASE_DIR, "assets", "architecture.png")
    if not os.path.exists(img_path):
        st.warning("`assets/architecture.png` not found.")
        return
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")

    region_html = []

    IMG_W = 1898
    IMG_H = 984

    for r in ARCHITECTURE_REGIONS:
        x1, y1, x2, y2 = r["coords"]

        left = x1 / IMG_W * 100
        top = y1 / IMG_H * 100
        width = (x2 - x1) / IMG_W * 100
        height = (y2 - y1) / IMG_H * 100

        region_html.append(textwrap.dedent(f"""
            <div class="arch-region arch-region-{r['key']}"
                 style="top:{top}%; left:{left}%;
                        width:{width}%; height:{height}%;">
              <button class="arch-q" type="button"
                      aria-label="{r['title']}"
                      style="top:50%; left:50%;
                             transform:translate(-50%, -50%);">?</button>
              <div class="arch-tooltip" role="tooltip">
                <div class="arch-tooltip-title">{r['title']}</div>
                <div class="arch-tooltip-body">{r['body']}</div>
              </div>
            </div>
        """).strip())

    html = textwrap.dedent("""
    <style>
      .arch-wrap {
          position: relative;
          max-width: 950px;
          margin: 0 auto 24px auto;
          font-family: "Source Sans Pro", -apple-system, BlinkMacSystemFont,
                       sans-serif;
      }
      .arch-img {
          width: 100%;
          height: auto;
          display: block;
          border-radius: 8px;
      }
      .arch-region {
          position: absolute;
          box-sizing: border-box;
      }
      .arch-q {
          position: absolute;
          width: 28px; height: 28px;
          border: none;
          border-radius: 50%;
          background: #7c3aed;
          color: #ffffff;
          font-size: 16px;
          font-weight: 700;
          line-height: 28px;
          padding: 0;
          cursor: pointer;
          box-shadow: 0 2px 6px rgba(0,0,0,0.25);
          z-index: 5;
          transition: transform 0.12s ease,
                      background 0.12s ease;
      }
      .arch-q:hover, .arch-q:focus {
          background: #6d28d9;
          transform: scale(1.12);
          outline: 2px solid #ffffff;
          outline-offset: 1px;
      }
      .arch-tooltip {
          visibility: hidden;
          opacity: 0;
          position: absolute;
          top: 0; left: 50%;
          transform: translate(-50%, -8px);
          width: 320px;
          max-width: 80vw;
          background: #ffffff;
          color: #1a1a2e;
          border: 1px solid #e1e6f0;
          border-radius: 10px;
          box-shadow: 0 8px 24px rgba(20, 30, 60, 0.18);
          padding: 12px 14px;
          font-size: 13px;
          line-height: 1.45;
          z-index: 10;
          transition: opacity 0.15s ease;
          pointer-events: none;
      }
      /* Click-to-open: tooltip shows only while the "?" button (or any
         child) holds focus. This unifies the desktop click and the mobile
         tap experience — no hover behaviour on PC. */
      .arch-region:focus-within .arch-tooltip {
          visibility: visible;
          opacity: 1;
          pointer-events: auto;
      }
      .arch-tooltip-title {
          font-weight: 700;
          color: #7c3aed;
          margin-bottom: 6px;
          font-size: 14px;
      }
      .arch-tooltip-body p { margin: 6px 0; }
      .arch-tooltip-body ul {
          margin: 4px 0 4px 18px;
          padding: 0;
      }
      .arch-tooltip-body li { margin-bottom: 3px; }
      .arch-tooltip-body code {
          background: #f0f2f7;
          padding: 1px 4px;
          border-radius: 3px;
          font-size: 12px;
      }
      /* Right-side tooltips flip to the left so they don't overflow */
      .arch-region-dice .arch-tooltip {
          left: auto; right: 50%;
          transform: translate(50%, -8px);
      }
      .arch-hint {
          text-align: center;
          font-size: 12px;
          color: #6b7280;
          margin-top: -16px;
          margin-bottom: 16px;
      }
    </style>
    <div class="arch-wrap">
      <img src="data:image/png;base64,__IMG__" class="arch-img"
           alt="XAI pipeline architecture diagram" />
      __REGIONS__
    </div>
    <div class="arch-hint">
      💡 Click (or tap on mobile) any <strong>?</strong> icon for the details
      of that pipeline stage. Click outside the popup to close it.
    </div>
    """).replace("__IMG__", img_b64).replace("__REGIONS__", "\n".join(region_html))

    st.markdown(html, unsafe_allow_html=True)

def _render_model_selection():
    """Render the model-selection narrative (baseline benchmark + TabPFN choice)."""

    baseline_df = pd.DataFrame([
        {"Model": "TabPFN", "Accuracy": 0.725085, "F1-Score": 0.796209, "ROC AUC": 0.768854, "Train Time (s)": 0.845071,
         "Test Time (s)": 9.573757},
        {"Model": "Random Forest", "Accuracy": 0.725461, "F1-Score": 0.794944, "ROC AUC": 0.760828,
         "Train Time (s)": 0.650108, "Test Time (s)": 0.028146},
        {"Model": "MLP Classifier", "Accuracy": 0.701015, "F1-Score": 0.769765, "ROC AUC": 0.750659,
         "Train Time (s)": 1.238512, "Test Time (s)": 0.001633},
        {"Model": "Gaussian Naive Bayes", "Accuracy": 0.677322, "F1-Score": 0.744643, "ROC AUC": 0.716987,
         "Train Time (s)": 0.002456, "Test Time (s)": 0.000792},
        {"Model": "K-Nearest Neighbors", "Accuracy": 0.663031, "F1-Score": 0.745599, "ROC AUC": 0.674272,
         "Train Time (s)": 0.001729, "Test Time (s)": 0.197069},
        {"Model": "Support Vector Machine", "Accuracy": 0.705528, "F1-Score": 0.793021, "ROC AUC": 0.636031,
         "Train Time (s)": 1.409051, "Test Time (s)": 0.551041},
        {"Model": "Decision Tree", "Accuracy": 0.649492, "F1-Score": 0.720288, "ROC AUC": 0.630331,
         "Train Time (s)": 0.034107, "Test Time (s)": 0.000845},
    ])

    styled = (baseline_df.style
              .format({"Accuracy": "{:.4f}",
                       "F1-Score": "{:.4f}",
                       "ROC AUC":  "{:.4f}",
                       "Train Time (s)": "{:.2f}",
                       "Test Time (s)":  "{:.2f}"})
              .background_gradient(subset=["Accuracy", "F1-Score", "ROC AUC"],
                                   cmap="Purples")
              .set_properties(**{"text-align": "center"})
              .set_table_styles([
                  {"selector": "th",
                   "props": [("text-align", "center"),
                             ("white-space", "nowrap")]},
              ]))
    st.dataframe(styled, width="stretch", hide_index=True)

# About Page
def render_about():
    """Landing page — study background and pipeline architecture diagram."""
    st.title("📖 About")

    with st.expander("📚 References", expanded=False):
        st.markdown(
            "[1] Sugaya, N., Yamamoto, T., Suzuki, N., & Uchiumi, C. (2020). "
            "\"A real-time survey on the psychological impact of mild lockdown for "
            "COVID-19 in the Japanese population.\" *Scientific Data*, 7(1), 372.\n\n"
            "[2] Yamamoto, T., Uchiumi, C., Suzuki, N., Sugaya, N., et al. (2022). "
            "\"Mental health and social isolation under repeated mild lockdowns in "
            "Japan.\" *Scientific Reports*, 12(1), 8452.\n\n"
            "[3] Ćosić, K., Popović, S., Šarlija, M., & Kesedžić, I. (2021). "
            "\"AI-based prediction and prevention of psychological and behavioral "
            "changes in ex-COVID-19 patients.\" *Frontiers in Psychology*, 12, 782866.\n\n"
            "[4] Sugaya, N., Yamamoto, T., Suzuki, N., & Uchiumi, C. (2024). "
            "\"Loneliness and social isolation factors under the prolonged COVID-19 "
            "pandemic in Japan: 2-year longitudinal study.\" *JMIR Public Health and "
            "Surveillance*, 10, e51653.\n\n"
            "[5] Hollmann, N., Müller, S., Eggensperger, K., & Hutter, F. (2023). "
            "\"TabPFN: A transformer that solves small tabular classification problems "
            "in a second.\" *ICLR 2023.*\n\n"
            "[6] Lundberg, S. M., & Lee, S.-I. (2017). \"A unified approach to "
            "interpreting model predictions.\" *NeurIPS 2017.*\n\n"
            "[7] Mothilal, R. K., Sharma, A., & Tan, C. (2020). \"Explaining machine "
            "learning classifiers through diverse counterfactual explanations.\" "
            "*ACM FAccT 2020*, 607–617.\n\n"
            "[8] Torres, A., Wenke, M., Lieneck, C., Ramamonjiarivelo, Z., & Ari, A. "
            "(2024). \"A systematic review of artificial intelligence used to predict "
            "loneliness, social isolation, and drug use during the COVID-19 "
            "pandemic.\" *Journal of Multidisciplinary Healthcare*, 17, 3403–3425."
        )

    st.subheader("Model Architecture & Pipeline")
    _render_architecture_diagram()

# Model Overview Page
def render_overview(model, X_test, y_test, class_names, precomputed_preds):
    px, ff = _import_plotly()
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

    st.title("📊 Model Overview")

    _render_model_selection()
    st.markdown("---")

    if precomputed_preds is not None:
        y_pred = precomputed_preds["y_pred"]
        y_prob = precomputed_preds["y_prob"]
    else:
        with st.spinner("Running model predictions..."):
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    st.subheader("How well does the model perform?")

    col1, col2, col3 = st.columns(3)
    col1.metric("ROC AUC", f"{auc:.4f}",
                help="How well the model distinguishes between classes. "
                     "1.0 = perfect separation, 0.5 = random guessing.")
    col2.metric("F1-Score", f"{f1:.4f}",
                help="Balance between precision and recall. Ranges 0 to 1.")
    col3.metric("Accuracy", f"{acc:.1%}",
                help="Percentage of correct predictions out of all predictions made.")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = ff.create_annotated_heatmap(
        z=cm.tolist(),
        x=[f"Predicted {n}" for n in class_names],
        y=[f"Actual {n}" for n in class_names],
        annotation_text=[[str(y) for y in x] for x in cm.tolist()],
        colorscale="Purples"
    )
    fig_cm.update_layout(yaxis=dict(autorange="reversed"), height=400)
    st.plotly_chart(fig_cm, width="stretch")

    st.subheader("Prediction Confidence Distribution")
    st.markdown(
        "This histogram shows how confident the model is in its predictions. "
        "Values near 0 mean confident about Not Isolated; "
        "values near 1 mean confident about Socially Isolated."
    )
    prob_df = pd.DataFrame({"Probability (Socially Isolated)": y_prob})
    fig = px.histogram(prob_df, x="Probability (Socially Isolated)", nbins=30,
                       color_discrete_sequence=["#7c3aed"])
    fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                  annotation_text="Decision boundary")
    st.plotly_chart(fig, width="stretch")


def render_bibliography():
    """Full bibliography page listing all references cited in this application."""
    st.title("📚 Bibliography")

    references = [
        {
            "key": "[1]",
            "text": (
                "N. Sugaya, T. Yamamoto, N. Suzuki, and C. Uchiumi, \"A real-time survey on the psychological impact of mild lockdown for COVID-19 in the Japanese population,\" *Scientific Data*, vol. 7, no. 1, p. 372, 2020."
            ),
        },
        {
            "key": "[2]",
            "text": (
                "T. Yamamoto, C. Uchiumi, N. Suzuki, N. Sugaya, et al., \"Mental health and social isolation under repeated mild lockdowns in Japan,\" *Scientific Reports*, vol. 12, no. 1, p. 8452, 2022."
            ),
        },
        {
            "key": "[3]",
            "text": (
                "K. Ćosić, S. Popović, M. Šarlija, and I. Kesedžić, \"AI-Based Prediction and Prevention of Psychological and Behavioral Changes in Ex-COVID-19 Patients,\" *Frontiers in Psychology*, vol. 12, p. 782866, 2021."
            ),
        },
        {
            "key": "[4]",
            "text": (
                "N. Sugaya, T. Yamamoto, N. Suzuki, and C. Uchiumi, \"Loneliness and Social Isolation Factors Under the Prolonged COVID-19 Pandemic in Japan: 2-Year Longitudinal Study,\" *JMIR Public Health and Surveillance*, vol. 10, p. e51653, 2024."
            ),
        },
        {
            "key": "[5]",
            "text": (
                "N. Hollmann, S. Müller, K. Eggensperger, and F. Hutter, \"TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second,\" *arXiv:2207.01848*, 2023."
            ),
        },
        {
            "key": "[6]",
            "text": (
                "S. M. Lundberg and S.-I. Lee, \"A Unified Approach to Interpreting Model Predictions,\" in *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, 2017."
            ),
        },
        {
            "key": "[7]",
            "text": (
                "R. K. Mothilal, A. Sharma, and C. Tan, \"Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations,\" in *Proc. ACM FAccT 2020*, pp. 607–617, 2020."
            ),
        },
        {
            "key": "[8]",
            "text": (
                "A. Torres, M. Wenke, C. Lieneck, Z. Ramamonjiarivelo, and A. Ari, \"A Systematic Review of Artificial Intelligence Used to Predict Loneliness, Social Isolation, and Drug Use During the COVID-19 Pandemic,\" *Journal of Multidisciplinary Healthcare*, vol. 17, pp. 3403–3425, 2024."
            ),
        },
        {
            "key": "[9]",
            "text": (
                "S. Wachter, B. Mittelstadt, and C. Russell, \"Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR,\" *SSRN Electronic Journal*, 2017."
            ),
        },
        {
            "key": "[10]",
            "text": (
                "T. Miller, \"Explanation in Artificial Intelligence: Insights from the Social Sciences,\" *arXiv:1706.07269*, 2018."
            ),
        },
        {
            "key": "[11]",
            "text": (
                "R. Guidotti, \"Counterfactual explanations and how to find them: Literature review and benchmarking,\" *Data Mining and Knowledge Discovery*, vol. 38, no. 5, pp. 2770–2824, 2024."
            ),
        },
        {
            "key": "[12]",
            "text": (
                "N. Kshetry and M. Kantardzic, \"What-If XAI Framework (WiXAI): From Counterfactuals towards Causal Understanding,\" *Journal of Computer and Communications*, vol. 12, no. 6, pp. 169–198, 2024."
            ),
        },
        {
            "key": "[13]",
            "text": (
                "E. Albini, J. Long, D. Dervovic, and D. Magazzeni, \"Counterfactual Shapley Additive Explanations,\" in *Proc. ACM FAccT 2022*, pp. 1054–1070, 2022."
            ),
        },
        {
            "key": "[14]",
            "text": (
                "M. T. Ribeiro, S. Singh, and C. Guestrin, \"'Why Should I Trust You?': Explaining the Predictions of Any Classifier,\" *arXiv:1602.04938*, 2016."
            ),
        },
        {
            "key": "[15]",
            "text": (
                "A. Stickley and M. Ueda, \"Loneliness in Japan during the COVID-19 pandemic: Prevalence, correlates and association with mental health,\" *Psychiatry Research*, vol. 307, p. 114318, 2022."
            ),
        },
        {
            "key": "[16]",
            "text": (
                "P. Engelmann, M. Reinke, C. Stein, S. Salzmann, et al., \"Psychological factors associated with Long COVID: A systematic review and meta-analysis,\" *eClinicalMedicine*, vol. 74, p. 102756, 2024."
            ),
        },
        {
            "key": "[17]",
            "text": (
                "N. Rius Ottenheim et al., \"Predictors of mental health deterioration from pre- to post-COVID-19 outbreak,\" *BJPsych Open*, vol. 8, no. 5, p. e162, 2022."
            ),
        },
        {
            "key": "[18]",
            "text": (
                "K. B. Nuñez, \"A data-driven approach to profiling coping behaviors during the COVID-19 pandemic using automated clustering algorithms,\" B.S. thesis, Dept. of Computer Science, University of the Philippines Diliman, 2023."
            ),
        },
        {
            "key": "[19]",
            "text": (
                "M. Kumar, K. Ramrakhiyani, and H. Garg, \"'Explainable AI' Disease Detection with Reasoning,\" in *Proc. 2024 ICAAIC*, pp. 222–227, 2024."
            ),
        },
        {
            "key": "[20]",
            "text": (
                "Y. Wang, X. Qiu, Y. Yue, X. Guo, et al., \"A Survey on Natural Language Counterfactual Generation,\" *arXiv:2407.03993*, 2024."
            ),
        },
        {
            "key": "[21]",
            "text": (
                "F. Doshi-Velez and B. Kim, \"Towards A Rigorous Science of Interpretable Machine Learning,\" *arXiv:1702.08608*, 2017."
            ),
        },
        {
            "key": "[22]",
            "text": (
                "L. Deckx, M. van den Akker, and F. Buntinx, \"Risk factors for loneliness in patients with cancer: A systematic literature review and meta-analysis,\" *European Journal of Oncology Nursing*, vol. 18, no. 5, pp. 466–477, 2014."
            ),
        },
        {
            "key": "[23]",
            "text": (
                "Y. Zhao and J. Ma, \"Faithful and Interpretable Explanations for Complex Ensemble Time Series Forecasts using Surrogate Models and Forecastability Analysis,\" *arXiv:2510.08739*, 2025."
            ),
        },
    ]

    for ref in references:
        st.markdown(f"**{ref['key']}** {ref['text']}")
        st.markdown("")
