import streamlit as st
import pandas as pd
import numpy as np

from model import (
    _import_shap, _import_matplotlib, _import_plotly,
    get_display_name, format_feature_value,
    build_person_options, build_person_label, get_prediction, build_profile_df,
    BINARY_FEATURES, JOB_FEATURES, JOB_OPTIONS,
)

# Feature grouping for What-If
DEMOGRAPHIC_FEATURES = ["Sex_Male",
                        "Age",
                        "Is_Married",
                        "Has_Child"]
LIFESTYLE_FEATURES = ["Activity",
                      "Exercise",
                      "Healthy_Diet",
                      "Healthy_Sleep",
                      "Interaction_Offline",
                      "Interaction_Online"]
COVID_FEATURES = ["Altruistic",
                  "Frustration",
                  "Optimism",
                  "Covid_Anxiety",
                  "Covid_Sleepless",
                  "Deterioration_Economy",
                  "Deterioration_Interact",
                  "Difficulty_Living",
                  "Difficulty_Work"]

# Helpers
CHANGE_THRESHOLD = 0.01  # min |cf - orig| considered a "real" feature change

# Longitudinal-survey layout: the encoded dataset is laid out wave-by-wave,
# 2,659 unique respondents per phase. So a row index `i` in the training
# table maps to person (i % N_PEOPLE) + 1 in phase (i // N_PEOPLE) + 1.
N_PEOPLE_PER_PHASE = 2659


def _person_phase_from_train_idx(idx):
    """Decode a training-row index into a (person_id, phase) pair."""
    person_id = int(idx) % N_PEOPLE_PER_PHASE + 1
    phase = int(idx) // N_PEOPLE_PER_PHASE + 1
    return person_id, phase


def _is_changed(cf, orig, f, threshold=CHANGE_THRESHOLD):
    return f in cf and f in orig and abs(cf[f] - orig[f]) > threshold


def _disp_list(feats, display_names):
    return ", ".join(get_display_name(f, display_names) for f in feats)


def _change_direction(feat, cf_val, orig_val):
    """Return direction label: Change for categoricals, Increase/Decrease for others."""
    if feat in BINARY_FEATURES or feat in JOB_FEATURES:
        return "Change"
    return "Increase" if cf_val > orig_val else "Decrease"


def _style_direction(val):
    """Cell style for direction column."""
    if val == "Increase":
        return "background-color: #c6efce; color: #006100"
    elif val == "Decrease":
        return "background-color: #ffc7ce; color: #9c0006"
    return "background-color: #fff2cc; color: #7f6003"


def _render_styled_changes(changes_df):
    """Render a styled dataframe with colored Direction column."""
    styler = changes_df.style
    apply = getattr(styler, "map", None) or styler.applymap
    st.dataframe(apply(_style_direction, subset=["Direction"]),
                 width="stretch", hide_index=True)


def _render_feature_control(feat, orig_val, display_names, category_labels,
                            likert_features, feat_min, feat_max, key_prefix="wi"):
    """Render a single feature control widget and return the modified value."""
    dname = get_display_name(feat, display_names)

    if feat in BINARY_FEATURES:
        if feat in category_labels:
            opts = category_labels[feat]
            return float(st.selectbox(dname, list(opts.keys()),
                                      format_func=lambda x, m=opts: m[x],
                                      index=int(orig_val), key=f"{key_prefix}_{feat}"))
        return float(st.selectbox(dname, [0, 1], index=int(orig_val),
                                  key=f"{key_prefix}_{feat}"))
    elif feat == "Income":
        cats = category_labels.get("Income", {})
        return float(st.selectbox(dname, list(cats.keys()),
                                  format_func=lambda x, m=cats: m[x],
                                  index=int(orig_val), key=f"{key_prefix}_{feat}"))
    elif feat in likert_features:
        return float(st.slider(f"{dname} (1-7)", 1, 7,
                               int(round(orig_val)), key=f"{key_prefix}_{feat}"))
    elif feat == "Age":
        return float(st.slider(dname, 19, 92, int(round(orig_val)),
                               key=f"{key_prefix}_{feat}"))
    else:
        lo = min(int(feat_min[feat]), int(round(orig_val)))
        hi = max(int(feat_max[feat]), int(round(orig_val)))
        return float(st.slider(dname, lo, hi, int(round(orig_val)),
                               key=f"{key_prefix}_{feat}"))


def _render_feature_group(title, icon, caption, feat_list, features, original,
                          display_names, category_labels, likert_features,
                          feat_min, feat_max, modified_values, image=None,
                          key_prefix="wi"):
    """Render a group of feature controls with a header."""
    st.markdown(f"#### {icon} {title}")
    st.caption(caption)
    if image:
        img_l, img_c, img_r = st.columns([1, 2, 1])
        with img_c:
            st.image(image, use_container_width=True)
    cols = st.columns(2)
    col_idx = 0
    for feat in feat_list:
        if feat not in features:
            continue
        with cols[col_idx % 2]:
            modified_values[feat] = _render_feature_control(
                feat, float(original[feat]), display_names, category_labels,
                likert_features, feat_min, feat_max, key_prefix=key_prefix)
        col_idx += 1

# Template Explanations
def _explain_whatif(original_values, modified_values, features, display_names,
                    category_labels, likert_features, shap_vals,
                    class_names, orig_pred, mod_pred, orig_prob, mod_prob):
    changed = [(feat, float(original_values[feat]), float(modified_values[feat]),
                shap_vals[features.index(feat)])
               for feat in features
               if abs(float(modified_values[feat]) - float(original_values[feat])) > 1e-9]
    if not changed:
        return "No features were changed. Adjust the sliders to see how the prediction changes."

    changed.sort(key=lambda x: abs(x[3]), reverse=True)
    flipped = orig_pred != mod_pred

    if flipped:
        header = (f"The prediction **changed** from **{class_names[orig_pred]}** "
                  f"to **{class_names[mod_pred]}** "
                  f"(probability: {orig_prob[1]:.1%} -> {mod_prob[1]:.1%}).")
    else:
        header = (f"The prediction **stayed** as **{class_names[orig_pred]}** "
                  f"(probability: {orig_prob[1]:.1%} -> {mod_prob[1]:.1%}).")

    details = []
    for feat, orig, mod, sv in changed[:5]:
        dname = get_display_name(feat, display_names)
        importance = "high" if abs(sv) > 0.05 else "moderate" if abs(sv) > 0.02 else "low"
        direction = "toward Social Isolation" if sv > 0 else "toward Not Isolated"
        details.append(
            f"- **{dname}**: {format_feature_value(feat, orig, category_labels, likert_features)} "
            f"-> {format_feature_value(feat, mod, category_labels, likert_features)} "
            f"({importance} importance, originally pushed {direction})")

    footer = ("The combination of changes tipped the prediction." if flipped
              else "Try adjusting features with higher importance values.")
    return f"{header}\n\n**Changes:**\n" + "\n".join(details) + f"\n\n{footer}"


def _explain_counterfactual(original_values, cf, features, display_names,
                            category_labels, likert_features, class_names, original_pred):
    target = class_names[1 - original_pred]
    changes = [(f, original_values[f], cf[f])
               for f in features if _is_changed(cf, original_values, f)]
    if not changes:
        return "No significant changes were needed in this counterfactual scenario."

    items = "\n".join(
        f"- **{get_display_name(f, display_names)}**: "
        f"{format_feature_value(f, o, category_labels, likert_features)} -> "
        f"{format_feature_value(f, c, category_labels, likert_features)} "
        f"({_change_direction(f, c, o).lower()})"
        for f, o, c in changes)
    return (f"To change the prediction to **{target}**, these changes would be needed:\n\n"
            f"{items}\n\nThese are hypothetical scenarios, not guaranteed real-world outcomes.")

# What-Ifs
def render_whatif(model, surrogate, X_train, X_explain, shap_values_test,
                  shap_expected_value, features, class_names,
                  display_names, category_labels, likert_features,
                  precomputed_preds):
    shap = _import_shap()
    plt = _import_matplotlib()
    predict_model = surrogate if surrogate is not None else model

    st.title("🔧 What-If Analysis")
    st.info(
        "**Live, interactive predictions.** Pick a real person from the test "
        "set, then move the sliders to see how altering their lifestyle, "
        "COVID-impact, or demographic features would shift the model's "
        "social isolation probability [1]. The SHAP waterfall [2] on the right keeps the "
        "*original* attribution visible so you can compare the live "
        "prediction against the explanation for the unchanged person."
        + (" Predictions are served by the **XGBoost surrogate** for "
           "instant response."
           if surrogate else ""))

    options, labels = build_person_options(X_explain, class_names, precomputed_preds)
    sample_idx = st.selectbox("Start from person:", options,
                              format_func=lambda x: labels[x], key="wi_sample")

    original = X_explain.iloc[sample_idx]
    orig_pred, orig_prob = get_prediction(model, X_explain, sample_idx, precomputed_preds)

    feat_min = X_train[features].min()
    feat_max = X_train[features].max()
    modified_values = {}
    key_prefix = f"wi_{sample_idx}"

    controls_col, results_col = st.columns([1, 1])

    with controls_col:
        st.markdown("---")
        st.subheader("Adjust Features")

        _render_feature_group("Demographics", "👤",
                              "Sex, Age, Marital Status, Children, Income, Occupation",
                              DEMOGRAPHIC_FEATURES, features, original,
                              display_names, category_labels, likert_features,
                              feat_min, feat_max, modified_values,
                              image="assets/demographics.png",
                              key_prefix=key_prefix)

        # Income + Occupation side-by-side
        inc_col, occ_col = st.columns(2)
        with inc_col:
            if "Income" in features:
                modified_values["Income"] = _render_feature_control(
                    "Income", float(original["Income"]), display_names,
                    category_labels, likert_features, feat_min, feat_max,
                    key_prefix=key_prefix)
        with occ_col:
            orig_job = next((jf for jf in JOB_FEATURES
                             if jf in features and float(original[jf]) == 1.0), "Job_Employed")
            selected_job = st.selectbox("Occupation", JOB_FEATURES,
                                        format_func=lambda x: JOB_OPTIONS[x],
                                        index=JOB_FEATURES.index(orig_job),
                                        key=f"{key_prefix}_occupation")
            for jf in JOB_FEATURES:
                modified_values[jf] = 1.0 if jf == selected_job else 0.0

        # Auto-derive Income_Missing from Income selection
        if "Income_Missing" in features:
            inc_val = modified_values.get("Income")
            inc_labels = category_labels.get("Income", {})
            inc_label = inc_labels.get(int(inc_val), "") if inc_val is not None else ""
            modified_values["Income_Missing"] = (
                1.0 if "missing" in str(inc_label).lower() else 0.0)

        st.markdown("---")
        _render_feature_group("Lifestyle", "🏃",
                              "Activity, Exercise, Diet, Sleep, Social Interaction",
                              LIFESTYLE_FEATURES, features, original,
                              display_names, category_labels, likert_features,
                              feat_min, feat_max, modified_values,
                              image="assets/lifestyle.png",
                              key_prefix=key_prefix)

        st.markdown("---")
        _render_feature_group("COVID Impact", "🦠",
                              "Anxiety, Sleep, Economic & Social Deterioration, Difficulties",
                              COVID_FEATURES, features, original,
                              display_names, category_labels, likert_features,
                              feat_min, feat_max, modified_values,
                              image="assets/covid.png",
                              key_prefix=key_prefix)

        # Any remaining features not in the groups
        remaining = [f for f in features if f not in modified_values and f not in JOB_FEATURES]
        if remaining:
            st.markdown("---")
            _render_feature_group("Other Features", "📋", "",
                                  remaining, features, original,
                                  display_names, category_labels, likert_features,
                                  feat_min, feat_max, modified_values,
                                  key_prefix=key_prefix)

    # Results
    modified_df = pd.DataFrame([modified_values], columns=features)
    mod_pred = predict_model.predict(modified_df)[0]
    mod_prob = predict_model.predict_proba(modified_df)[0]
    changed_feats = [f for f in features
                     if abs(modified_values[f] - float(original[f])) > 1e-9]

    with results_col:
        st.markdown("---")
        st.subheader("Results")

        if changed_feats:
            # Side-by-side comparison only makes sense once at least one
            # feature has actually been edited.
            left, right = st.columns(2)
            with left:
                st.markdown("##### Original")
                st.metric("Prediction", class_names[orig_pred])
                st.metric("P(Isolated)", f"{orig_prob[1]:.1%}")
            with right:
                st.markdown("##### Modified")
                flipped = mod_pred != orig_pred
                st.metric("Prediction", class_names[mod_pred],
                          delta="Flipped!" if flipped else "No change",
                          delta_color="normal" if flipped else "off")
                st.metric("P(Isolated)", f"{mod_prob[1]:.1%}",
                          delta=f"{mod_prob[1] - orig_prob[1]:+.1%}")

            st.markdown("**What you changed:**")
            st.dataframe(pd.DataFrame([{
                "Feature": get_display_name(f, display_names),
                "Original": format_feature_value(f, float(original[f]),
                                                 category_labels, likert_features),
                "Modified": format_feature_value(f, modified_values[f],
                                                 category_labels, likert_features),
            } for f in changed_feats]), width="stretch", hide_index=True)
        else:
            st.markdown("##### Original")
            st.metric("Prediction", class_names[orig_pred])
            st.metric("P(Isolated)", f"{orig_prob[1]:.1%}")
            st.info("No features changed yet. Adjust the controls on the left "
                    "to see the modified prediction here.")

        # SHAP waterfall
        n_features = len(features)
        disp_names = [get_display_name(f, display_names) for f in features]
        st.markdown("##### 📊 Original SHAP Breakdown")
        base_val = (shap_expected_value[1] if len(shap_expected_value) > 1
                    else float(shap_expected_value[0]))
        explanation = shap.Explanation(
            values=shap_values_test[1][sample_idx], base_values=base_val,
            data=original.values, feature_names=disp_names)
        fig, _ = plt.subplots(figsize=(8, max(6, n_features * 0.25)))
        shap.waterfall_plot(explanation, show=False, max_display=n_features)
        plt.title("Original Prediction Breakdown")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        if changed_feats:
            with st.expander("📖 What does this mean?", expanded=True):
                st.markdown(_explain_whatif(
                    original.to_dict(), modified_values, features, display_names,
                    category_labels, likert_features, shap_values_test[1][sample_idx],
                    class_names, orig_pred, mod_pred, orig_prob, mod_prob))

    # Scale descriptions
    st.markdown("---")
    with st.expander("📏 Feature Scale Descriptions"):
        st.markdown(
            "Likert items are scored from **1 = *Not at all true*** to "
            "**7 = *Very true*** [4]. Each row shows the original survey "
            "question that the respondent answered.\n\n"
            "**Demographics**\n\n"
            "| Feature | Scale |\n|---|---|\n"
            "| Sex | Binary: Male / Female |\n"
            "| Age | Years (19-92) |\n"
            "| Marital Status | Binary: Married / Unmarried |\n"
            "| Has Children | Binary: Yes / No |\n"
            "| Household Income | Ordinal categories (JPY brackets, incl. Missing) |\n"
            "| Occupation | One-hot: Employed, Homemaker, Student, Unemployed, Other |\n\n"
            "**Lifestyle** _(1 = Not at all true … 7 = Very true)_\n\n"
            "| Feature | Survey item |\n|---|---|\n"
            "| Social/Physical Activity | I engaged in hobbies and other activities that I could become passionate about. |\n"
            "| Exercise Frequency | I try to exercise to stay healthy (both indoors and outdoors). |\n"
            "| Healthy Diet | I ate meals with nutritional balance in mind. |\n"
            "| Healthy Sleep | My wake-up and bedtimes were pretty consistent. |\n"
            "| Offline Social Interaction | I interacted with family and friends in person (excluding work and classes). |\n"
            "| Online Social Interaction | I interacted with family and friends online via chat or video calls (excluding work or class). |\n\n"
            "**COVID Impact** _(1 = Not at all true … 7 = Very true)_\n\n"
            "| Feature | Survey item |\n|---|---|\n"
            "| Altruistic Behavior | I voluntarily took preventive actions (mask, hand-washing, limiting going out, etc.) to prevent spreading COVID-19 to family and others. |\n"
            "| Frustration Level | Changes in my life sometimes made me irritable and angry. |\n"
            "| Optimism Level | I thought positively about the future. |\n"
            "| COVID Anxiety | Watching the news about the new coronavirus made me feel nervous and anxious. |\n"
            "| COVID-related Sleeplessness | I couldn't sleep because I was worried about catching the new coronavirus. |\n"
            "| Economic Deterioration | The economic situation worsened. |\n"
            "| Social Interaction Deterioration | Relationships with close people such as family and friends have deteriorated. |\n"
            "| Difficulty Living | Daily life was disrupted by shortages of COVID-19 prevention supplies (masks, thermometers, etc.) and other daily necessities. |\n"
            "| Difficulty Working | Changes in my lifestyle have caused problems with my work and studies. |"
        )

    st.markdown("---")
    with st.expander("📚 References", expanded=False):
        st.markdown(
            "[1] N. Kshetry and M. Kantardzic (2024). \"What-If XAI Framework "
            "(WiXAI): From Counterfactuals towards Causal Understanding.\" "
            "*Journal of Computer and Communications*, 12(6), 169–198.\n\n"
            "[2] S. M. Lundberg and S.-I. Lee (2017). \"A unified approach to "
            "interpreting model predictions.\" *NeurIPS*, vol. 30.\n\n"
            "[3] R. K. Mothilal, A. Sharma, and C. Tan (2020). \"Explaining machine "
            "learning classifiers through diverse counterfactual explanations.\" "
            "*Proc. ACM FAccT 2020*, pp. 607–617.\n\n"
            "[4] N. Sugaya et al. (2020–2024). Longitudinal survey instruments "
            "(Phases 1–4), Tokyo, Osaka, Hyogo, Fukuoka, Japan."
        )

# Counterfactual Explorer helpers
def _scenario_changes_table(cf, original_values, features, display_names,
                            category_labels, likert_features):
    """Return a DataFrame of feature changes for a single scenario."""
    rows = [{
        "Feature": get_display_name(f, display_names),
        "Current": format_feature_value(f, original_values[f], category_labels, likert_features),
        "Needed": format_feature_value(f, cf[f], category_labels, likert_features),
        "Direction": _change_direction(f, cf[f], original_values[f]),
    } for f in features if _is_changed(cf, original_values, f)]
    return pd.DataFrame(rows)


def _check_immutable(cf, original_values, immutable_features):
    """Return list of immutable features that were changed (constraint violations)."""
    if not immutable_features:
        return []
    return [f for f in immutable_features
            if _is_changed(cf, original_values, f)]


def _render_cf_set(cf_records, original_values, features, display_names,
                   category_labels, likert_features, class_names,
                   original_pred, key_prefix, immutable_features=None,
                   mode="scenario", X_train=None, y_train=None):
    """Render scenario navigator + change table + explanation + frequency chart."""
    if not cf_records:
        st.info("No counterfactuals available for this person in this set.")
        return

    if immutable_features:
        st.caption(
            f"🔒 These scenarios keep **{_disp_list(immutable_features, display_names)}** constant."
        )

    n_scenarios = len(cf_records)

    # Navigator (prev/next + "Scenario X of N")
    si_key = f"{key_prefix}_scenario_idx"
    if n_scenarios > 1:
        nav_l, nav_c, nav_r = st.columns([1, 3, 1])
        with nav_l:
            if st.button("◀ Prev", key=f"{key_prefix}_prev",
                         use_container_width=True):
                st.session_state[si_key] = (
                    st.session_state.get(si_key, 0) - 1) % n_scenarios
        with nav_r:
            if st.button("Next ▶", key=f"{key_prefix}_next",
                         use_container_width=True):
                st.session_state[si_key] = (
                    st.session_state.get(si_key, 0) + 1) % n_scenarios
        si = st.session_state.get(si_key, 0)
        with nav_c:
            st.markdown(
                f"<div style='text-align:center;padding-top:5px;'>"
                f"<strong>Scenario {si + 1} of {n_scenarios}</strong></div>",
                unsafe_allow_html=True,
            )
    else:
        si = 0
        st.markdown("**Scenario 1 of 1**")

    cf = cf_records[si]

    # KDTree banner
    src_idx = cf.get("source_train_idx") if mode == "kdtree" else None
    if mode == "kdtree":
        if src_idx is not None:
            person_id, phase = _person_phase_from_train_idx(src_idx)
            st.success(
                f"Person {person_id} (Phase {phase}) was used as the "
                "reference for this counterfactual explanation."
            )
        else:
            st.warning(
                "Source training row could not be matched (likely DiCE "
                "rounding); treat this CF as synthesized."
            )

    # Change table
    changes_df = _scenario_changes_table(
        cf, original_values, features, display_names, category_labels, likert_features)

    if not changes_df.empty:
        _render_styled_changes(changes_df)

        violations = _check_immutable(cf, original_values, immutable_features)
        if immutable_features:
            if violations:
                st.error(
                    "⚠️ Constraint violation — immutable feature(s) changed: "
                    f"{_disp_list(violations, display_names)}"
                )
            else:
                st.success(
                    f"✓ Constraint satisfied: {_disp_list(immutable_features, display_names)} held constant."
                )

        st.markdown("---")
        st.markdown(_explain_counterfactual(
            original_values.to_dict(), cf, features, display_names,
            category_labels, likert_features, class_names, original_pred))
    else:
        st.info("No significant changes in this scenario.")

    # KDTree-only: full profile of the source training row
    if mode == "kdtree" and X_train is not None and src_idx is not None:
        try:
            src_row = X_train.iloc[int(src_idx)]
        except (IndexError, KeyError):
            src_row = None

        # Look up the actual training-set label in y_train
        src_label_str = None
        if y_train is not None:
            try:
                src_label_int = int(y_train.iloc[int(src_idx)])
                src_label_str = class_names[src_label_int]
            except (IndexError, ValueError, KeyError):
                src_label_str = None
        if src_label_str is None:
            src_label_str = class_names[1 - original_pred]

        if src_row is not None:
            person_id, phase = _person_phase_from_train_idx(src_idx)
            st.markdown("---")
            st.markdown(
                f"#### 📋 Full profile of Person {person_id} (Phase {phase})"
            )
            st.markdown(
                f"🎯 **Classification in the training set:** **{src_label_str}**"
            )
            st.caption(
                "This is the actual row from `X_train` that DiCE returned as "
                "the counterfactual. The change table above shows how this "
                "person differs from the test instance you selected; the "
                "table below is their full profile."
            )
            st.dataframe(
                build_profile_df(src_row, features, display_names,
                                 category_labels, likert_features),
                width="stretch", hide_index=True,
            )

    if n_scenarios > 1:
        _render_frequency_chart(cf_records, original_values, features,
                                display_names, key_prefix)


def _render_frequency_chart(cf_records, original_values, features,
                            display_names, key_prefix):
    """Bar chart of how often each feature is changed across all scenarios for the sample."""
    px, _ = _import_plotly()
    st.markdown("---")
    st.subheader("📊 Most Frequently Changed Features")

    feat_counts = {}
    for cf in cf_records:
        for f in features:
            if _is_changed(cf, original_values, f):
                dname = get_display_name(f, display_names)
                feat_counts[dname] = feat_counts.get(dname, 0) + 1

    if not feat_counts:
        st.info("No features were changed across the available scenarios.")
        return

    summary_df = pd.DataFrame(
        sorted(feat_counts.items(), key=lambda x: -x[1]),
        columns=["Feature", "Times Changed"])
    fig = px.bar(summary_df, x="Times Changed", y="Feature",
                 orientation="h", color_discrete_sequence=["steelblue"])
    fig.update_layout(yaxis=dict(autorange="reversed"),
                      height=max(300, len(feat_counts) * 30))
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_freq_chart")


# Aggregate evaluation across CF methods (mirrors notebook §4.6)
METHOD_COLORS = {
    "KDTree": "#7c3aed",
    "Genetic": "#f72585",
    "Random": "#0891b2",
    "Demographics-Excluded": "#fb8500",
}


def _aggregate_method_metrics(cfs_dict, X_test, features):
    """Compute validity / sparsity / proximity / diversity for one method's CFs."""
    n_samples = len(cfs_dict)
    n_with_valid = sum(1 for cfs in cfs_dict.values() if cfs)
    sparsities, proximities, diversities = [], [], []
    n_total = 0

    for sample_idx, cfs in cfs_dict.items():
        if not cfs:
            continue
        try:
            orig = X_test.iloc[int(sample_idx)]
        except (IndexError, KeyError):
            continue
        n_total += len(cfs)
        for cf in cfs:
            diffs = [abs(cf[f] - orig[f]) for f in features if f in cf]
            sparsities.append(sum(1 for d in diffs if d > CHANGE_THRESHOLD))
            proximities.append(sum(diffs))
        if len(cfs) > 1:
            pairs = []
            for i in range(len(cfs)):
                for j in range(i + 1, len(cfs)):
                    pairs.append(sum(
                        abs(cfs[i][f] - cfs[j][f])
                        for f in features
                        if f in cfs[i] and f in cfs[j]
                    ))
            diversities.append(float(np.mean(pairs)))

    return {
        "Validity": n_with_valid / n_samples if n_samples else float("nan"),
        "Total CFs": n_total,
        "Avg CFs / valid sample": (n_total / n_with_valid) if n_with_valid else float("nan"),
        "Avg Sparsity": float(np.mean(sparsities)) if sparsities else float("nan"),
        "Avg Proximity": float(np.mean(proximities)) if proximities else float("nan"),
        "Avg Diversity": float(np.mean(diversities)) if diversities else float("nan"),
    }


def _render_method_evaluation(cf_dicts, X_test, features, display_names):
    """Aggregate per-method evaluation panel"""
    methods_data = {m: d for m, d in cf_dicts.items() if d}
    if len(methods_data) < 2:
        # Nothing meaningful to compare with fewer than two methods present haha
        return

    px, _ = _import_plotly()

    st.markdown("---")
    st.header("📊 Method Evaluation (Aggregate Across All Samples)")
    st.markdown(
        "How the loaded CF methods compare on:\n\n"
        "- **Validity** — fraction of samples with at least one valid CF "
        "(higher is better)\n"
        "- **Sparsity** — number of features changed per CF "
        "(lower is better; sparser = simpler explanation)\n"
        "- **Proximity** — L1 distance from the original instance "
        "(lower is better; closer = smaller intervention)\n"
        "- **Diversity** — average pairwise L1 distance between the "
        "CFs returned for the same input (higher = more variety)\n"
        "- **Plausibility** — KDTree CFs are real training rows; "
        "Random and Genetic CFs are synthesized"
    )

    # Aggregate metrics table
    summaries = []
    for method, cfs_dict in methods_data.items():
        m = _aggregate_method_metrics(cfs_dict, X_test, features)
        m["Method"] = method
        m["Plausibility"] = (
            "real training rows" if method == "KDTree" else "synthesized"
        )
        summaries.append(m)

    df_compare = pd.DataFrame(summaries).set_index("Method")[
        ["Validity", "Total CFs", "Avg CFs / valid sample",
         "Avg Sparsity", "Avg Proximity", "Avg Diversity", "Plausibility"]
    ]
    st.subheader("Aggregate metrics")
    st.dataframe(df_compare.round(3), width="stretch")

    # Bar charts
    metrics = [
        ("Validity", "higher is better"),
        ("Avg Sparsity", "lower is better"),
        ("Avg Proximity", "lower is better"),
        ("Avg Diversity", "higher is better"),
    ]
    rows = [metrics[:2], metrics[2:]]
    for row in rows:
        cols = st.columns(2)
        for col, (metric, hint) in zip(cols, row):
            with col:
                df_metric = df_compare[[metric]].reset_index()
                fig = px.bar(
                    df_metric, x="Method", y=metric,
                    color="Method", text_auto=".3f",
                    color_discrete_map=METHOD_COLORS,
                    title=f"{metric} ({hint})",
                )
                fig.update_layout(showlegend=False, height=320,
                                  margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, width="stretch",
                                key=f"eval_bar_{metric}")

    # Per-feature change frequency, grouped by method
    st.subheader("Per-feature change frequency by method")
    st.caption(
        "Which features does each method tend to touch? Long parallel "
        "bars mean methods agree; large differences mean a method has "
        "its own preferred set of edits."
    )
    feat_count_data = {}
    for method, cfs_dict in methods_data.items():
        counts = {f: 0 for f in features}
        for sample_idx, cfs in cfs_dict.items():
            try:
                orig = X_test.iloc[int(sample_idx)]
            except (IndexError, KeyError):
                continue
            for cf in cfs:
                for f in features:
                    if _is_changed(cf, orig, f):
                        counts[f] += 1
        feat_count_data[method] = counts

    feat_df = pd.DataFrame(feat_count_data)
    feat_df["Display Name"] = [
        get_display_name(f, display_names) for f in feat_df.index
    ]
    method_cols = [c for c in feat_df.columns if c != "Display Name"]
    feat_df = feat_df.sort_values(method_cols[0], ascending=True)

    melted = feat_df.reset_index().melt(
        id_vars=["index", "Display Name"],
        value_vars=method_cols,
        var_name="Method", value_name="Times Changed",
    )
    fig = px.bar(
        melted, x="Times Changed", y="Display Name",
        color="Method", barmode="group", orientation="h",
        color_discrete_map=METHOD_COLORS,
        height=max(500, len(feat_df) * 30),
    )
    fig.update_layout(yaxis_title="")
    st.plotly_chart(fig, width="stretch", key="eval_feat_freq")

    # Spearman rank correlation between methods
    if len(method_cols) > 1:
        try:
            from scipy.stats import spearmanr
        except ImportError:
            spearmanr = None
        if spearmanr is not None:
            st.subheader("Inter-method agreement (Spearman ρ)")
            st.caption(
                "Do the methods agree on which features matter? 1.0 means "
                "identical feature-change rankings; 0.0 means independent."
            )
            corr = pd.DataFrame(index=method_cols, columns=method_cols,
                                dtype=float)
            for m1 in method_cols:
                for m2 in method_cols:
                    rho, _ = spearmanr(feat_df[m1], feat_df[m2])
                    corr.loc[m1, m2] = rho
            st.dataframe(corr.round(3), width="stretch")

    # Per-criterion winners
    st.subheader("Best method per criterion")
    winners = {
        "Validity": df_compare["Validity"].idxmax(),
        "Sparsity": df_compare["Avg Sparsity"].idxmin(),
        "Proximity": df_compare["Avg Proximity"].idxmin(),
        "Diversity": df_compare["Avg Diversity"].idxmax(),
    }
    cols = st.columns(len(winners))
    for col, (criterion, winner) in zip(cols, winners.items()):
        col.metric(criterion, winner)

    # Trade-offs
    with st.expander("📖 Trade-offs", expanded=False):
        st.markdown(
            "- **🌳 KDTree** returns *real* training people who are already "
            "classified in the desired class, so every CF is guaranteed "
            "plausible. In our experiment it also achieved the **lowest "
            "proximity** (≈20.2 L1 distance), meaning the matched real "
            "neighbours were closer to the original instance than the "
            "synthetic CFs from the other methods. The cost is sparsity — "
            "real neighbours differ on many features at once — and "
            "diversity, since the candidate pool is bounded by `X_train`.\n"
            "- **🧬 Genetic** evolves a population of CFs toward valid, "
            "low-cost solutions. In our experiment it produced the **most "
            "diverse** CFs (≈24.9 average pairwise L1 distance) — useful "
            "when you want to show a range of distinct routes to the same "
            "outcome — at the cost of run-time and slightly higher "
            "proximity than KDTree.\n"
            "- **🎲 Random** perturbs the original instance until the "
            "prediction flips. In our experiment it produced the "
            "**sparsest** CFs by a wide margin (≈3.4 features changed per "
            "CF, vs ≈10–12 for the other methods), making each CF the "
            "easiest to read. The trade-off is plausibility: the "
            "perturbations are synthetic and may land on combinations "
            "that don't exist in any real respondent.\n"
            "- **🔒 Demographics-Excluded** is Random with `Age` and "
            "`Sex_Male` held constant — useful when the explanation has "
            "to be actionable for the individual.\n\n"
            "**Method agreement.** Spearman rank correlation on per-feature "
            "change frequency was very high between **KDTree ↔ Genetic "
            "(ρ ≈ 0.99)** and moderate for Random against the other two "
            "(ρ ≈ 0.62–0.64): all three methods touch a similar core set "
            "of features, but Random distributes the edits differently."
        )


# Counterfactual Explorer
def render_counterfactuals(model, X_train, X_test, y_train, y_test, features,
                           class_names, feature_info, display_names,
                           category_labels, likert_features, precomputed_cfs,
                           precomputed_cfs_limited,
                           precomputed_cfs_kdtree,
                           precomputed_cfs_genetic,
                           precomputed_preds):
    st.title("🔄 Counterfactual Explorer")

    immutable_features = feature_info.get("immutable_features", []) or []

    st.info(
        "Counterfactual explanations answer the question every clinician "
        "eventually asks: **\"What would need to change for this person's "
        "prediction to be different?\"** [3] Whereas SHAP [2] attributes the "
        "*current* prediction, DiCE [1] searches for the **smallest, most "
        "diverse, and most actionable changes** that would flip the model's "
        "decision — turning a black-box probability into a concrete "
        "intervention plan."
    )

    # Single source of truth for the four CF methods.
    immut_disp_str = _disp_list(immutable_features, display_names)
    methods = [
        # (key, dict, tab label, summary noun, key_prefix, mode, description, immutable_features for this method)
        ("kdtree", precomputed_cfs_kdtree, "🌳 KDTree", "kdtree", "kdt",
         "kdtree",
         "**🌳 KDTree** — DiCE returns *real training-set rows* of the "
         "desired class. Each CF below points back to the actual person "
         "in `X_train` it came from.",
         None),
        ("genetic", precomputed_cfs_genetic, "🧬 Genetic", "genetic", "gen",
         "scenario",
         "**🧬 Genetic** — evolutionary search over CF candidates. "
         "Synthetic like Random, but the search prefers low-cost, valid CFs.",
         None),
        ("random", precomputed_cfs, "🎲 Random", "random", "rnd",
         "scenario",
         "**🎲 Random** — DiCE perturbs the original instance until the "
         "prediction flips. CFs are *synthetic* (new points in feature space).",
         None),
        ("lim", precomputed_cfs_limited, "🔒 Demographics-Excluded",
         "demographics-excluded", "lim", "scenario",
         (f"**🔒 Demographics-Excluded** — random method, but "
          f"**{immut_disp_str}** are held constant so the suggested changes "
          "are actionable.") if immutable_features else None,
         immutable_features),
    ]
    methods = [m for m in methods if m[1]]  # only methods with data

    if not methods:
        st.warning("No pre-computed counterfactuals found. Run the notebook first.")
        return

    descriptions = [f"- {m[6]}" for m in methods if m[6]]
    if descriptions:
        st.markdown("**Methods available below:**\n\n" + "\n".join(descriptions))

    available_per_method = {m[0]: sorted(m[1].keys()) for m in methods}
    available = sorted(set().union(*available_per_method.values()))

    parts = [f"**{len(available_per_method[m[0]])}** {m[3]}" for m in methods]
    st.success(
        f"Loaded {' / '.join(parts)} CF set(s) "
        f"covering {len(available)} unique persons."
    )

    sample_idx = st.selectbox(
        "Basis point (index in test set):",
        available,
        format_func=lambda x: build_person_label(
            x, X_test.iloc[x], class_names, precomputed_preds),
        key="cf_sample",
    )
    instance = X_test.iloc[[sample_idx]]
    original_values = instance.iloc[0]

    pred, prob = get_prediction(model, X_test, sample_idx, precomputed_preds)
    original_pred = pred if pred is not None else 0

    profile_col, meta_col = st.columns([2, 1])
    with profile_col:
        st.subheader("Selected Person")
        if pred is not None:
            st.markdown(f"**Current Prediction**: {class_names[pred]} "
                        f"({prob[pred]:.1%} confidence)")
        st.dataframe(
            build_profile_df(original_values, features, display_names,
                             category_labels, likert_features),
            width="stretch", hide_index=True,
        )
    with meta_col:
        if immutable_features:
            st.markdown("**🔒 Immutable Features**")
            st.caption("Held constant in the demographics-excluded set.")
            for f in immutable_features:
                st.markdown(
                    f"- **{get_display_name(f, display_names)}**: "
                    f"{format_feature_value(f, float(original_values[f]), category_labels, likert_features)}"
                )
        st.markdown("**CF availability for this person**")
        for m in methods:
            mark = "✅" if sample_idx in available_per_method[m[0]] else "—"
            label = m[2].split(" ", 1)[1] if " " in m[2] else m[2]
            st.markdown(f"- {mark} {label}")

    st.markdown("---")

    tabs = st.tabs([m[2] for m in methods])
    for tab, (key, cf_dict, _label, _noun, key_prefix, mode, _desc, imm) in zip(tabs, methods):
        with tab:
            extra = {"X_train": X_train, "y_train": y_train} if mode == "kdtree" else {}
            _render_cf_set(
                cf_dict.get(sample_idx),
                original_values, features, display_names,
                category_labels, likert_features, class_names,
                original_pred, key_prefix=key_prefix,
                immutable_features=imm,
                mode=mode,
                **extra,
            )

    # Aggregate method evaluation
    _render_method_evaluation(
        cf_dicts={
            "KDTree": precomputed_cfs_kdtree,
            "Genetic": precomputed_cfs_genetic,
            "Random": precomputed_cfs,
            "Demographics-Excluded": precomputed_cfs_limited,
        },
        X_test=X_test,
        features=features,
        display_names=display_names,
    )

    st.markdown("---")
    with st.expander("📚 References", expanded=False):
        st.markdown(
            "[1] R. K. Mothilal, A. Sharma, and C. Tan (2020). \"Explaining machine "
            "learning classifiers through diverse counterfactual explanations.\" "
            "*Proc. ACM FAccT 2020*, pp. 607–617.\n\n"
            "[2] S. M. Lundberg and S.-I. Lee (2017). \"A unified approach to "
            "interpreting model predictions.\" *NeurIPS*, vol. 30.\n\n"
            "[3] S. Wachter, B. Mittelstadt, and C. Russell (2017). "
            "\"Counterfactual explanations without opening the black box: "
            "Automated decisions and the GDPR.\" *Harvard Journal of Law & "
            "Technology*, vol. 31, no. 2."
        )
