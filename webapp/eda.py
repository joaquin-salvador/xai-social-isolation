import os
import streamlit as st
import pandas as pd
import numpy as np

from model import (
    _import_plotly, get_display_name, format_feature_value,
    DATA_DIR, ARTIFACTS_DIR,
)

PHASE_YEARS = {1: 2020, 2: 2021, 3: 2022, 4: 2024}
PHASE_PALETTE = {
    1: "#7c3aed",
    2: "#a855f7",
    3: "#f72585",
    4: "#fb8500",
}

# Wave-by-wave participant counts from the original longitudinal survey.
# (Used to build the Sankey diagram at the top of the EDA page.)
WAVE_INFO = [
    {"phase": 1, "year": 2020, "dates": "May 11–12, 2020",   "n": 6723},
    {"phase": 2, "year": 2021, "dates": "June 14–20, 2021",  "n": 4592},
    {"phase": 3, "year": 2022, "dates": "May 13–30, 2022",   "n": 3892},
    {"phase": 4, "year": 2024, "dates": "May 10–17, 2024",   "n": 2659},
]
DROPOUT_COLOR = "#9aa0a6"


def _hex_to_rgba(hex_color, alpha):
    """Convert a #RRGGBB string into an rgba(r,g,b,a) string Plotly accepts."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _build_wave_sankey():
    """Return a plotly Sankey figure illustrating drop-off across the four waves."""
    import plotly.graph_objects as go

    # Node order: [P1, P2, P3, P4, Dropped after P1, Dropped after P2, Dropped after P3]
    node_labels = [
        f"Phase 1 ({WAVE_INFO[0]['year']})\n{WAVE_INFO[0]['n']:,}",
        f"Phase 2 ({WAVE_INFO[1]['year']})\n{WAVE_INFO[1]['n']:,}",
        f"Phase 3 ({WAVE_INFO[2]['year']})\n{WAVE_INFO[2]['n']:,}",
        f"Phase 4 ({WAVE_INFO[3]['year']})\n{WAVE_INFO[3]['n']:,}",
    ]
    node_colors = [
        PHASE_PALETTE[1], PHASE_PALETTE[2],
        PHASE_PALETTE[3], PHASE_PALETTE[4],
        DROPOUT_COLOR, DROPOUT_COLOR, DROPOUT_COLOR,
    ]

    # Per-wave drop-out counts (from the survey design)
    drop_p1_p2 = WAVE_INFO[0]["n"] - WAVE_INFO[1]["n"]   # 6723 - 4592 = 2131
    drop_p2_p3 = WAVE_INFO[1]["n"] - WAVE_INFO[2]["n"]   # 4592 - 3892 = 700
    drop_p3_p4 = WAVE_INFO[2]["n"] - WAVE_INFO[3]["n"]   # 3892 - 2659 = 1233

    sources = [0, 0, 1, 1, 2, 2]
    targets = [1, 4, 2, 5, 3, 6]
    values = [
        WAVE_INFO[1]["n"], drop_p1_p2,
        WAVE_INFO[2]["n"], drop_p2_p3,
        WAVE_INFO[3]["n"], drop_p3_p4,
    ]
    flow_alpha = 0.55
    drop_alpha = 0.45
    link_colors = [
        _hex_to_rgba(PHASE_PALETTE[2], flow_alpha),
        _hex_to_rgba(DROPOUT_COLOR,    drop_alpha),
        _hex_to_rgba(PHASE_PALETTE[3], flow_alpha),
        _hex_to_rgba(DROPOUT_COLOR,    drop_alpha),
        _hex_to_rgba(PHASE_PALETTE[4], flow_alpha),
        _hex_to_rgba(DROPOUT_COLOR,    drop_alpha),
    ]

    outline_shadow = (
        "-1px -1px 0 black, 1px -1px 0 black, "
        "-1px 1px 0 black, 1px 1px 0 black"
    )
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        textfont=dict(
            color="white", size=13, weight="bold",
            shadow=outline_shadow,
        ),
        node=dict(
            pad=20, thickness=22,
            line=dict(color="white", width=0.5),
            label=node_labels, color=node_colors,
        ),
        link=dict(
            source=sources, target=targets, value=values,
            color=link_colors,
        ),
    ))
    fig.update_layout(
        title="Survey waves: how respondents flow across the four phases",
        font_size=12, height=420,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


def _render_wave_sankey():
    """Render the Sankey diagram at the top of the EDA page."""
    st.subheader("🌊 Longitudinal survey: participant flow across waves")
    st.caption(
        "The dataset comes from a four-wave longitudinal survey of "
        "respondents in Tokyo, Osaka, Hyogo, and Fukuoka. The Sankey "
        "diagram traces how many respondents continued from one wave "
        "to the next and how many dropped out at each step."
    )
    st.plotly_chart(_build_wave_sankey(), width="stretch")
    wave_table = pd.DataFrame([
        {
            "Phase": f"Phase {w['phase']} ({w['year']})",
            "Field period": w["dates"],
            "Participants (n)": f"{w['n']:,}",
            "Lost since previous wave":
                "—" if i == 0
                else f"{WAVE_INFO[i-1]['n'] - w['n']:,}",
        }
        for i, w in enumerate(WAVE_INFO)
    ])
    with st.expander("📋 Wave-by-wave participant counts", expanded=False):
        st.table(_centered(wave_table.set_index("Phase"), precision=0))


@st.cache_data
def _load_encoded_dataset():
    """Load the encoded dataset"""
    path = os.path.join(ARTIFACTS_DIR, "df_encoded.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _phase_label(p):
    return f"Phase {int(p)} ({PHASE_YEARS.get(int(p), '?')})"


def _centered(df, precision=3):
    """Return a Pandas Styler that centers headers + cells"""
    return (df.style
              .format(precision=precision)
              .set_properties(**{"text-align": "center"})
              .set_table_styles([
                  {"selector": "th",
                   "props": [("text-align", "center"),
                             ("white-space", "nowrap")]},
                  {"selector": "td",
                   "props": [("white-space", "nowrap")]},
              ]))


def render_eda(df_encoded, features, target, display_names,
               category_labels, likert_features, class_names):
    """Render the EDA tab."""
    px, _ = _import_plotly()

    st.title("📈 Exploratory Data Analysis")

    st.info(
        "Before diving into the model, this page traces the **longitudinal "
        "structure** of the survey. The dataset spans four waves "
        "(**Phase 1 → 2020**, **Phase 2 → 2021**, **Phase 3 → 2022**, "
        "**Phase 4 → 2024**) [1, 2, 3], and the model is trained on Phases 1–3 and "
        "evaluated on Phase 4. The visualisations below show how respondent "
        "counts and feature distributions evolved across the pandemic — "
        "context that motivates the explainability layer on the next pages."
    )

    # Sankey diagram of survey-wave flow — sits at the top of the page so
    # readers see the longitudinal design before drilling into features.
    _render_wave_sankey()
    st.markdown("---")

    st.markdown(
        "**Drill into the features.** The tabs below let you compare each "
        "feature's distribution across the four waves, look for shifts that "
        "line up with COVID-19 milestones, and see how the *social isolation* "
        "rate moves over time."
    )

    if df_encoded is None or "Phase" not in df_encoded.columns:
        st.warning(
            "`artifacts/df_encoded.csv` is not present. Run the notebook's "
            "**§7. Save & Verify Artifacts** cell to produce it, then "
            "reload this page."
        )
        return

    phases = sorted(df_encoded["Phase"].dropna().unique().astype(int).tolist())
    if not phases:
        st.warning("No phase column data found in df_encoded.csv.")
        return

    # Summary cards
    n_per_phase = int((df_encoded["Phase"] == phases[0]).sum())
    st.metric("Participants per phase", f"{n_per_phase:,}",
              help="The same panel of respondents was surveyed in each "
                   "phase, so the participant count is identical across "
                   "phases.")

    cols = st.columns(len(phases))
    for col, p in zip(cols, phases):
        rate = float(df_encoded.loc[df_encoded["Phase"] == p, target].mean())
        col.metric(_phase_label(p), f"P(High) = {rate:.1%}")

    st.markdown("---")

    tab_overview, tab_feature, tab_target = st.tabs([
        "🌐 All-feature overview",
        "🔬 Drill into one feature",
        "🎯 Target rate by phase",
    ])

    # =================================================================
    # Tab 1 — overview: per-feature mean trend across phases
    # =================================================================
    with tab_overview:
        st.subheader("Per-feature mean ± 1 std across phases")
        st.caption(
            "Each panel shows one feature's **mean per phase** (line) with a "
            "**±1 standard-deviation band** shaded around it to illustrate "
            "the variance within each wave. Watch for shifts that line up "
            "with COVID milestones (lockdowns, reopenings)."
        )

        mean_by_phase = (
            df_encoded.groupby("Phase")[features].mean().T
        )
        std_by_phase = (
            df_encoded.groupby("Phase")[features].std().T
        )
        phase_cols_raw = list(mean_by_phase.columns)  # e.g. [1, 2, 3, 4]
        phase_labels = [_phase_label(p) for p in phase_cols_raw]
        mean_by_phase.columns = phase_labels
        std_by_phase.columns = phase_labels
        mean_by_phase.index.name = "Feature"
        mean_by_phase["Display Name"] = [
            get_display_name(f, display_names) for f in mean_by_phase.index
        ]

        # Layer shaded standard deviation
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go

        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        subplot_titles = [
            get_display_name(f, display_names) for f in features
        ]
        fig = make_subplots(
            rows=n_rows, cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.06, vertical_spacing=0.10,
        )
        for i, feat in enumerate(features):
            r, c = i // n_cols + 1, i % n_cols + 1
            means = [float(mean_by_phase.loc[feat, lbl]) for lbl in phase_labels]
            stds = [float(std_by_phase.loc[feat, lbl]) for lbl in phase_labels]
            upper = [m + s for m, s in zip(means, stds)]
            lower = [m - s for m, s in zip(means, stds)]
            # Shaded band
            fig.add_trace(go.Scatter(
                x=phase_labels, y=upper, mode="lines",
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=phase_labels, y=lower, mode="lines",
                line=dict(width=0), fill="tonexty",
                fillcolor="rgba(70,130,180,0.18)",  # steelblue @ 18% alpha
                hoverinfo="skip", showlegend=False,
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=phase_labels, y=means, mode="lines+markers",
                line=dict(color="steelblue", width=2),
                marker=dict(size=5, color="steelblue"),
                name="Mean", showlegend=False,
                hovertemplate=(
                    f"<b>{get_display_name(feat, display_names)}</b><br>"
                    "%{x}<br>mean = %{y:.3f}<extra></extra>"
                ),
            ), row=r, col=c)
        fig.update_layout(
            height=160 * n_rows, margin=dict(t=50, b=20),
            showlegend=False,
        )
        fig.update_annotations(font_size=10)
        fig.update_xaxes(tickfont=dict(size=7))
        fig.update_yaxes(tickfont=dict(size=8))
        st.plotly_chart(fig, width="stretch")

        with st.expander("📋 Mean ± std per feature per phase (table)"):
            mean_tbl = (mean_by_phase[phase_labels + ["Display Name"]]
                        .reset_index().rename(columns={"Feature": "Raw Feature"})
                        .set_index("Display Name")
                        .round(3))
            std_tbl = std_by_phase[phase_labels].copy()
            std_tbl.index = [
                get_display_name(f, display_names) for f in std_tbl.index
            ]
            std_tbl.index.name = "Display Name"
            st.markdown("**Mean per phase**")
            st.table(_centered(mean_tbl))
            st.markdown("**Standard deviation per phase**")
            st.table(_centered(std_tbl.round(3)))

    # =================================================================
    # Tab 2 — drill into one feature
    # =================================================================
    with tab_feature:
        st.subheader("Feature drill-down")

        feat = st.selectbox(
            "Feature to explore:",
            features,
            format_func=lambda f: get_display_name(f, display_names),
            key="eda_feat",
        )
        feat_label = get_display_name(feat, display_names)

        df_sub = df_encoded[["Phase", feat, target]].dropna()
        df_sub["Phase Label"] = df_sub["Phase"].map(_phase_label)
        df_sub["Class"] = df_sub[target].map(
            {0: class_names[0], 1: class_names[1]}
        )

        # Boxplot or histogram depending on feature shape
        unique_vals = df_sub[feat].nunique()
        if unique_vals <= 7:
            # categorical / Likert — show stacked bar of counts per phase
            counts = (
                df_sub.groupby(["Phase Label", feat]).size()
                .reset_index(name="Count")
            )
            counts[feat_label] = counts[feat].apply(
                lambda v: format_feature_value(
                    feat, v, category_labels, likert_features))
            fig = px.bar(
                counts, x="Phase Label", y="Count", color=feat_label,
                barmode="stack",
                title=f"Counts of {feat_label} per phase",
            )
        else:
            # continuous — boxplot per phase
            fig = px.box(
                df_sub, x="Phase Label", y=feat, color="Phase Label",
                points=False,
                color_discrete_map={
                    _phase_label(p): PHASE_PALETTE[p] for p in phases
                },
                title=f"Distribution of {feat_label} by phase",
            )
            fig.update_layout(showlegend=False)
        st.plotly_chart(fig, width="stretch")

        # Per-class mean ± 1 std per phase. The shaded band illustrates
        # the within-class variance at each wave.
        import plotly.graph_objects as go
        agg = (
            df_sub.groupby(["Phase", "Class"])[feat]
            .agg(["mean", "std"]).reset_index()
        )
        agg["Phase Label"] = agg["Phase"].map(_phase_label)
        class_colors = {
            class_names[0]: ("steelblue", "rgba(70,130,180,0.18)"),
            class_names[1]: ("coral",     "rgba(255,127,80,0.18)"),
        }
        fig2 = go.Figure()
        for cls_name in [class_names[0], class_names[1]]:
            sub = agg[agg["Class"] == cls_name].sort_values("Phase")
            if sub.empty:
                continue
            line_color, band_color = class_colors[cls_name]
            stds = sub["std"].fillna(0.0).values
            means = sub["mean"].values
            x_vals = sub["Phase Label"].tolist()
            upper = (means + stds).tolist()
            lower = (means - stds).tolist()
            fig2.add_trace(go.Scatter(
                x=x_vals, y=upper, mode="lines",
                line=dict(width=0), hoverinfo="skip", showlegend=False,
            ))
            fig2.add_trace(go.Scatter(
                x=x_vals, y=lower, mode="lines",
                line=dict(width=0), fill="tonexty", fillcolor=band_color,
                hoverinfo="skip", showlegend=False,
            ))
            fig2.add_trace(go.Scatter(
                x=x_vals, y=means.tolist(),
                mode="lines+markers", name=cls_name,
                line=dict(color=line_color, width=2),
                marker=dict(size=7, color=line_color),
                hovertemplate=(
                    f"<b>{cls_name}</b><br>%{{x}}<br>"
                    "mean = %{y:.3f}<extra></extra>"
                ),
            ))
        fig2.update_layout(
            title=f"Mean ± 1 std of {feat_label} by phase, split by class",
            xaxis_title="Phase Label", yaxis_title=feat_label,
            legend=dict(title="Class"),
        )
        st.plotly_chart(fig2, width="stretch")

        # Summary table
        summary = df_sub.groupby("Phase")[feat].agg(
            ["count", "mean", "std", "median",
             lambda s: s.quantile(0.25),
             lambda s: s.quantile(0.75)]
        )
        summary.columns = ["count", "mean", "std", "median", "q25", "q75"]
        summary.index = [_phase_label(p) for p in summary.index]
        st.table(_centered(summary.round(3)))

    # =================================================================
    # Tab 3 — target rate by phase
    # =================================================================
    with tab_target:
        st.subheader(f"{class_names[1]} rate by phase")
        st.caption(
            "Share of respondents classified as **Socially Isolated** "
            "(LSNS-6 < 12) [4]. This is the prediction target — its "
            "phase-level shift gives context for the model task."
        )

        rate = (
            df_encoded.groupby("Phase")[target].mean()
            .rename("Socially Isolated Rate").reset_index()
        )
        rate["Phase Label"] = rate["Phase"].map(_phase_label)

        fig = px.bar(
            rate, x="Phase Label", y="Socially Isolated Rate",
            text_auto=".2%",
            color="Phase Label",
            color_discrete_map={
                _phase_label(p): PHASE_PALETTE[p] for p in phases
            },
        )
        fig.update_layout(yaxis_range=[0, 1], showlegend=False)
        st.plotly_chart(fig, width="stretch")

        cnt = (
            df_encoded.groupby(["Phase", target]).size().unstack(fill_value=0)
        )
        cnt.columns = [class_names[c] for c in cnt.columns]
        cnt.index = [_phase_label(p) for p in cnt.index]
        cnt["Total"] = cnt.sum(axis=1)
        st.table(_centered(cnt, precision=0))

    st.markdown("---")
    with st.expander("📚 References", expanded=False):
        st.markdown(
            "[1] N. Sugaya et al. (2021). \"Psychological impact of the COVID-19 "
            "epidemic on college students in Japan.\" *Psychiatry Research.*\n\n"
            "[2] N. Sugaya et al. (2022). \"Mental health in Japan during the "
            "COVID-19 pandemic: A longitudinal survey.\" *Scientific Reports.*\n\n"
            "[3] N. Sugaya, T. Yamamoto, N. Ueda, and M. Suzuki (2024). "
            "\"Long-term mental health impacts of COVID-19: Findings from a "
            "four-wave longitudinal survey in Japan.\" *Journal of Affective "
            "Disorders.*\n\n"
            "[4] J. Lubben, E. Blozik, G. Gillmann, S. Iliffe, et al. (2006). "
            "\"Performance of an abbreviated version of the Lubben Social Network "
            "Scale among three European community-dwelling older adult populations.\" "
            "*The Gerontologist*, 46(4), 503–513."
        )
