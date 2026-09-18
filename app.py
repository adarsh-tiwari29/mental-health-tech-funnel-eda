"""
================================================================================
Mental Health in Tech — The Help-Seeking Funnel
================================================================================
An interactive dashboard built on the 2014 OSMI Mental Health in Tech Survey.

Structured around a four-stage funnel — Need → Awareness → Safety → Action —
rather than a flat variable-by-variable tour, because the interesting question
is not "how many people are unwell" but "where do they drop out".

Run with:
    streamlit run app.py

Author : Adarsh
Data   : Open Sourcing Mental Illness (OSMI), 2014
================================================================================
"""

import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Mental Health in Tech — Funnel Analysis",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------------------
# VISUAL IDENTITY — mirrors the notebook exactly so the two read as one project
# ------------------------------------------------------------------------------
INK = "#1B2A41"
FLAME = "#D64933"
OCHRE = "#E0A458"
TEAL = "#2A9D8F"
PLUM = "#6A4C93"
GREY = "#A8A8A8"

SEQ = [INK, FLAME, OCHRE, TEAL, PLUM, "#B56576"]
TREAT_SCALE = alt.Scale(domain=["Yes", "No"], range=[TEAL, FLAME])
SUPPORT_SCALE = alt.Scale(
    domain=["Yes", "Don't know", "Not sure", "No"],
    range=[TEAL, OCHRE, OCHRE, FLAME],
)

SIZE_ORDER = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
LEAVE_ORDER = ["Very easy", "Somewhat easy", "Don't know",
               "Somewhat difficult", "Very difficult"]
WI_ORDER = ["Never", "Rarely", "Sometimes", "Often", "Not Applicable"]
WI_SEVERITY = ["Never", "Rarely", "Sometimes", "Often"]
SUPPORT_COLS = ["benefits", "care_options", "wellness_program",
                "seek_help", "anonymity"]

# ------------------------------------------------------------------------------
# STYLING — all colours pinned explicitly so callouts stay readable on the
# dark theme (inherited white text would make light-background boxes invisible)
# ------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .hero-title {
        font-size: 2.2rem; font-weight: 800; color: #D64933;
        letter-spacing: -0.5px; margin-bottom: 0.1rem;
    }
    .hero-sub { font-size: 1rem; color: #9A9A9A; margin-bottom: 1.2rem; }
    .stage-pill {
        display: inline-block; padding: 0.18rem 0.7rem; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.4px;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def callout(text, kind="insight"):
    """
    Render a coloured callout box.

    Colours are set with INLINE styles rather than CSS classes: Streamlit's own
    dark-theme rules out-specify custom classes, which would leave dark text on
    a light background invisible. Inline styles always win.
    """
    bg, border, icon = {
        "insight": ("#EAF5F3", TEAL, "▲"),
        "warn": ("#FDF0ED", FLAME, "▲"),
        "note": ("#FBF3E7", OCHRE, "■"),
    }[kind]
    st.markdown(
        f'<div style="background-color:{bg};border-left:5px solid {border};'
        f'padding:0.95rem 1.15rem;border-radius:3px;margin:0.7rem 0;'
        f'color:#1A1A1A;font-size:0.93rem;line-height:1.6;">'
        f'<span style="color:{border};font-weight:800;">{icon}</span>&nbsp;&nbsp;'
        f'{text}</div>',
        unsafe_allow_html=True,
    )


def stage_pill(label, colour):
    """Small coloured pill marking which funnel stage a section belongs to."""
    st.markdown(
        f'<span class="stage-pill" style="background-color:{colour};'
        f'color:#FFFFFF;">{label}</span>',
        unsafe_allow_html=True,
    )


# ==============================================================================
# DATA LOADING — mirrors the notebook's wrangling pipeline exactly
# ==============================================================================
MALE_SET = {
    "male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "man",
    "msle", "mail", "malr", "cis man", "cis male", "male ", "guy (-ish) ^_^",
    "male leaning androgynous", "something kinda male?",
    "ostensibly male, unsure what that really means",
}
FEMALE_SET = {
    "female", "f", "woman", "femake", "female ", "cis female", "femail",
    "cis-female/femme", "female (cis)", "femaile", "femal",
}


def clean_gender(value):
    """Collapse a raw free-text gender string to Male / Female / Other."""
    if pd.isna(value):
        return "Other"
    v = str(value).strip().lower()
    if v in MALE_SET:
        return "Male"
    if v in FEMALE_SET:
        return "Female"
    return "Other"


@st.cache_data(show_spinner="Loading and cleaning survey data…")
def load_data():
    """Load survey.csv and apply the notebook's seven-step cleaning pipeline."""
    for path in ("survey.csv", "data/survey.csv", "./survey.csv"):
        if os.path.exists(path):
            try:
                d = pd.read_csv(path)
                break
            except Exception:
                continue
    else:
        return None

    d["Timestamp"] = pd.to_datetime(d["Timestamp"], errors="coerce")

    # Age: repair the cell, never delete the respondent
    bad = (d["Age"] < 15) | (d["Age"] > 80)
    d.loc[bad, "Age"] = np.nan
    d["Age"] = d["Age"].fillna(d["Age"].median()).astype(int)

    d["Gender_Clean"] = d["Gender"].apply(clean_gender)

    # Missing values, differentiated by cause
    d["self_employed"] = d["self_employed"].fillna(d["self_employed"].mode()[0])
    d["work_interfere"] = d["work_interfere"].fillna("Not Applicable")
    d["state"] = d["state"].fillna("Not in US")
    if "comments" in d.columns:
        d["has_comment"] = d["comments"].notna().astype(int)
        d = d.drop(columns=["comments"])

    # Engineered features
    d["Age_Group"] = pd.cut(
        d["Age"], bins=[14, 25, 35, 45, 60, 80],
        labels=["15-25", "26-35", "36-45", "46-60", "60+"])
    d["support_score"] = sum((d[c] == "Yes").astype(int) for c in SUPPORT_COLS)
    d["treatment_binary"] = (d["treatment"] == "Yes").astype(int)
    d["Country_Group"] = d["Country"].where(
        d["Country"].isin(["United States", "United Kingdom", "Canada"]), "Other")
    return d


full = load_data()

if full is None:
    st.error(
        "**survey.csv not found.**\n\n"
        "Place `survey.csv` beside `app.py` and reload."
    )
    st.stop()


# ==============================================================================
# SIDEBAR — navigation first, then filters
# ==============================================================================
st.sidebar.markdown("### ◆ Navigate")
section = st.sidebar.radio(
    "Section",
    [
        "Overview",
        "① Need — who is affected",
        "② Awareness — who knows",
        "③ Safety — who feels safe",
        "④ Action — who gets help",
        "Driver ranking",
        "Data explorer",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙ Filters")

genders = st.sidebar.multiselect(
    "Gender", sorted(full["Gender_Clean"].unique()),
    default=sorted(full["Gender_Clean"].unique()))

a_lo, a_hi = int(full["Age"].min()), int(full["Age"].max())
age_rng = st.sidebar.slider("Age", a_lo, a_hi, (a_lo, a_hi))

country_opts = full["Country"].value_counts().head(12).index.tolist()
countries = st.sidebar.multiselect(
    "Country (top 12)", country_opts, default=country_opts)

sizes = st.sidebar.multiselect("Company size", SIZE_ORDER, default=SIZE_ORDER)

remote = st.sidebar.select_slider(
    "Remote work", options=["All", "Yes", "No"], value="All")

# ---- Apply filters -----------------------------------------------------------
df = full[
    full["Gender_Clean"].isin(genders)
    & full["Age"].between(*age_rng)
    & full["Country"].isin(countries)
    & full["no_employees"].isin(sizes)
].copy()

if remote != "All":
    df = df[df["remote_work"] == remote]

st.sidebar.markdown("---")
st.sidebar.metric("Respondents in view", f"{len(df):,}",
                  delta=f"{len(df) - len(full):,} vs full sample")
if st.sidebar.button("Reset filters", use_container_width=True):
    st.rerun()

st.sidebar.caption(
    "Source: OSMI Mental Health in Tech Survey, 2014 · "
    "1,259 responses · 27 attributes"
)

if len(df) == 0:
    st.warning("No respondents match these filters. Widen the selection.")
    st.stop()

N = len(df)
BASE = df["treatment_binary"].mean() * 100


# ==============================================================================
# SHARED CHART HELPERS
# ==============================================================================
def rate_table(column, order=None, min_n=1):
    """Treatment rate and sample size for each level of `column`."""
    g = (df.groupby(column, observed=True)
         .agg(rate=("treatment_binary", "mean"),
              n=("treatment_binary", "size"))
         .reset_index())
    g["rate"] = g["rate"] * 100
    g = g[g["n"] >= min_n]
    g[column] = g[column].astype(str)
    g = g.rename(columns={column: "level"})
    if order:
        order_s = [str(o) for o in order]
        g["level"] = pd.Categorical(g["level"], order_s, ordered=True)
        g = g.sort_values("level")
        g["level"] = g["level"].astype(str)
    return g


def base_rule():
    """A dashed reference line at the overall treatment rate."""
    return (alt.Chart(pd.DataFrame({"v": [BASE]}))
            .mark_rule(strokeDash=[5, 4], color=GREY, size=2)
            .encode(x=alt.X("v:Q")))


def rate_dots(column, title, order=None, min_n=1, height=300):
    """
    Horizontal lollipop of treatment rate by category, against the base rate.

    Dots rather than bars: the quantity is a position on a rate scale, not an
    area, and bars would give visual weight proportional to the rate rather
    than to the evidence behind it.
    """
    data = rate_table(column, order=order, min_n=min_n)
    data["label"] = data["rate"].round(1).astype(str) + "%  (n=" + data["n"].astype(str) + ")"

    y_enc = alt.Y("level:N", title=None,
                  sort=list(data["level"]) if order else "-x")

    stem = (alt.Chart(data).mark_rule(color="#DDDDDD", size=3)
            .encode(y=y_enc, x=alt.X("base:Q"), x2="rate:Q")
            .transform_calculate(base=str(BASE)))
    dot = (alt.Chart(data).mark_circle(size=260, stroke="white", strokeWidth=2)
           .encode(
               y=y_enc,
               x=alt.X("rate:Q", title="% who sought treatment",
                       scale=alt.Scale(domain=[0, 100])),
               color=alt.value(TEAL),
               tooltip=["level", alt.Tooltip("rate:Q", format=".1f"), "n"]))
    txt = (alt.Chart(data).mark_text(align="left", dx=14, fontSize=11,
                                     fontWeight="bold", color=INK)
           .encode(y=y_enc, x="rate:Q", text="label:N"))

    return (stem + base_rule() + dot + txt).properties(
        title=title, height=height).configure_view(strokeWidth=0)


def count_bars(column, title, order=None, colour=INK, height=300):
    """Simple count bar chart with percentage labels."""
    vc = df[column].value_counts()
    if order:
        vc = vc.reindex([o for o in order if o in vc.index])
    data = vc.reset_index()
    data.columns = ["level", "count"]
    data["level"] = data["level"].astype(str)
    data["pct"] = (data["count"] / N * 100).round(1)
    data["label"] = data["count"].astype(str) + " (" + data["pct"].astype(str) + "%)"

    x_enc = alt.X("level:N", title=None,
                  sort=[str(o) for o in order] if order else "-y",
                  axis=alt.Axis(labelAngle=0, labelLimit=140))

    bars = (alt.Chart(data).mark_bar(size=42, cornerRadiusEnd=3)
            .encode(x=x_enc,
                    y=alt.Y("count:Q", title="Respondents"),
                    color=alt.value(colour),
                    tooltip=["level", "count", alt.Tooltip("pct:Q", format=".1f")]))
    txt = (alt.Chart(data).mark_text(dy=-9, fontSize=10.5,
                                     fontWeight="bold", color=INK)
           .encode(x=x_enc, y="count:Q", text="label:N"))
    return (bars + txt).properties(title=title, height=height)


def support_stack(column, title, height=150):
    """100% stacked ribbon showing the response split for a support variable."""
    vc = (df[column].value_counts(normalize=True) * 100).reset_index()
    vc.columns = ["response", "pct"]
    vc["pct"] = vc["pct"].round(1)
    vc["q"] = title

    return (alt.Chart(vc).mark_bar(size=46)
            .encode(
                x=alt.X("pct:Q", stack="normalize", title="% of respondents",
                        axis=alt.Axis(format="%")),
                y=alt.Y("q:N", title=None),
                color=alt.Color("response:N", scale=SUPPORT_SCALE,
                                legend=alt.Legend(title="Response",
                                                  orient="bottom")),
                tooltip=["response", alt.Tooltip("pct:Q", format=".1f")])
            .properties(height=height))


# ==============================================================================
# HEADER
# ==============================================================================
st.markdown('<div class="hero-title">The Help-Seeking Funnel</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Mental Health in Tech Survey, OSMI 2014 — '
    'tracing where people drop out between needing help and getting it.</div>',
    unsafe_allow_html=True,
)


# ==============================================================================
# SECTION: OVERVIEW
# ==============================================================================
if section == "Overview":

    # ---- Funnel stage computation -------------------------------------------
    stages = [
        ("All respondents", pd.Series(True, index=df.index), GREY),
        ("① Need — reports a condition",
         df["work_interfere"] != "Not Applicable", PLUM),
        ("② Awareness — knows care options",
         df["care_options"] == "Yes", OCHRE),
        ("③ Safety — confident of anonymity",
         df["anonymity"] == "Yes", FLAME),
        ("④ Action — sought treatment",
         df["treatment"] == "Yes", TEAL),
    ]

    fdata = pd.DataFrame({
        "stage": [s[0] for s in stages],
        "n": [int(s[1].sum()) for s in stages],
        "colour": [s[2] for s in stages],
    })
    fdata["pct"] = (fdata["n"] / N * 100).round(1)
    fdata["order"] = range(len(fdata))
    fdata["label"] = fdata["pct"].astype(str) + "%  (n=" + fdata["n"].astype(str) + ")"

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Respondents", f"{N:,}")
    k2.metric("Report a condition", f"{fdata.loc[1,'pct']:.1f}%")
    k3.metric("Know care options", f"{fdata.loc[2,'pct']:.1f}%")
    k4.metric("Sought treatment", f"{fdata.loc[4,'pct']:.1f}%")

    st.markdown("### The funnel")
    st.caption(
        "Every stage measured against the **same denominator** — all respondents "
        "in view — so the bar widths are directly comparable and the largest "
        "narrowing is visible at a glance."
    )

    bars = (alt.Chart(fdata).mark_bar(height=40, cornerRadius=3)
            .encode(
                y=alt.Y("stage:N", sort=alt.EncodingSortField("order"),
                        title=None, axis=alt.Axis(labelLimit=260,
                                                  labelFontSize=12)),
                x=alt.X("pct:Q", title="% of respondents",
                        scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("colour:N", scale=None, legend=None),
                tooltip=["stage", "n", alt.Tooltip("pct:Q", format=".1f")]))
    lab = (alt.Chart(fdata).mark_text(align="left", dx=10, fontSize=12,
                                      fontWeight="bold", color=INK)
           .encode(y=alt.Y("stage:N", sort=alt.EncodingSortField("order"),
                           title=None),
                   x="pct:Q", text="label:N"))

    st.altair_chart((bars + lab).properties(height=280), use_container_width=True)

    action = fdata.loc[4, "pct"]
    aware = fdata.loc[2, "pct"]
    safe = fdata.loc[3, "pct"]

    if action > aware:
        callout(
            f"<b>Action ({action:.1f}%) exceeds Awareness ({aware:.1f}%) and "
            f"Safety ({safe:.1f}%).</b> That is not an error — it means a large "
            f"share of people who got treatment did so <i>despite</i> not knowing "
            f"their care options and not trusting confidentiality. They went "
            f"around the employer entirely: private care, own insurance, a GP, "
            f"out of pocket. The employer pathway is not the main route to "
            f"treatment for most of these people.",
            "warn",
        )

    st.markdown("### Where the uncertainty sits")

    unc_rows = []
    for col, label in [
        ("anonymity", "Anonymity is protected"),
        ("mental_vs_physical", "MH treated as seriously as PH"),
        ("benefits", "Employer provides MH benefits"),
        ("seek_help", "Resources to seek help"),
        ("care_options", "Knows their care options"),
        ("leave", "Ease of taking leave"),
        ("wellness_program", "Wellness programme covers MH"),
    ]:
        unc_rows.append({
            "question": label,
            "pct": round(df[col].isin(["Don't know", "Not sure"]).mean() * 100, 1),
        })
    unc = pd.DataFrame(unc_rows)
    unc["label"] = unc["pct"].astype(str) + "%"

    stem = (alt.Chart(unc).mark_rule(color="#E5E5E5", size=3)
            .encode(y=alt.Y("question:N", sort="-x", title=None),
                    x=alt.value(0), x2="pct:Q"))
    dots = (alt.Chart(unc).mark_circle(size=250, color=OCHRE,
                                       stroke="white", strokeWidth=2)
            .encode(y=alt.Y("question:N", sort="-x", title=None),
                    x=alt.X("pct:Q", title="% answering \"Don't know\" / \"Not sure\"",
                            scale=alt.Scale(domain=[0, 80])),
                    tooltip=["question", "pct"]))
    txt = (alt.Chart(unc).mark_text(align="left", dx=14, fontSize=11,
                                    fontWeight="bold", color=INK)
           .encode(y=alt.Y("question:N", sort="-x"), x="pct:Q", text="label:N"))

    st.altair_chart((stem + dots + txt).properties(height=300),
                    use_container_width=True)

    callout(
        "On most support questions the largest single response is <b>\"Don't "
        "know\"</b>. As the Awareness section shows, uncertainty is not a neutral "
        "middle ground — it performs <i>at or below</i> a known absence of support.",
        "insight",
    )


# ==============================================================================
# SECTION: ① NEED
# ==============================================================================
elif section == "① Need — who is affected":
    stage_pill("STAGE ① · NEED", PLUM)
    st.markdown("### Who is affected, and how badly?")

    c1, c2, c3 = st.columns(3)
    any_int = df["work_interfere"].isin(["Rarely", "Sometimes", "Often"]).mean() * 100
    c1.metric("Any work interference", f"{any_int:.1f}%")
    c2.metric("Frequent interference",
              f"{(df['work_interfere'] == 'Often').mean()*100:.1f}%")
    c3.metric("Family history",
              f"{(df['family_history'] == 'Yes').mean()*100:.1f}%")

    left, right = st.columns(2)
    with left:
        st.altair_chart(
            count_bars("work_interfere", "Reported work interference",
                       order=WI_ORDER, colour=FLAME),
            use_container_width=True)
    with right:
        st.altair_chart(
            rate_dots("work_interfere", "Treatment rate by severity",
                      order=WI_ORDER),
            use_container_width=True)

    # ---- Threshold analysis -------------------------------------------------
    sev = [df[df["work_interfere"] == s]["treatment_binary"].mean() * 100
           for s in WI_SEVERITY]
    if all(not np.isnan(v) for v in sev):
        steps = pd.DataFrame({
            "step": [f"{WI_SEVERITY[i]} → {WI_SEVERITY[i+1]}" for i in range(3)],
            "jump": [round(sev[i + 1] - sev[i], 1) for i in range(3)],
        })
        steps["label"] = "+" + steps["jump"].astype(str) + " pp"
        mx = steps["jump"].max()
        steps["colour"] = np.where(steps["jump"] == mx, PLUM, "#CCCCCC")

        st.markdown("#### The recognition threshold")
        bar = (alt.Chart(steps).mark_bar(size=60, cornerRadiusEnd=3)
               .encode(x=alt.X("step:N", title=None, sort=None,
                               axis=alt.Axis(labelAngle=0)),
                       y=alt.Y("jump:Q", title="Change in treatment rate (pp)"),
                       color=alt.Color("colour:N", scale=None, legend=None),
                       tooltip=["step", "jump"]))
        lab = (alt.Chart(steps).mark_text(dy=-10, fontSize=12,
                                          fontWeight="bold", color=INK)
               .encode(x=alt.X("step:N", sort=None), y="jump:Q", text="label:N"))
        st.altair_chart((bar + lab).properties(height=280),
                        use_container_width=True)

        callout(
            f"Nearly the whole effect happens in <b>one step</b>: "
            f"<b>+{sev[1]-sev[0]:.1f} pp</b> from \"Never\" to \"Rarely\", versus "
            f"only +{sev[3]-sev[1]:.1f} pp across the entire remaining range. "
            f"Help-seeking is triggered by <b>crossing a threshold of noticing</b>, "
            f"not by severity. The opportunity is upstream — helping people "
            f"recognise early impact themselves.",
            "insight",
        )

    st.markdown("#### Family history")
    st.altair_chart(rate_dots("family_history", "Treatment rate by family history",
                              height=200),
                    use_container_width=True)
    callout(
        "The strongest single relationship in the dataset — and one an employer "
        "cannot change. Its value is as a <b>calibration benchmark</b>: it shows "
        "what a strong driver looks like, so the controllable levers can be "
        "judged against it.",
        "note",
    )


# ==============================================================================
# SECTION: ② AWARENESS
# ==============================================================================
elif section == "② Awareness — who knows":
    stage_pill("STAGE ② · AWARENESS", OCHRE)
    st.markdown("### Does anyone know what support exists?")

    st.markdown("#### The support landscape")
    stacks = alt.vconcat(*[
        support_stack(c, lbl)
        for c, lbl in [
            ("benefits", "MH benefits provided"),
            ("care_options", "Knows care options"),
            ("wellness_program", "Wellness programme"),
            ("seek_help", "Seek-help resources"),
        ]
    ]).resolve_scale(color="shared")
    st.altair_chart(stacks, use_container_width=True)

    st.markdown("#### Does uncertainty behave like absence?")
    st.caption(
        "For each variable, the treatment rate at three response levels. "
        "Watch where the **uncertain** point sits relative to **No**."
    )

    slope_rows = []
    for col, unc_lab in [("care_options", "Not sure"),
                         ("benefits", "Don't know"),
                         ("seek_help", "Don't know"),
                         ("wellness_program", "Don't know")]:
        for pos, lv in enumerate(["No", unc_lab, "Yes"]):
            sub = df[df[col] == lv]
            if len(sub) >= 10:
                slope_rows.append({
                    "variable": col,
                    "position": ["No", "Uncertain", "Yes"][pos],
                    "order": pos,
                    "rate": round(sub["treatment_binary"].mean() * 100, 1),
                    "n": len(sub),
                })
    slope = pd.DataFrame(slope_rows)

    if len(slope):
        lines = (alt.Chart(slope).mark_line(size=3, point=alt.OverlayMarkDef(
                    size=140, filled=True, stroke="white", strokeWidth=2))
                 .encode(
                     x=alt.X("position:N", title=None, sort=["No", "Uncertain", "Yes"],
                             axis=alt.Axis(labelAngle=0, labelFontSize=12)),
                     y=alt.Y("rate:Q", title="% who sought treatment",
                             scale=alt.Scale(domain=[25, 80])),
                     color=alt.Color("variable:N",
                                     scale=alt.Scale(range=[FLAME, INK, PLUM, TEAL]),
                                     legend=alt.Legend(title="Variable",
                                                       orient="right")),
                     tooltip=["variable", "position", "rate", "n"]))
        st.altair_chart(lines.properties(height=380), use_container_width=True)

        # Head-to-head: uncertainty vs known absence
        cmp_rows = []
        for var in slope["variable"].unique():
            sub = slope[slope["variable"] == var].set_index("position")
            if "No" in sub.index and "Uncertain" in sub.index:
                cmp_rows.append({
                    "variable": var,
                    "diff": round(sub.loc["No", "rate"] - sub.loc["Uncertain", "rate"], 1),
                })
        cmp = pd.DataFrame(cmp_rows)
        if len(cmp):
            worse = cmp[cmp["diff"] > 0]
            callout(
                f"<b>Uncertainty sits at or below \"No\" on "
                f"{len(worse)} of {len(cmp)} variables.</b> Someone who knows their "
                f"employer offers nothing can arrange private care. Someone in "
                f"uncertainty is paralysed — resolving it would itself require a "
                f"disclosure they are unwilling to make. "
                f"<b>An unknown benefit is behaviourally worse than no benefit.</b>",
                "warn",
            )

    st.markdown("#### Composite support score")
    c1, c2 = st.columns(2)
    with c1:
        st.altair_chart(count_bars("support_score",
                                   "How many of 5 supports can they confirm?",
                                   colour=OCHRE),
                        use_container_width=True)
    with c2:
        st.altair_chart(rate_dots("support_score",
                                  "Treatment rate by support score"),
                        use_container_width=True)

    st.metric("Mean support score", f"{df['support_score'].mean():.2f} / 5")
    callout(
        "The index counts supports a respondent can <b>confirm</b>, not supports "
        "that <b>exist</b>. A low score is ambiguous between genuine absence and "
        "invisible provision — which is exactly why the provision audit must come "
        "<i>before</i> any awareness campaign.",
        "note",
    )


# ==============================================================================
# SECTION: ③ SAFETY
# ==============================================================================
elif section == "③ Safety — who feels safe":
    stage_pill("STAGE ③ · SAFETY", FLAME)
    st.markdown("### Is it safe to actually use the support?")

    DK = "Don't know"
    c1, c2, c3 = st.columns(3)
    c1.metric("Unsure about anonymity",
              f"{(df['anonymity'] == DK).mean()*100:.1f}%")
    c2.metric("Find leave difficult",
              f"{df['leave'].isin(['Somewhat difficult','Very difficult']).mean()*100:.1f}%")
    c3.metric("Witnessed consequences",
              f"{(df['obs_consequence'] == 'Yes').mean()*100:.1f}%")

    st.altair_chart(
        alt.vconcat(support_stack("anonymity", "Anonymity protected?"),
                    support_stack("leave", "Ease of taking leave")),
        use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.altair_chart(rate_dots("anonymity", "Treatment rate by anonymity"),
                        use_container_width=True)
    with right:
        st.altair_chart(rate_dots("leave", "Treatment rate by leave difficulty",
                                  order=LEAVE_ORDER),
                        use_container_width=True)

    callout(
        "<b>The leave panel runs backwards — read it carefully.</b> People who "
        "find leave <i>difficult</i> seek treatment more, almost certainly because "
        "you only discover the process is hard by trying to use it, and you only "
        "try if you are unwell. The \"easy\" group largely contains people who have "
        "never tested the system. <b>This is reverse causality, not a finding that "
        "harder leave helps.</b>",
        "warn",
    )

    # ---- Stigma gap: mental vs physical -------------------------------------
    st.markdown("#### The stigma gap")
    st.caption(
        "The survey asks matched mental and physical versions of the same "
        "questions, so general health privacy is held constant. Whatever gap "
        "remains is the **mental-health-specific penalty**."
    )

    gap_rows = []
    for label, m_col, p_col, val in [
        ("Would raise in a job interview",
         "mental_health_interview", "phys_health_interview", "Yes"),
        ("Fears negative consequences",
         "mental_health_consequence", "phys_health_consequence", "Yes"),
    ]:
        gap_rows.append({"item": label, "type": "Mental health",
                         "pct": round((df[m_col] == val).mean() * 100, 1)})
        gap_rows.append({"item": label, "type": "Physical health",
                         "pct": round((df[p_col] == val).mean() * 100, 1)})
    gaps = pd.DataFrame(gap_rows)

    conn = (alt.Chart(gaps).mark_line(color="#D6D6D6", size=5)
            .encode(y=alt.Y("item:N", title=None,
                            axis=alt.Axis(labelLimit=240, labelFontSize=12)),
                    x=alt.X("pct:Q", title="% of respondents",
                            scale=alt.Scale(domain=[0, 55])),
                    detail="item:N"))
    pts = (alt.Chart(gaps).mark_circle(size=300, stroke="white", strokeWidth=2)
           .encode(y="item:N", x="pct:Q",
                   color=alt.Color("type:N",
                                   scale=alt.Scale(
                                       domain=["Mental health", "Physical health"],
                                       range=[FLAME, TEAL]),
                                   legend=alt.Legend(title=None, orient="bottom")),
                   tooltip=["item", "type", "pct"]))
    st.altair_chart((conn + pts).properties(height=220),
                    use_container_width=True)

    mh_i = (df["mental_health_interview"] == "Yes").mean() * 100
    ph_i = (df["phys_health_interview"] == "Yes").mean() * 100
    mh_c = (df["mental_health_consequence"] == "Yes").mean() * 100
    ph_c = (df["phys_health_consequence"] == "Yes").mean() * 100

    if mh_i > 0 and ph_c > 0:
        callout(
            f"Employees are <b>{mh_c/ph_c:.1f}×</b> more likely to fear "
            f"consequences from disclosing a mental health issue than a physical "
            f"one, and <b>{ph_i/mh_i:.1f}×</b> less likely to raise it in an "
            f"interview. Because the physical questions act as a built-in control, "
            f"this gap is <b>stigma, measured</b> — not general health privacy.",
            "warn",
        )

    st.markdown("#### Who would they actually talk to?")
    talk_rows = []
    for col, label in [("coworkers", "Coworkers"), ("supervisor", "Supervisor")]:
        for lv in ["Yes", "Some of them", "No"]:
            talk_rows.append({
                "audience": label, "response": lv,
                "pct": round((df[col] == lv).mean() * 100, 1),
            })
    talk = pd.DataFrame(talk_rows)

    st.altair_chart(
        alt.Chart(talk).mark_bar(size=34, cornerRadiusEnd=3).encode(
            x=alt.X("response:N", title=None, sort=["Yes", "Some of them", "No"],
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("pct:Q", title="% of respondents"),
            color=alt.Color("audience:N",
                            scale=alt.Scale(range=[INK, OCHRE]),
                            legend=alt.Legend(title=None, orient="bottom")),
            xOffset="audience:N",
            tooltip=["audience", "response", "pct"],
        ).properties(height=320),
        use_container_width=True)

    cw = (df["coworkers"] == "Yes").mean() * 100
    sv = (df["supervisor"] == "Yes").mean() * 100
    callout(
        f"<b>Counter-intuitive:</b> {sv:.1f}% would be fully open with their "
        f"supervisor versus only {cw:.1f}% with coworkers. Disclosure to a manager "
        f"is instrumental and bounded; disclosure to peers is diffuse and "
        f"uncontrolled. <b>This argues for manager training over broad cultural "
        f"campaigns</b> — and a 500-person company has perhaps 60 managers, which "
        f"is a tractable project.",
        "insight",
    )


# ==============================================================================
# SECTION: ④ ACTION
# ==============================================================================
elif section == "④ Action — who gets help":
    stage_pill("STAGE ④ · ACTION", TEAL)
    st.markdown("### Who actually got help?")

    c1, c2, c3 = st.columns(3)
    c1.metric("Sought treatment", f"{BASE:.1f}%")
    c2.metric("Male treatment rate",
              f"{df[df['Gender_Clean']=='Male']['treatment_binary'].mean()*100:.1f}%"
              if (df["Gender_Clean"] == "Male").any() else "—")
    c3.metric("Female treatment rate",
              f"{df[df['Gender_Clean']=='Female']['treatment_binary'].mean()*100:.1f}%"
              if (df["Gender_Clean"] == "Female").any() else "—")

    left, right = st.columns(2)
    with left:
        st.altair_chart(rate_dots("Gender_Clean", "Treatment rate by gender",
                                  height=220),
                        use_container_width=True)
    with right:
        # Untreated headcount — rates alone under-sell the male finding
        rows = []
        for g in ["Male", "Female", "Other"]:
            sub = df[df["Gender_Clean"] == g]
            if len(sub):
                rows.append({"gender": g,
                             "status": "Did NOT seek treatment",
                             "n": int(len(sub) - sub["treatment_binary"].sum())})
                rows.append({"gender": g, "status": "Sought treatment",
                             "n": int(sub["treatment_binary"].sum())})
        hc = pd.DataFrame(rows)
        st.altair_chart(
            alt.Chart(hc).mark_bar(size=48).encode(
                x=alt.X("gender:N", title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y("n:Q", title="Respondents"),
                color=alt.Color("status:N",
                                scale=alt.Scale(
                                    domain=["Sought treatment",
                                            "Did NOT seek treatment"],
                                    range=["#E0E0E0", FLAME]),
                                legend=alt.Legend(title=None, orient="bottom")),
                tooltip=["gender", "status", "n"],
            ).properties(title="Untreated headcount by gender", height=220),
            use_container_width=True)

    callout(
        "Read this as a <b>help-seeking gap, not a prevalence gap</b>. The variable "
        "is <code style='background:rgba(0,0,0,.07);color:#B3261E;padding:1px 4px;"
        "border-radius:3px;'>treatment</code>, not <i>condition</i> — and the "
        "clinical literature is consistent that men are less likely to seek help "
        "for equivalent need. Since men are ~79% of this workforce, untreated men "
        "are the <b>largest single block of unmet need</b>. "
        "<b>Never read this as \"women have more mental health problems\"</b> — that "
        "conclusion is unsupported and legally exposed.",
        "warn",
    )

    st.markdown("#### Structural variables — a cluster of null results")
    g1, g2 = st.columns(2)
    with g1:
        st.altair_chart(rate_dots("no_employees", "By company size",
                                  order=SIZE_ORDER, min_n=20),
                        use_container_width=True)
    with g2:
        st.altair_chart(rate_dots("Age_Group", "By career stage", min_n=20),
                        use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        st.altair_chart(rate_dots("remote_work", "By remote work", height=180),
                        use_container_width=True)
    with g4:
        st.altair_chart(rate_dots("tech_company", "By tech company", height=180),
                        use_container_width=True)

    callout(
        "Company size, age, remote work and industry all sit close to the base "
        "rate. <b>Structure does not predict treatment-seeking; communication and "
        "safety do.</b> A five-person company posts one of the highest rates in "
        "the dataset — above companies with more than a thousand employees.",
        "insight",
    )


# ==============================================================================
# SECTION: DRIVER RANKING
# ==============================================================================
elif section == "Driver ranking":
    st.markdown("### Every driver, ranked by effect size")
    st.caption(
        "Bar spans the lowest to highest treatment rate across a variable's "
        "levels. Colour marks whether an employer can actually change it — "
        "because effect size only matters alongside controllability."
    )

    MIN_CELL = 20
    CONTROLLABLE = {"care_options", "benefits", "seek_help", "wellness_program",
                    "anonymity", "leave", "mental_vs_physical"}
    drivers = ["work_interfere", "family_history", "care_options", "benefits",
               "Gender_Clean", "anonymity", "obs_consequence", "seek_help",
               "wellness_program", "mental_vs_physical", "leave",
               "no_employees", "Age_Group", "remote_work", "tech_company",
               "self_employed"]

    rows = []
    for col in drivers:
        g = (df.groupby(col, observed=True)["treatment_binary"]
             .agg(["mean", "size"]))
        g = g[g["size"] >= MIN_CELL]
        if len(g) < 2:
            continue
        rows.append({
            "variable": col,
            "low": round(g["mean"].min() * 100, 1),
            "high": round(g["mean"].max() * 100, 1),
            "spread": round((g["mean"].max() - g["mean"].min()) * 100, 1),
            "control": "Employer CAN control" if col in CONTROLLABLE
                       else "Employer cannot control",
        })

    # Guard: with a heavily filtered sample no variable may clear the minimum
    # cell size, leaving `rows` empty — sorting an empty frame on a column that
    # does not exist would raise.
    if not rows:
        st.info(
            f"No variable has at least two levels with n ≥ {MIN_CELL} in the "
            f"current selection. Widen the filters to see the driver ranking."
        )
        st.stop()

    tor = pd.DataFrame(rows).sort_values("spread", ascending=False)

    if len(tor):
        tor["label"] = tor["spread"].astype(str) + " pp"

        bars = (alt.Chart(tor).mark_bar(height=17, cornerRadius=8)
                .encode(
                    y=alt.Y("variable:N", sort="-x", title=None,
                            axis=alt.Axis(labelFontSize=12)),
                    x=alt.X("low:Q", title="% who sought treatment",
                            scale=alt.Scale(domain=[0, 100])),
                    x2="high:Q",
                    color=alt.Color("control:N",
                                    scale=alt.Scale(
                                        domain=["Employer CAN control",
                                                "Employer cannot control"],
                                        range=[TEAL, GREY]),
                                    legend=alt.Legend(title=None, orient="bottom")),
                    tooltip=["variable", "low", "high", "spread", "control"]))
        lab = (alt.Chart(tor).mark_text(align="left", dx=9, fontSize=11,
                                        fontWeight="bold", color=INK)
               .encode(y=alt.Y("variable:N", sort="-x"), x="high:Q",
                       text="label:N"))

        st.altair_chart((bars + lab).properties(height=480),
                        use_container_width=True)

        st.dataframe(
            tor[["variable", "low", "high", "spread", "control"]],
            use_container_width=True, hide_index=True)

        ctrl = tor[tor["control"] == "Employer CAN control"]
        if len(ctrl):
            top_ctrl = ctrl.iloc[0]
            callout(
                f"The top-ranked <b>controllable</b> lever is "
                f"<b>{top_ctrl['variable']}</b> at {top_ctrl['spread']} pp. "
                f"The two variables above it — symptom severity and family history "
                f"— are not levers at all. This is why the ranking is colour-coded: "
                f"effect size alone would send a client chasing something they "
                f"cannot change.",
                "insight",
            )

        callout(
            "<b>Do not add these spreads together.</b> The support variables are "
            "heavily intercorrelated — one latent \"organisational openness\" "
            "factor measured several ways. Summing their individual effects would "
            "forecast a combined lift no programme could deliver, and the "
            "credibility cost of that lands on everything else. "
            "Also note that <b>every relationship here is correlational</b>, and "
            "<code style='background:rgba(0,0,0,.07);color:#B3261E;padding:1px 4px;"
            "border-radius:3px;'>leave</code> ranks high purely on a "
            "reverse-causal relationship.",
            "warn",
        )


# ==============================================================================
# SECTION: DATA EXPLORER
# ==============================================================================
else:
    st.markdown("### Explore the cleaned dataset")

    c1, c2 = st.columns([3, 1])
    with c1:
        cols = st.multiselect(
            "Columns", list(df.columns),
            default=["Age", "Gender_Clean", "Country", "no_employees",
                     "treatment", "family_history", "work_interfere",
                     "care_options", "anonymity", "support_score"])
    with c2:
        nrows = st.number_input("Rows", 10, 1500, 150, 10)

    if cols:
        st.dataframe(df[cols].head(int(nrows)),
                     use_container_width=True, height=420)
        st.download_button(
            "Download filtered CSV",
            df[cols].to_csv(index=False).encode("utf-8"),
            "mental_health_funnel_filtered.csv", "text/csv")

    e1, e2 = st.columns(2)
    with e1:
        st.markdown("**Numeric summary**")
        st.dataframe(df[["Age", "support_score", "treatment_binary"]]
                     .describe().T.round(2), use_container_width=True)
    with e2:
        st.markdown("**Remaining nulls**")
        miss = df.isnull().sum()
        miss = miss[miss > 0]
        if len(miss):
            st.dataframe(miss.reset_index().rename(
                columns={"index": "Column", 0: "Missing"}),
                use_container_width=True, hide_index=True)
        else:
            st.success("Zero nulls — cleaning pipeline verified.")

    st.markdown("**Value counts**")
    vc_col = st.selectbox("Column", list(df.columns), index=7)
    vc = df[vc_col].value_counts(dropna=False).reset_index()
    vc.columns = ["Value", "Count"]
    vc["Percent"] = (vc["Count"] / N * 100).round(2)
    st.dataframe(vc, use_container_width=True, hide_index=True)


# ==============================================================================
# FOOTER
# ==============================================================================
st.markdown("---")
st.caption(
    "**Mental Health in Tech — The Help-Seeking Funnel** · "
    "Data: Open Sourcing Mental Illness (OSMI), 2014 · Built with Streamlit "
    "and Altair. Every relationship shown is correlational; none should be read "
    "as causal. Sample is ~80% US/UK/Canada and ~79% male, so findings "
    "generalise to Western tech workforces and poorly elsewhere. "
    "Analytical use only — not clinical guidance."
)
