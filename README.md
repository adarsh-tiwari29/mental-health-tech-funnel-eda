# Mental Health in Tech — The Help-Seeking Funnel

> Where between *"I need help"* and *"I got help"* do people actually drop out?

[![Live Dashboard](https://img.shields.io/badge/Live_Dashboard-Open-D64933?style=for-the-badge&logo=streamlit&logoColor=white)](PASTE_YOUR_STREAMLIT_URL_HERE)
[![Notebook](https://img.shields.io/badge/Notebook-View-1B2A41?style=for-the-badge&logo=jupyter&logoColor=white)](Mental_Health_Tech_Funnel_EDA.ipynb)

![Python](https://img.shields.io/badge/Python-3.9+-2A9D8F?style=flat-square&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-data_wrangling-1B2A41?style=flat-square&logo=pandas&logoColor=white)
![Altair](https://img.shields.io/badge/Altair-dashboard-E0A458?style=flat-square)
![Charts](https://img.shields.io/badge/charts-22-6A4C93?style=flat-square)
![Errors](https://img.shields.io/badge/notebook_errors-0-2A9D8F?style=flat-square)

---

## Live Dashboard

**→ [Open the interactive dashboard](PASTE_YOUR_STREAMLIT_URL_HERE)**

Seven sections that follow the funnel, with live filters for gender, age, country, company size and remote-work status. Every chart responds to the filters.

---

## The framing

Most workplace mental health analyses ask *how many people are unwell?*
This one asks **where the funnel leaks.**

| Stage | Question | Measured by |
|:---|:---|:---|
| **1. Need** | Do I have a condition affecting me? | `work_interfere`, `family_history` |
| **2. Awareness** | Do I know what help exists? | `benefits`, `care_options`, `seek_help` |
| **3. Safety** | Is it safe for me to use it? | `anonymity`, `leave`, `obs_consequence` |
| **4. Action** | Did I actually seek treatment? | `treatment` *(target)* |

Framing it this way changes what the numbers mean. A low treatment rate isn't one problem — it's a sum of leaks at four different valves, and each valve has a different fix.

---

## The headline numbers

<table>
<tr>
<td align="center"><b>79.0%</b><br/><sub>report a condition</sub></td>
<td align="center"><b>35.3%</b><br/><sub>know their care options</sub></td>
<td align="center"><b>29.8%</b><br/><sub>trust anonymity</sub></td>
<td align="center"><b>50.6%</b><br/><sub>sought treatment</sub></td>
</tr>
</table>

Need is enormous. Awareness collapses. **The largest leak — over 43 percentage points — sits between Need and Awareness, which is the cheapest stage in the entire funnel to repair.**

---

## Key findings

**Action exceeds Awareness — and that's the most revealing result**
More than half sought treatment (50.6%) despite only 35.3% knowing their care options and 29.8% trusting confidentiality. They went *around* the employer — private care, own insurance, a GP, out of pocket. **The employer pathway is not the main route to treatment for most of these people.**

**Uncertainty is worse than known absence**
Replicated across four independent variables. Employees who *didn't know* whether they had benefits sought treatment at **37.1%** — below those who knew they had **none** (48.2%). Someone who knows their employer offers nothing can arrange private care; someone in uncertainty is paralysed.

**The controllable levers nearly match the uncontrollable ones**
Family history produces a 38.6 pp spread — and isn't a lever. Care-option awareness produces **27.7 pp**, roughly 72% of that effect, and is fixable with a document rather than a budget.

**Structure doesn't matter**
Company size, remote work, industry and age all sit inside the no-meaningful-difference band. A five-person company posts one of the highest treatment rates in the dataset — above companies with more than a thousand employees.

**Stigma, measured**
The survey asks matched mental/physical versions of the same questions, giving a built-in control. Employees are **4.8×** more likely to fear consequences from disclosing a mental health issue, and **4.6×** less willing to raise it in an interview (3.5% vs 16.0%).

**Two problems, not one**
Hierarchical clustering splits the variables into a *support* block and a separate *stigma* block. Provision and psychological safety are structurally independent — a company can score well on one while failing entirely on the other.

**No individual prediction is possible**
The pair plot shows no clean class separation on any variable pair. Conclusion: **universal provision, never targeted risk-scoring.**

---

## Recommendations *(ranked by cost-to-impact)*

| # | Action | Stage | Effect | Cost |
|:--|:---|:---|:---|:---|
| 1 | Audit provision, **then** publish it in plain language | Awareness | 27.7 pp | ~free |
| 2 | Verify anonymity systems, **then** guarantee it in writing | Safety | 65.1% uncertain | ~free |
| 3 | Train the 60 managers, not the 500 employees | Safety | targeted | low |
| 4 | Intervene at the recognition threshold — self-directed only | Need to Action | +56 pp first step | low |
| 5 | Design for the group least likely to self-refer | Action | largest unmet need | low |

> **Sequence matters.** Publishing an anonymity guarantee the systems can't honour destroys the programme permanently — trust in confidentiality is built slowly and broken instantly. Audit first, communicate second.

**KPIs to track annually**

| Metric | Baseline | Target |
|:---|:---|:---|
| Mean employee-confirmed support score | 1.42 / 5 | 3.0 / 5 |
| Share answering "Don't know" on anonymity | 65.1% | under 20% |
| Share who have witnessed negative consequences | 14.6% | zero |

---

## Contents

| File | Description |
|:---|:---|
| `Mental_Health_Tech_Funnel_EDA.ipynb` | Full EDA notebook — 22 charts, UBM structure, outputs pre-rendered |
| `app.py` | Interactive Streamlit dashboard — 7 sections, Altair charts, live filters |
| `survey.csv` | Raw dataset (1,259 × 27) |
| `requirements.txt` | Dependencies for the dashboard |

---

## Running locally

**Notebook**
```bash
pip install pandas numpy matplotlib seaborn scipy jupyter
jupyter notebook Mental_Health_Tech_Funnel_EDA.ipynb
```

Runs end-to-end with **zero errors**. `survey.csv` must sit in the same folder — the loader also checks `data/`, `/content/` and Google Drive paths for Colab.

**Dashboard**
```bash
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

---

## Notebook structure

**1. Know Your Data** — shape, duplicates, nullity matrix
**2. Understanding Your Variables** — variable dictionary grouped by funnel stage
**3. Data Wrangling** — 7-step pipeline, **zero rows dropped**
**4. Visualisation (UBM)**

| Act | Charts | UBM stage | Question |
|:---|:---|:---|:---|
| I — The Population | 1–7 | Univariate | Who answered, and what's their baseline need? |
| II — The Funnel | 8–12 | Univariate | Where do people fall out? |
| III — Who Gets Help | 13–19 | Bivariate | What separates the treated from the untreated? |
| IV — Everything Together | 20–22 | Multivariate | Do the relationships hold jointly? |

**5. Solution to Business Objective** — 5 recommendations ranked by cost-to-impact
**6. Conclusion**

---

## Data wrangling highlights

The raw data required real repair, and one judgement call shaped the whole analysis:

- **`Age`** contained a minimum of **−1,726** and a maximum of **99,999,999,999** — an unvalidated free-text field. Repaired at cell level with median imputation, so no respondent lost their other 26 valid answers.
- **`Gender`** contained **49 distinct free-text spellings** of three underlying categories — case variants, typos (`Malr`, `msle`, `Femake`), and genuine non-binary self-descriptions (`Enby`, `Androgyne`, `fluid`).
- **`work_interfere`** had 264 blanks that were **structurally missing**, not random — the question was conditioned on *having* a condition, so a blank *is* the answer. Preserved as an explicit `"Not Applicable"` category. Mode-imputing it would have inflated "Sometimes" by ~57% and manufactured a correlation that isn't in the data.
- **Zero rows dropped.** Every fix repaired a cell rather than deleting a person — which matters both statistically and ethically for a survey on a stigmatised topic.

---

## Limitations

Stated plainly, because they bound every conclusion above:

- **Correlational only.** No causal claims are supportable. `leave` and `obs_consequence` show clear **reverse-causality** patterns — read literally, `leave` would suggest that making leave *harder* increases treatment-seeking, which is absurd.
- **Sampling.** ~80% US/UK/Canada (India contributes **10 respondents**), ~79% male, three-quarters aged 26–36. Findings transfer to Western tech workforces and poorly elsewhere.
- **Self-reported**, with no clinical validation. Self-selection likely inflates the 50.6% treatment figure.
- **Encoding.** The ordinal encoding behind the correlation heatmap assumes equal spacing between levels — an assumption Chart 16 demonstrates is **false** for exactly the support variables that matter most.
- **Age.** 2014 data, predating both mass remote work and a substantial shift in public discourse around mental health.

---

## Data source

[Open Sourcing Mental Illness (OSMI)](https://osmihelp.org) — 2014 Mental Health in Tech Survey.

---

**Author:** Aniket
*Analytical use only — not clinical guidance.*
