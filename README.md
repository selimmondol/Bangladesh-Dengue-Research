# Dengue Forecasting in Bangladesh (2000–2025)

Research code and monthly data used to investigate dengue surveillance trends, climate associations, machine-learning forecasts, and Bayesian count models in Bangladesh.

> **Research status:** This repository supports an academic manuscript in preparation. The repository is currently private. The dataset contains provisional or derived observations for parts of 2025 and unresolved historical source discrepancies; it should not be treated as a fully verified official surveillance release.

## Research questions

- Do machine-learning models that perform well during validation continue to perform well during subsequent, unusually large dengue outbreaks?
- How do Random Forest and XGBoost compare with simple historical forecasting baselines?
- What do climate predictors, principal component analysis (PCA), feature ablation, and SHAP reveal about model behavior?
- How adequately do Bayesian Negative Binomial (NB) and Zero-Inflated Negative Binomial (ZINB) models represent dengue counts and predictive uncertainty?

## Repository contents

```text
Bangladesh-Dengue-Research/
├── 01_Dengue_EDA_PCA.ipynb
├── Data/
│   └── Bangladesh_Dengue_2000_2025_UPDATED_AUDITED.xlsx
├── .gitignore
└── README.md
```

- **`01_Dengue_EDA_PCA.ipynb`** — research notebook containing analysis code. The availability of all manuscript analyses in this single notebook and clean-environment reproducibility have not yet been independently verified.
- **`Data/Bangladesh_Dengue_2000_2025_UPDATED_AUDITED.xlsx`** — working audited research dataset, including data and accompanying documentation sheets.

## Data overview

- **Geographic coverage:** Bangladesh (national monthly reported dengue counts).
- **Time coverage:** January 2000–December 2025 (312 months).
- **Research dataset totals:** 769,500 reported cases and 3,538 reported deaths, as recorded in this version of the workbook.
- **Variables:** dengue cases and deaths, calendar information, meteorological variables, and source/record-status metadata.
- **Climate data:** NASA POWER reanalysis data extracted for one reference coordinate near 23.685° N, 90.3563° E. These measurements must **not** be interpreted as spatially representative observations for every district in Bangladesh.

### Important data limitations

1. **2025 observations:** Some monthly surveillance records are provisional, and November–December 2025 include derived observations. Consult the workbook's record-status and source documentation before analysis or reuse.
2. **Historical surveillance discrepancies:** Death totals for 2019 and 2023 require source reconciliation; the values in this research workbook may differ from figures published elsewhere.
3. **Provenance:** The workbook includes source and audit metadata, but original surveillance-source traceability and public-redistribution rights require confirmation before any public dataset release.
4. **Surveillance interpretation:** Reported cases and deaths depend on testing, case ascertainment, reporting practices, and revisions; they are not necessarily exhaustive measures of true infections and mortality.

**Do not replace or silently correct observations** without recording the source, rationale, date, and effect on published analyses.

## Analytical approach

The manuscript's analysis includes descriptive epidemiology, seasonal patterns, lagged climate correlations, PCA, Random Forest, XGBoost, historical forecasting baselines, SHAP interpretation, feature ablation, and moving-block bootstrap comparisons. Bayesian NB and ZINB models are analyzed separately for count-model fit and validation-period predictive assessment.

The principal machine-learning evaluation uses:

| Stage | Period | Role |
| --- | --- | --- |
| Initial training | Jan 2001–Dec 2019 | Model development |
| Validation | Jan 2020–Dec 2022 | Model comparison and selection |
| Refit | Jan 2001–Dec 2022 | Training before final evaluation |
| Final test | Jan 2023–Dec 2025 | Retrospective one-month-ahead evaluation |

The final test uses observed historical information available up to the month preceding each prediction. It is **not** a fixed-origin 36-month-ahead forecast. Bayesian NB/ZINB comparisons were evaluated on the 2020–2022 validation period, rather than the 2023–2025 ML test period.

## Running the notebook

1. Install Python and create a virtual environment.
2. Install the packages imported by `01_Dengue_EDA_PCA.ipynb` in your environment.
3. Open the notebook in JupyterLab or VS Code and select the environment's Python kernel.
4. Confirm that local dataset paths in the notebook point to `Data/Bangladesh_Dengue_2000_2025_UPDATED_AUDITED.xlsx`.
5. Run cells in order and inspect generated figures, diagnostics, and outputs.

A pinned dependency file and a fresh-environment end-to-end execution check are **not yet provided**; reproduce results with appropriate caution. Local absolute paths, if present in the notebook, may require adjustment.

## Responsible use and citation

This dataset is assembled for secondary-data research. **Do not describe it as an official Bangladesh dengue database or assume the files carry an open redistribution license.** Check the original data providers' attribution, licensing, and redistribution terms before making the repository public, reusing, or republishing the data. Cite original surveillance and climate sources as well as the related research manuscript when its bibliographic details become available.

## Manuscript and contact

Associated work: *Dengue forecasting in Bangladesh (2000–2025): temporal generalization, historical baselines, and predictive uncertainty* (working descriptive title; manuscript citation pending).

Maintained by [Md Selim Mondol](https://github.com/selimmondol).
