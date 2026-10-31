<div align="center">

# Bangladesh Dengue Forecasting

### Machine Learning & Bayesian Count Models | 2000–2025

**Temporal generalization · Outbreak-year evaluation · Predictive uncertainty**

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Research](https://img.shields.io/badge/Project-Reproducible%20Research-0A7F66)](#reproducibility)
[![Data](https://img.shields.io/badge/Data-Bangladesh%202000–2025-44546A)](#dataset)
[![Status](https://img.shields.io/badge/Status-Research%20Code-0B7285)](#repository-structure)

*A reproducible research workflow for forecasting monthly dengue cases in Bangladesh, comparing tree-based machine-learning models with Bayesian count models under temporal distribution shifts.*

</div>

---

## Overview

Dengue case counts are seasonal, overdispersed and subject to abrupt outbreaks. This project investigates how predictive models perform when evaluated on **later years rather than randomly shuffled observations**, with particular attention to the exceptional 2023 outbreak and uncertainty calibration.

The associated manuscript is titled:

> **Forecasting Dengue Outbreaks in Bangladesh Using Machine Learning and Bayesian Count Models: Temporal Generalization and Predictive Uncertainty, 2000–2025**

This repository provides the research code, input data and supporting materials used for the analyses. It is a research workflow, **not a real-time surveillance or clinical decision tool**.

## Research questions

- How do machine-learning and Bayesian count models compare on out-of-time dengue forecasts?
- How stable are their errors during major outbreak periods?
- How well do Bayesian predictive intervals reflect observed uncertainty?
- Do observed model differences persist under moving-block bootstrap sensitivity checks?

## Methods

| Component | Approach |
| --- | --- |
| Machine learning | Random Forest; XGBoost; PCA-assisted variants |
| Baselines | Previous-month persistence; seasonal naïve baseline |
| Bayesian modeling | Negative Binomial (NB); Zero-Inflated Negative Binomial (ZINB) |
| Predictors | Lagged dengue cases, lagged weather variables, seasonal sine/cosine features |
| Interpretation | SHAP feature importance; feature ablation |
| Reliability | Posterior predictive checks; MCMC convergence diagnostics |
| Uncertainty | 89% and 95% credible intervals; 90% forecast intervals |
| Sensitivity | Moving-block bootstrap with 3-, 6- and 12-month blocks |

### Temporal evaluation

The analysis has **distinct evaluation setups**; do not combine their scores as if they came from the same split.

- **Bayesian NB/ZINB:** train on **2001–2019** (228 months), validate on **2020–2022** (36 months).
- **Final machine-learning outbreak evaluation:** evaluate on **2023–2025** (36 months).
- Additional ML validation and diagnostic analyses are implemented in the script.

## Selected results

### Bayesian validation: 2020–2022

| Metric | NB | ZINB |
| :--- | ---: | ---: |
| Mean absolute error (MAE) | 2,243.86 | **2,170.10** |
| Root mean squared error (RMSE) | 5,157.95 | **5,033.03** |
| Empirical coverage of nominal 90% interval | **88.89%** | 86.11% |
| Mean predictive interval width (cases) | 3,951.42 | **3,229.30** |

ZINB has lower point-forecast errors and narrower intervals on this validation period, but **its empirical 90% interval coverage is lower**. These findings should not be interpreted as uniformly superior calibration.

### Outbreak-year stress test: September 2023

| Series / model | Monthly cases |
| --- | ---: |
| **Observed** | **79,598** |
| Previous month | 71,976 |
| Random Forest prediction | 22,508.54 |
| XGBoost prediction | 20,264.54 |

Both machine-learning models substantially underpredicted the outbreak peak. This illustrates an important limitation of models trained on earlier patterns when epidemic severity shifts sharply.

### Bayesian diagnostics

| Diagnostic | NB | ZINB |
| --- | ---: | ---: |
| Maximum R-hat | 1.003 | 1.003 |
| Minimum bulk effective sample size | 2,032 | 1,958 |
| Divergent transitions | 0 | 0 |

Sampling employed four chains, with 1,000 tuning steps and 1,000 retained draws per chain. Convergence diagnostics are satisfactory, **but convergence does not guarantee predictive adequacy**.

> **Interpretation note:** These figures are specific to the evaluation periods shown above. The machine-learning outbreak-year figures and Bayesian 2020–2022 validation figures are not directly comparable as head-to-head estimates from one shared test set.

## Dataset

- **Geography:** Bangladesh
- **Frequency:** Monthly
- **Coverage:** January 2000 – December 2025 (312 records)
- **Main workbook:** `Data/Bangladesh_Dengue_2000_2025_Data.xlsx`
- **Main worksheet:** `Research_Data`
- **Outcome:** Monthly reported dengue cases
- **Additional variables:** Dengue deaths and meteorological indicators, subject to data availability and provenance notes

**Data quality:** Some historical records require further primary-source verification. Parts of the 2025 series are marked provisional or derived in the provenance audit. These values should not be represented as independently verified official monthly figures. Consult the S1 provenance table and manuscript limitations before reuse.

## Repository structure

```text
Bangladesh-Dengue-Research/
├── Data/
│   └── Bangladesh_Dengue_2000_2025_Data.xlsx
├── Dengue_Research.py        # Main research workflow
├── requirements.txt          # Python dependencies
├── README.md
├── .gitignore
└── outputs/                  # Generated locally; availability in GitHub may vary
```

The `outputs/` directory is created or populated by running the code. Files shown as generated outputs are not necessarily committed to the repository.

## Reproducibility

### 1. Clone

```bash
git clone https://github.com/selimmondol/Bangladesh-Dengue-Research.git
cd Bangladesh-Dengue-Research
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate with `.venv\Scripts\activate`.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the research script

```bash
python Dengue_Research.py
```

The script expects the dataset at the relative path shown above. Run it **from the repository root**. It performs multiple analyses, including Bayesian sampling, and saves generated artifacts under `outputs/`. Runtime depends on the machine and Python environment.

### 5. Verify outputs

Check the terminal summaries and generated files for:

- Model validation and outbreak-year diagnostics
- Moving-block bootstrap and sensitivity results (**S2**)
- Annual data provenance (**S1**)
- Bayesian NB/ZINB posterior and predictive diagnostics (**S3**)

For reproducibility, retain the exact input dataset, package environment and reporting period used for the manuscript.

## Supplementary materials

| Supplement | Content |
| --- | --- |
| **S1 Table** | Annual dengue cases and deaths with source/provenance annotations |
| **S2 Table** | Moving-block bootstrap comparisons and block-length sensitivity |
| **S3 Table** | Bayesian MCMC diagnostics, posterior intervals, posterior predictive checks and 2020–2022 validation |

These tables are part of the manuscript's supporting analyses. Their presence in this README **does not imply that the final journal-ready spreadsheets are committed to GitHub**.

## Limitations

1. Reported cases reflect surveillance and reporting processes, not necessarily all infections.
2. Some data sources and recent monthly allocations remain provisional or await primary-source verification.
3. Extreme outbreak peaks are difficult to extrapolate from earlier epidemiological patterns.
4. Forecast uncertainty can be miscalibrated even when Bayesian sampling diagnostics look good.
5. Results depend on the particular temporal split and should not be generalized to prospective deployment without fresh validation.

## Citation

If you use this repository, cite the associated manuscript and link to the specific Git commit used for reproducibility. Publication metadata (journal, DOI and final citation) should be added only after it is confirmed.

**Code repository:** [selimmondol/Bangladesh-Dengue-Research](https://github.com/selimmondol/Bangladesh-Dengue-Research)

## Authors

Research collaboration associated with the above manuscript, including **Md Selim Mondol** (Heinrich Heine University Düsseldorf) and **Mst Sumaya Sultana** (Begum Rokeya University, Rangpur).

---

<div align="center"><sub>Research repository · Bangladesh dengue forecasting · 2000–2025</sub></div>
