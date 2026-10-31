#!/usr/bin/env python3
# Bangladesh Dengue Forecasting Research (2000-2025)
# Data/Bangladesh_Dengue_2000_2025_Data.xlsx | sheet: Research_Data

try:
    from IPython.display import display
except ImportError:
    def display(obj):
        print(obj)


# -----------------------------------------------------------------
# 01  Environment and dependencies
# -----------------------------------------------------------------

import sys
import pandas as pd
import numpy as np
import sklearn

print("Python:", sys.version)
print("Environment:", sys.executable)
print("Dengue Research Environment Ready!")


# -----------------------------------------------------------------
# 02  Output folder and configuration
# -----------------------------------------------------------------

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 130)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

print("Python notebook is ready.")
print("Pandas version:", pd.__version__)
print("Output folder:", OUTPUT_DIR.resolve())

from pathlib import Path

print("Current folder:", Path.cwd())

print("\nExcel files:")
for file in Path.cwd().rglob("*.xlsx"):
    print(file)

from pathlib import Path

print("Current working directory:", Path.cwd())

print("\n===== AVAILABLE EXCEL FILES =====")
for file in Path.cwd().rglob("*.xlsx"):
    print(file)


# -----------------------------------------------------------------
# 03  Load research dataset
# -----------------------------------------------------------------

import pandas as pd
from pathlib import Path

DATASET_NAME = "Bangladesh_Dengue_2000_2025_Data.xlsx"
CANDIDATE_FILES = [Path.cwd() / "Data" / DATASET_NAME, Path.cwd() / DATASET_NAME]
FILE = next((path for path in CANDIDATE_FILES if path.is_file()), CANDIDATE_FILES[0])

print("File exists:", FILE.exists())
print("File location:", FILE)

if not FILE.exists():
    print("\nFiles in current folder:")
    for f in Path.cwd().iterdir():
        print(f.name)
    raise FileNotFoundError("Excel dataset not found!")

print("\nAvailable sheets:", pd.ExcelFile(FILE).sheet_names)

df_updated = pd.read_excel(
    FILE,
    sheet_name="Research_Data"
)

print("\n===== UPDATED DATASET =====")
print("Observations:", len(df_updated))
print("Variables:", len(df_updated.columns))
print("Total cases:", df_updated["Cases"].sum())
print("Total deaths:", df_updated["Deaths"].sum())

print("\n===== ANNUAL TOTALS =====")
display(
    df_updated.groupby("Year")[["Cases", "Deaths"]]
    .sum()
    .tail(7)
)

df = df_updated.copy()

df["date"] = pd.to_datetime({
    "year": df["Year"],
    "month": df["Month_Number"],
    "day": 1
})

df = df.sort_values("date").reset_index(drop=True)

assert len(df) == 312
assert df["Cases"].sum() == 769500

print("Updated dataset is now active.")
print("Cases:", df["Cases"].sum())
print("Deaths:", df["Deaths"].sum())

print("Rows:", len(df))
print("Total Cases:", df["Cases"].sum())
print("Total Deaths:", df["Deaths"].sum())


# -----------------------------------------------------------------
# 04  Data quality and validation
# -----------------------------------------------------------------

import pandas as pd

print("========== DATASET INFORMATION ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE RECORDS ==========")
duplicates = df.duplicated(
    subset=["Year", "Month_Number"]
).sum()
print("Duplicate months:", duplicates)

print("\n========== MISSING MONTHS ==========")

dates = pd.to_datetime(
    dict(year=df["Year"], month=df["Month_Number"], day=1)
)

expected = pd.date_range(
    "2000-01-01", "2025-12-01", freq="MS"
)

missing_months = expected.difference(dates)

print("Missing months:", len(missing_months))
print(missing_months)

print("\n========== INVALID VALUES ==========")

cases = pd.to_numeric(df["Cases"], errors="coerce")
deaths = pd.to_numeric(df["Deaths"], errors="coerce")

print("Negative cases:", (cases < 0).sum())
print("Negative deaths:", (deaths < 0).sum())
print("Non-numeric cases:", cases.isna().sum())
print("Non-numeric deaths:", deaths.isna().sum())

print("\n========== RECORD STATUS ==========")
if "Dengue_Record_Status" in df.columns:
    print(df["Dengue_Record_Status"].value_counts(dropna=False))
else:
    print("Status flags are not part of the public 16-column Research_Data sheet; see README.")

print("\n========== BASIC STATISTICS ==========")
display(df[["Cases", "Deaths"]].describe())

import pandas as pd

dates = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month_Number"],
        day=1
    )
)

expected = pd.date_range(
    "2000-01-01",
    "2025-12-01",
    freq="MS"
)

print("===== FINAL DATA QUALITY REPORT =====")

print("Total observations:", len(df))
print("Total variables:", len(df.columns))
print("Missing values:", df.isna().sum().sum())

print("Duplicate months:",
      df.duplicated(["Year", "Month_Number"]).sum())

print("Missing months:",
      len(expected.difference(dates)))

print("Negative dengue cases:",
      (df["Cases"] < 0).sum())

print("Negative dengue deaths:",
      (df["Deaths"] < 0).sum())

print("\n===== SURVEILLANCE RECORD STATUS =====")

if "Dengue_Record_Status" in df.columns:
    status = df["Dengue_Record_Status"].astype(str)
    print("Records mentioning provisional:",
          status.str.contains("provisional", case=False).sum())
    print("Records mentioning derived:",
          status.str.contains("derived", case=False).sum())
else:
    print("No per-month source flags in the public analysis dataset.")
    print("Document 2025 provisional and derived observations in README/Methods.")

print("\n===== TOTAL DENGUE RECORDS =====")
print("Total cases:", df["Cases"].sum())
print("Total deaths:", df["Deaths"].sum())


# -----------------------------------------------------------------
# 05  Datetime variables and checks
# -----------------------------------------------------------------

expected = [
    "Year", "Month_Number", "Cases", "Deaths", "T2M_C", "T2M_MIN_C",
    "T2M_MAX_C", "RH2M_pct", "PRECTOTCORR_mm_day",
    "PRECTOTCORR_SUM_mm_month", "PS_kPa", "WS50M_MAX_m_s",
    "QV2M_g_kg", "WS2M_m_s"
]
missing_columns = [c for c in expected if c not in df.columns]
if missing_columns:
    raise ValueError(f"Required column(s) missing: {missing_columns}")

for col in expected:
    df[col] = pd.to_numeric(df[col], errors="coerce")

if df[["Year", "Month_Number"]].isna().any().any():
    raise ValueError("Year or Month_Number has missing/non-numeric values")

df["date"] = pd.to_datetime(
    {"year": df["Year"].astype(int),
     "month": df["Month_Number"].astype(int),
     "day": 1}
)
df = df.sort_values("date").reset_index(drop=True)

print("From:", df["date"].min().strftime("%Y-%m"))
print("To:  ", df["date"].max().strftime("%Y-%m"))
display(df[["date", "Cases", "Deaths", "T2M_C", "RH2M_pct"]].head())

print("Missing values in research variables:")
print(df[expected].isna().sum())

print("\nTotal observations:", len(df))

print("Date range:")
print(df["date"].min(), "to", df["date"].max())

print("\nDuplicate dates:")
print(df["date"].duplicated().sum())

numeric_cols = [c for c in expected if c not in ["Year", "Month_Number"]]

missing_by_column = df[numeric_cols].isna().sum().sort_values(ascending=False)
negative_cases = int((df["Cases"] < 0).sum())
negative_deaths = int((df["Deaths"] < 0).sum())
duplicate_months = int(df["date"].duplicated().sum())
expected_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="MS")
missing_months = expected_dates.difference(pd.DatetimeIndex(df["date"]))
sentinel_counts = (df[numeric_cols] == -999).sum()

print("Number of records:", len(df))
print("Number of unique months:", df["date"].nunique())
print("Duplicate months:", duplicate_months)
print("Missing months:", len(missing_months))
print("Negative dengue case counts:", negative_cases)
print("Negative death counts:", negative_deaths)
print("Missing numeric values by column:")
display(missing_by_column.to_frame("missing"))
print("-999 sentinel count:", int(sentinel_counts.sum()))

assert duplicate_months == 0, "Duplicate months found; fix before analysis."
assert len(missing_months) == 0, "Missing months found; fix before analysis."
assert negative_cases == 0 and negative_deaths == 0, "Negative counts found."
assert (sentinel_counts == 0).all(), "-999 sentinel values found; replace them with NaN and investigate."
assert missing_by_column.eq(0).all(), "Some numeric fields missing; investigate before PCA."

print("Structural numeric-data checks passed (NOT source verification).")

import pandas as pd

print("===== DATE RANGE VALIDATION =====")

expected_dates = pd.date_range(
    start="2000-01-01",
    end="2025-12-01",
    freq="MS"
)

actual_dates = pd.DatetimeIndex(df["date"])

missing_dates = expected_dates.difference(actual_dates)
extra_dates = actual_dates.difference(expected_dates)

print("Expected months:", len(expected_dates))
print("Actual months:", len(df))
print("Missing months:", len(missing_dates))
print("Unexpected months:", len(extra_dates))

print("\n===== CLIMATE VALIDATION =====")

checks = {
    "Invalid humidity": (
        (df["RH2M_pct"] < 0) |
        (df["RH2M_pct"] > 100)
    ).sum(),

    "Negative precipitation": (
        (df["PRECTOTCORR_mm_day"] < 0) |
        (df["PRECTOTCORR_SUM_mm_month"] < 0)
    ).sum(),

    "Temperature order violations": (
        (df["T2M_MIN_C"] > df["T2M_C"]) |
        (df["T2M_C"] > df["T2M_MAX_C"])
    ).sum()
}

for name, count in checks.items():
    print(f"{name}: {count}")

print("\n===== RECORD VERIFICATION =====")

if "Dengue_Record_Status" in df.columns:
    status = df["Dengue_Record_Status"].astype(str)
    review = df[
        status.str.contains("provisional|derived", case=False, regex=True)
    ]
    print("Records requiring verification:", len(review))
    display(review[["Year", "Month", "Cases", "Deaths", "Dengue_Record_Status"]])
else:
    print("Record-status annotations omitted from the public analysis sheet.")
    print("See the README/Methods for source and provisional-data limitations.")


# -----------------------------------------------------------------
# 06  Descriptive statistics
# -----------------------------------------------------------------

display(
    df[
        ["Cases", "Deaths", "T2M_C",
         "RH2M_pct", "PRECTOTCORR_SUM_mm_month"]
    ].describe().T
)

print("Months with zero cases:",
      int((df["Cases"] == 0).sum()))

print("Months with zero deaths:",
      int((df["Deaths"] == 0).sum()))

print("Total cases in workbook:",
      f"{df['Cases'].sum():,.0f}")

print("Total deaths in workbook:",
      f"{df['Deaths'].sum():,.0f}")

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------
# 07  Annual trends
# -----------------------------------------------------------------

annual = df.groupby("Year", as_index=False).agg(
    Cases=("Cases", "sum"),
    Deaths=("Deaths", "sum")
)

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

axes[0].plot(annual["Year"], annual["Cases"], marker="o")
axes[0].set_ylabel("Annual reported cases")
axes[0].set_title("Annual reported dengue cases in Bangladesh, 2000–2025")
axes[0].grid(alpha=0.2)

axes[1].plot(annual["Year"], np.log1p(annual["Cases"]), marker="o")
axes[1].set_ylabel("log(1 + annual cases)")
axes[1].set_xlabel("Year")
axes[1].grid(alpha=0.2)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_01_annual_dengue_trend.png", dpi=180, bbox_inches="tight")
plt.show()

display(annual.sort_values("Cases", ascending=False).head(8))


# -----------------------------------------------------------------
# 08  Monthly seasonality
# -----------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt

month_names = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

monthly = (
    df.groupby("Month_Number", as_index=False)
    .agg(
        Mean_Cases=("Cases", "mean"),
        Median_Cases=("Cases", "median"),
        Mean_Deaths=("Deaths", "mean"),
        Median_Deaths=("Deaths", "median")
    )
    .sort_values("Month_Number")
)

monthly["Month"] = month_names

print("MONTHLY DENGUE SEASONALITY SUMMARY")
display(monthly.round(2))

peak_mean = monthly.loc[
    monthly["Mean_Cases"].idxmax()
]

peak_median = monthly.loc[
    monthly["Median_Cases"].idxmax()
]

print("\nHighest average cases:",
      peak_mean["Month"],
      round(peak_mean["Mean_Cases"], 2))

print("Highest median cases:",
      peak_median["Month"],
      round(peak_median["Median_Cases"], 2))

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    monthly["Month"],
    monthly["Mean_Cases"],
    marker="o",
    linewidth=2,
    label="Mean cases"
)

ax.plot(
    monthly["Month"],
    monthly["Median_Cases"],
    marker="s",
    linewidth=2,
    label="Median cases"
)

ax.set_title(
    "Monthly Dengue Seasonality in Bangladesh (2000–2025)"
)

ax.set_xlabel("Month")
ax.set_ylabel("Reported Dengue Cases")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_02_monthly_seasonality.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------------------------------------------
# 09  Climate and dengue seasonality
# -----------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

climate_monthly = (
    df.groupby("Month_Number", as_index=False)
    .agg(
        Mean_Cases=("Cases", "mean"),
        Median_Cases=("Cases", "median"),
        Mean_Temperature=("T2M_C", "mean"),
        Mean_Humidity=("RH2M_pct", "mean"),
        Mean_Rainfall=("PRECTOTCORR_SUM_mm_month", "mean")
    )
    .sort_values("Month_Number")
)

month_names = [
    "Jan", "Feb", "Mar", "Apr",
    "May", "Jun", "Jul", "Aug",
    "Sep", "Oct", "Nov", "Dec"
]

climate_monthly["Month"] = month_names

print("CLIMATE AND DENGUE MONTHLY SUMMARY")
display(climate_monthly.round(2))

fig, axes = plt.subplots(
    4, 1, figsize=(12, 14), sharex=True
)

axes[0].plot(
    climate_monthly["Month"],
    climate_monthly["Median_Cases"],
    marker="o"
)
axes[0].set_ylabel("Median Cases")
axes[0].set_title("Dengue Seasonality")
axes[0].grid(alpha=0.3)

axes[1].plot(
    climate_monthly["Month"],
    climate_monthly["Mean_Temperature"],
    marker="o"
)
axes[1].set_ylabel("Temperature (°C)")
axes[1].set_title("Average Monthly Temperature")
axes[1].grid(alpha=0.3)

axes[2].plot(
    climate_monthly["Month"],
    climate_monthly["Mean_Humidity"],
    marker="o"
)
axes[2].set_ylabel("Humidity (%)")
axes[2].set_title("Average Monthly Relative Humidity")
axes[2].grid(alpha=0.3)

axes[3].plot(
    climate_monthly["Month"],
    climate_monthly["Mean_Rainfall"],
    marker="o"
)
axes[3].set_ylabel("Rainfall (mm/month)")
axes[3].set_xlabel("Month")
axes[3].set_title("Average Monthly Precipitation")
axes[3].grid(alpha=0.3)

plt.suptitle(
    "Dengue and Climate Seasonality in Bangladesh (2000–2025)",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_03_climate_dengue_seasonality.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------------------------------------------
# 10  Climate-lag correlation
# -----------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

data = df.sort_values("date").copy().reset_index(drop=True)

climate_vars = {
    "T2M_C": "Temperature",
    "RH2M_pct": "Humidity",
    "PRECTOTCORR_SUM_mm_month": "Rainfall"
}

data["Log_Cases"] = np.log1p(data["Cases"])

data["Case_Anomaly"] = (
    data["Log_Cases"]
    - data.groupby("Month_Number")["Log_Cases"].transform("mean")
)

results = []

for col, name in climate_vars.items():

    anomaly = (
        data[col]
        - data.groupby("Month_Number")[col].transform("mean")
    )

    for lag in range(4):

        raw_lagged = data[col].shift(lag)
        anomaly_lagged = anomaly.shift(lag)

        raw_corr = data["Cases"].corr(
            raw_lagged, method="spearman"
        )

        adjusted_corr = data["Case_Anomaly"].corr(
            anomaly_lagged, method="spearman"
        )

        results.append({
            "Variable": name,
            "Lag_Months": lag,
            "Raw_Spearman": raw_corr,
            "Season_Adjusted_Spearman": adjusted_corr
        })

lag_results = pd.DataFrame(results)

print("CLIMATE-DENGUE LAG CORRELATION")
display(lag_results.round(3))

fig, ax = plt.subplots(figsize=(10, 5))

for name in climate_vars.values():

    subset = lag_results[
        lag_results["Variable"] == name
    ]

    ax.plot(
        subset["Lag_Months"],
        subset["Season_Adjusted_Spearman"],
        marker="o",
        linewidth=2,
        label=name
    )

ax.axhline(0, linewidth=1, color="gray")

ax.set_title(
    "Season-Adjusted Climate-Dengue Lag Correlations"
)
ax.set_xlabel("Climate lag (months)")
ax.set_ylabel("Spearman correlation")
ax.set_xticks([0, 1, 2, 3])
ax.set_ylim(-1, 1)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_04_lag_correlations.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------------------------------------------
# 11  Monthly time series
# -----------------------------------------------------------------

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
axes[0].plot(df["date"], df["Cases"])
axes[0].set_ylabel("Cases per month")
axes[0].set_title("Monthly Dengue Cases in Bangladesh, 2000–2025")
axes[0].grid(alpha=0.2)
axes[1].plot(df["date"], np.log1p(df["Cases"]))
axes[1].set_ylabel("log(1 + cases)")
axes[1].set_xlabel("Month")
axes[1].grid(alpha=0.2)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figure_03_monthly_cases.png", dpi=180, bbox_inches="tight")
plt.show()


# -----------------------------------------------------------------
# 12  PCA: explained variance
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

climate_features = [
    "T2M_C",
    "T2M_MIN_C",
    "T2M_MAX_C",
    "RH2M_pct",
    "PRECTOTCORR_SUM_mm_month",
    "PS_kPa",
    "WS50M_MAX_m_s",
    "QV2M_g_kg",
    "WS2M_m_s"
]

data = df.sort_values("date").copy()

train = data[data["date"] < "2023-01-01"].copy()

X_train = train[climate_features]

assert X_train.notna().all().all(), "Missing values detected"

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)

pca = PCA()
pca.fit(X_scaled)

explained = pca.explained_variance_ratio_
cumulative = np.cumsum(explained)

pca_summary = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(explained))],
    "Explained_Variance_%": explained * 100,
    "Cumulative_Variance_%": cumulative * 100
})

print("PCA EXPLAINED VARIANCE")
display(pca_summary.round(3))

n_components = np.searchsorted(cumulative, 0.90) + 1

print("Components needed for 90% variance:", n_components)

fig, ax = plt.subplots(figsize=(10, 6))

ax.bar(
    range(1, 10),
    explained * 100,
    label="Individual variance"
)

ax.plot(
    range(1, 10),
    cumulative * 100,
    marker="o",
    label="Cumulative variance"
)

ax.axhline(
    90,
    linestyle="--",
    label="90% threshold"
)

ax.set_title("PCA Explained Variance: Climate Variables")
ax.set_xlabel("Principal Component")
ax.set_ylabel("Explained Variance (%)")
ax.set_xticks(range(1, 10))
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "figure_04_pca_variance.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# -----------------------------------------------------------------
# 13  PCA: feature loadings
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

loadings = pd.DataFrame(
    pca.components_[:3].T *
    np.sqrt(pca.explained_variance_[:3]),
    index=climate_features,
    columns=["PC1", "PC2", "PC3"]
)

print("===== PCA LOADINGS =====")
display(loadings.round(3))

print("\n===== TOP VARIABLES PER COMPONENT =====")

for pc in ["PC1", "PC2", "PC3"]:

    top_variables = (
        loadings[pc]
        .abs()
        .sort_values(ascending=False)
        .head(5)
    )

    print(f"\nTop 5 variables for {pc}:")
    display(
        loadings.loc[
            top_variables.index, [pc]
        ].round(3)
    )

fig, ax = plt.subplots(figsize=(9, 6))

im = ax.imshow(
    loadings.values,
    cmap="RdBu_r",
    vmin=-1,
    vmax=1,
    aspect="auto"
)

ax.set_xticks(range(3))
ax.set_xticklabels(loadings.columns)

ax.set_yticks(range(len(climate_features)))
ax.set_yticklabels(climate_features)

ax.set_title(
    "PCA Loadings of Climate Variables"
)

for i in range(len(climate_features)):
    for j in range(3):
        ax.text(
            j, i,
            f"{loadings.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=9
        )

fig.colorbar(
    im,
    ax=ax,
    label="PCA Loading"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_05_pca_loadings.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

loadings.to_csv(
    OUTPUT_DIR / "pca_loadings.csv"
)

print("PCA Loadings Analysis Completed!")


# -----------------------------------------------------------------
# 14  PCA: component scores
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

pca_data = df.sort_values("date").reset_index(drop=True).copy()

X_all = pca_data[climate_features]

assert X_all.notna().all().all(), "Missing climate data found"

X_all_scaled = scaler.transform(X_all)

scores = pca.transform(X_all_scaled)

for i in range(3):
    pca_data[f"PC{i+1}"] = scores[:, i]

pca_data["Period"] = np.where(
    pca_data["date"] < "2023-01-01",
    "Training",
    "Test"
)

print("===== PCA SCORES =====")

display(
    pca_data[
        ["date", "Cases", "PC1", "PC2", "PC3", "Period"]
    ].head(10).round(3)
)

print("\n===== DATA SPLIT =====")
print(pca_data["Period"].value_counts())

fig, ax = plt.subplots(figsize=(10, 7))

scatter = ax.scatter(
    pca_data["PC1"],
    pca_data["PC2"],
    c=np.log1p(pca_data["Cases"]),
    cmap="viridis",
    alpha=0.75,
    s=35
)

ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

ax.set_title(
    "PCA Climate Scores and Dengue Cases (2000–2025)"
)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

fig.colorbar(
    scatter,
    ax=ax,
    label="log(1 + dengue cases)"
)

ax.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_06_pca_scores.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

pca_data[
    ["date", "Cases", "Deaths", "PC1", "PC2", "PC3", "Period"]
].to_csv(
    OUTPUT_DIR / "pca_scores_2000_2025.csv",
    index=False
)

print("\nPCA Scores Analysis Completed!")


# -----------------------------------------------------------------
# 15  Historical forecasting baselines
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

forecast_data = (
    df.sort_values("date")
    .reset_index(drop=True)
    .copy()
)

forecast_data["Previous_Month"] = (
    forecast_data["Cases"].shift(1)
)

forecast_data["Seasonal_Naive"] = (
    forecast_data["Cases"].shift(12)
)

test = forecast_data[
    forecast_data["date"] >= "2023-01-01"
].copy()

results = []

for model in ["Previous_Month", "Seasonal_Naive"]:

    y_true = test["Cases"]
    y_pred = test[model]

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    log_mae = mean_absolute_error(
        np.log1p(y_true),
        np.log1p(y_pred)
    )

    results.append({
        "Model": model,
        "MAE": mae,
        "RMSE": rmse,
        "Log_MAE": log_mae
    })

baseline_results = pd.DataFrame(results)

print("===== BASELINE FORECAST RESULTS =====")
display(baseline_results.round(3))

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    test["date"],
    test["Cases"],
    label="Actual cases",
    linewidth=2
)

ax.plot(
    test["date"],
    test["Previous_Month"],
    label="Previous-month forecast",
    linestyle="--"
)

ax.plot(
    test["date"],
    test["Seasonal_Naive"],
    label="Seasonal naive forecast",
    linestyle=":"
)

ax.set_title(
    "Dengue Forecasting Baselines (2023–2025)"
)

ax.set_xlabel("Date")
ax.set_ylabel("Monthly dengue cases")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_07_baseline_forecasting.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

baseline_results.to_csv(
    OUTPUT_DIR / "baseline_results.csv",
    index=False
)

print("Baseline forecasting completed!")


# -----------------------------------------------------------------
# 16  Random Forest: validation and PCA comparison
# -----------------------------------------------------------------

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

ml_data = df.sort_values("date").copy().reset_index(drop=True)

ml_data["Cases_lag1"] = ml_data["Cases"].shift(1)
ml_data["Cases_lag12"] = ml_data["Cases"].shift(12)

climate_lag_cols = []

for col in climate_features:
    new_col = col + "_lag1"
    ml_data[new_col] = ml_data[col].shift(1)
    climate_lag_cols.append(new_col)

ml_data["month_sin"] = np.sin(
    2 * np.pi * ml_data["Month_Number"] / 12
)

ml_data["month_cos"] = np.cos(
    2 * np.pi * ml_data["Month_Number"] / 12
)

other_features = [
    "Cases_lag1",
    "Cases_lag12",
    "month_sin",
    "month_cos"
]

feature_cols = climate_lag_cols + other_features

ml_data = ml_data.dropna(
    subset=feature_cols + ["Cases"]
).reset_index(drop=True)

train = ml_data[
    ml_data["date"] < "2020-01-01"
].copy()

val = ml_data[
    (ml_data["date"] >= "2020-01-01") &
    (ml_data["date"] < "2023-01-01")
].copy()

test = ml_data[
    ml_data["date"] >= "2023-01-01"
].copy()

print("Training observations:", len(train))
print("Validation observations:", len(val))
print("Test observations:", len(test))

X_train = train[feature_cols]
y_train = train["Cases"]

X_val = val[feature_cols]
y_val = val["Cases"]

rf_original = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_original.fit(X_train, y_train)

pred_original = rf_original.predict(X_val)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "climate_pca",
            Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=3))
            ]),
            climate_lag_cols
        ),
        (
            "other_features",
            "passthrough",
            other_features
        )
    ]
)

rf_pca = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ))
])

rf_pca.fit(X_train, y_train)

pred_pca = rf_pca.predict(X_val)

def evaluate_model(name, actual, predicted):
    predicted = np.maximum(np.asarray(predicted), 0)

    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "Log_MAE": mean_absolute_error(
            np.log1p(actual),
            np.log1p(predicted)
        )
    }

comparison = pd.DataFrame([
    evaluate_model(
        "Previous Month",
        y_val,
        val["Cases_lag1"]
    ),
    evaluate_model(
        "Random Forest",
        y_val,
        pred_original
    ),
    evaluate_model(
        "Random Forest + PCA",
        y_val,
        pred_pca
    )
])

print("\n===== VALIDATION RESULTS (2020-2022) =====")
display(comparison.round(3))

print("\nRandom Forest validation completed!")


# -----------------------------------------------------------------
# 17  Random Forest plots
# -----------------------------------------------------------------

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

metrics = ["MAE", "RMSE", "Log_MAE"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for ax, metric in zip(axes, metrics):

    bars = ax.bar(
        comparison["Model"],
        comparison[metric]
    )

    ax.set_title(f"{metric} Comparison")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

fig.suptitle(
    "Random Forest Validation Performance (2020–2022)",
    fontsize=15
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_08_RF_model_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    val["date"],
    y_val,
    label="Actual Cases",
    linewidth=2.5
)

ax.plot(
    val["date"],
    pred_original,
    label="Random Forest",
    linestyle="--",
    linewidth=2
)

ax.plot(
    val["date"],
    pred_pca,
    label="Random Forest + PCA",
    linestyle=":",
    linewidth=2
)

ax.plot(
    val["date"],
    val["Cases_lag1"],
    label="Previous Month Baseline",
    alpha=0.7
)

ax.set_title(
    "Actual vs Predicted Monthly Dengue Cases (2020–2022)"
)

ax.set_xlabel("Year")
ax.set_ylabel("Monthly Dengue Cases")

ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "figure_09_RF_actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# -----------------------------------------------------------------
# 18  XGBoost: validation and PCA comparison
# -----------------------------------------------------------------

import numpy as np
import pandas as pd

from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone

xgb_original = XGBRegressor(
    n_estimators=300,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    reg_lambda=5,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=4
)

xgb_original.fit(X_train, y_train)

pred_xgb = np.maximum(
    xgb_original.predict(X_val), 0
)

xgb_pca = Pipeline([
    (
        "preprocessing",
        clone(preprocessor)
    ),
    (
        "model",
        XGBRegressor(
            n_estimators=300,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=3,
            reg_lambda=5,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
            n_jobs=4
        )
    )
])

xgb_pca.fit(X_train, y_train)

pred_xgb_pca = np.maximum(
    xgb_pca.predict(X_val), 0
)

all_results = pd.DataFrame([
    evaluate_model(
        "Previous Month",
        y_val,
        val["Cases_lag1"]
    ),
    evaluate_model(
        "Random Forest",
        y_val,
        pred_original
    ),
    evaluate_model(
        "Random Forest + PCA",
        y_val,
        pred_pca
    ),
    evaluate_model(
        "XGBoost",
        y_val,
        pred_xgb
    ),
    evaluate_model(
        "XGBoost + PCA",
        y_val,
        pred_xgb_pca
    )
])

all_results = (
    all_results
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("===== ML VALIDATION COMPARISON =====")

display(all_results.round(3))

from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

all_results.to_csv(
    OUTPUT_DIR / "ml_validation_comparison.csv",
    index=False
)

print("\nXGBoost comparison completed!")


# -----------------------------------------------------------------
# 19  Validation visualizations
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300
})

plot_df = (
    all_results
    .sort_values("MAE", ascending=True)
    .reset_index(drop=True)
)

best_model = plot_df.iloc[0]["Model"]

colors = [
    "#16867D" if model == best_model else "#9BAABD"
    for model in plot_df["Model"]
]

fig, ax = plt.subplots(figsize=(10, 5.7))

bars = ax.barh(
    plot_df["Model"],
    plot_df["MAE"],
    color=colors,
    height=0.57,
    edgecolor="none"
)

ax.invert_yaxis()

for bar, value in zip(bars, plot_df["MAE"]):
    ax.text(
        value + 25,
        bar.get_y() + bar.get_height() / 2,
        f"{value:,.0f}",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="#263238"
    )

ax.set_title(
    "Machine Learning Model Performance",
    loc="left",
    fontweight="bold",
    pad=20
)

ax.text(
    0, 1.015,
    "One-month-ahead dengue forecasting | Validation: 2020–2022",
    transform=ax.transAxes,
    fontsize=10,
    color="#596579"
)

ax.set_xlabel("Mean Absolute Error (reported cases)")
ax.set_xlim(0, plot_df["MAE"].max() * 1.23)

ax.xaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

ax.grid(
    axis="x",
    linestyle="--",
    alpha=0.20
)

ax.set_axisbelow(True)

fig.text(
    0.12, 0.015,
    "Lower MAE indicates better forecasting performance. "
    "Results are validation estimates, not final test results.",
    fontsize=9,
    color="#697586"
)

plt.tight_layout(rect=[0, 0.05, 1, 1])

fig.savefig(
    OUTPUT_DIR / "Figure_10_ML_Performance.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_10_ML_Performance.pdf",
    bbox_inches="tight"
)

plt.show()

import matplotlib.dates as mdates

dates = pd.to_datetime(val["date"])

actual = np.asarray(y_val)
rf_pred = np.maximum(np.asarray(pred_original), 0)
xgb_pred = np.maximum(np.asarray(pred_xgb), 0)
naive_pred = np.asarray(val["Cases_lag1"])

fig, ax = plt.subplots(figsize=(13, 6))

ax.plot(
    dates,
    actual,
    color="#172D49",
    linewidth=2.8,
    marker="o",
    markersize=4.5,
    label="Observed cases",
    zorder=5
)

ax.plot(
    dates,
    rf_pred,
    color="#118A80",
    linewidth=2.2,
    linestyle="--",
    label="Random Forest"
)

ax.plot(
    dates,
    xgb_pred,
    color="#D68C36",
    linewidth=1.9,
    linestyle="-.",
    label="XGBoost"
)

ax.plot(
    dates,
    naive_pred,
    color="#9399A5",
    linewidth=1.6,
    linestyle=":",
    label="Previous-month baseline",
    alpha=0.85
)

ax.set_title(
    "Observed vs Predicted Monthly Dengue Cases",
    loc="left",
    fontweight="bold",
    pad=22
)

ax.text(
    0, 1.018,
    "Bangladesh | One-month-ahead forecasts | 2020–2022 validation",
    transform=ax.transAxes,
    fontsize=10,
    color="#596579"
)

ax.set_xlabel("Month")
ax.set_ylabel("Reported dengue cases")

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=3)
)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

ax.yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

ax.set_ylim(bottom=0)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.22
)

ax.set_axisbelow(True)

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper left",
    fontsize=10
)

plt.xticks(rotation=35, ha="right")

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_11_Observed_vs_Predicted.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_11_Observed_vs_Predicted.pdf",
    bbox_inches="tight"
)

plt.show()


# -----------------------------------------------------------------
# 20  Validation errors by year
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

validation_predictions = pd.DataFrame({
    "Date": pd.to_datetime(val["date"]).to_numpy(),
    "Actual": np.asarray(y_val),
    "Previous Month": np.asarray(val["Cases_lag1"]),
    "Random Forest": np.maximum(pred_original, 0),
    "Random Forest + PCA": np.maximum(pred_pca, 0),
    "XGBoost": np.maximum(pred_xgb, 0),
    "XGBoost + PCA": np.maximum(pred_xgb_pca, 0)
})

validation_predictions["Year"] = (
    validation_predictions["Date"].dt.year
)

models = [
    "Previous Month",
    "Random Forest",
    "Random Forest + PCA",
    "XGBoost",
    "XGBoost + PCA"
]

yearly_results = []

for year, group in validation_predictions.groupby("Year"):

    for model in models:

        actual = group["Actual"].to_numpy()
        predicted = group[model].to_numpy()

        yearly_results.append({
            "Year": year,
            "Model": model,
            "MAE": mean_absolute_error(
                actual, predicted
            ),
            "Bias": np.mean(predicted - actual)
        })

yearly_results = pd.DataFrame(yearly_results)

print("===== YEAR-WISE MODEL PERFORMANCE =====")
display(yearly_results.round(2))

mae_table = yearly_results.pivot(
    index="Year",
    columns="Model",
    values="MAE"
)

fig, ax = plt.subplots(figsize=(12, 6))

mae_table[models].plot(
    kind="bar",
    ax=ax,
    width=0.75
)

ax.set_title(
    "Year-wise Forecasting Errors (2020–2022)",
    fontweight="bold",
    fontsize=15
)

ax.set_xlabel("Validation Year")
ax.set_ylabel("Mean Absolute Error (Cases)")
ax.tick_params(axis="x", rotation=0)

ax.legend(
    title="Forecasting Model",
    frameon=False,
    fontsize=9
)

ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Figure_12_Yearwise_Model_Errors.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

yearly_results.to_csv(
    OUTPUT_DIR / "yearwise_model_errors.csv",
    index=False
)

print("Year-wise error analysis completed!")


# -----------------------------------------------------------------
# 21  Independent test: 2023-2025
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.ticker import StrMethodFormatter

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

trainval = ml_data[
    ml_data["date"] < "2023-01-01"
].copy()

final_test = ml_data[
    ml_data["date"] >= "2023-01-01"
].copy()

X_trainval = trainval[feature_cols]
y_trainval = trainval["Cases"]

X_test = final_test[feature_cols]
y_test = final_test["Cases"]

print("Training observations:", len(trainval))
print("Test observations:", len(final_test))

final_models = {
    "Random Forest": clone(rf_original),
    "Random Forest + PCA": clone(rf_pca),
    "XGBoost": clone(xgb_original),
    "XGBoost + PCA": clone(xgb_pca)
}

predictions = {
    "Previous Month": final_test["Cases_lag1"].to_numpy()
}

for name, model in final_models.items():
    model.fit(X_trainval, y_trainval)

    predictions[name] = np.maximum(
        model.predict(X_test), 0
    )

test_results = []

for name, pred in predictions.items():

    test_results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
        "Log_MAE": mean_absolute_error(
            np.log1p(y_test),
            np.log1p(pred)
        ),
        "Bias": np.mean(pred - y_test.to_numpy())
    })

test_results = (
    pd.DataFrame(test_results)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("\n===== FINAL TEST RESULTS (2023-2025) =====")
display(test_results.round(3))

test_results.to_csv(
    OUTPUT_DIR / "final_test_results.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(13, 6))

dates = final_test["date"]

ax.plot(
    dates, y_test,
    label="Observed cases",
    color="#17324D",
    linewidth=2.7,
    marker="o",
    markersize=3.5
)

ax.plot(
    dates, predictions["Random Forest"],
    label="Random Forest",
    color="#16867D",
    linestyle="--",
    linewidth=2
)

ax.plot(
    dates, predictions["Random Forest + PCA"],
    label="Random Forest + PCA",
    color="#CE8640",
    linestyle="-.",
    linewidth=1.8
)

ax.plot(
    dates, predictions["Previous Month"],
    label="Previous-month baseline",
    color="#9AA4AF",
    linestyle=":",
    linewidth=1.8
)

ax.set_title(
    "Independent Test: Dengue Forecasts (2023–2025)",
    fontweight="bold",
    fontsize=15,
    loc="left"
)

ax.set_xlabel("Month")
ax.set_ylabel("Reported dengue cases")

ax.xaxis.set_major_locator(
    mdates.MonthLocator(interval=3)
)
ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)
ax.yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

ax.set_ylim(bottom=0)
ax.grid(axis="y", linestyle="--", alpha=0.22)
ax.legend(frameon=False, ncol=2)

plt.xticks(rotation=35, ha="right")
plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_13_Final_Test.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_13_Final_Test.pdf",
    bbox_inches="tight"
)

plt.show()

print("\nFinal test evaluation completed!")


# -----------------------------------------------------------------
# 22  Outbreak-year diagnostics
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

test_predictions = pd.DataFrame({
    "Date": pd.to_datetime(final_test["date"]).to_numpy(),
    "Actual": y_test.to_numpy()
})

models = [
    "Previous Month",
    "Random Forest",
    "Random Forest + PCA",
    "XGBoost",
    "XGBoost + PCA"
]

for model in models:
    test_predictions[model] = np.asarray(predictions[model])

test_predictions["Year"] = test_predictions["Date"].dt.year

yearly_test_results = []

for year, group in test_predictions.groupby("Year"):
    for model in models:

        actual = group["Actual"].to_numpy()
        predicted = group[model].to_numpy()

        yearly_test_results.append({
            "Year": year,
            "Model": model,
            "MAE": mean_absolute_error(actual, predicted),
            "Bias": np.mean(predicted - actual)
        })

yearly_test_results = pd.DataFrame(yearly_test_results)

print("===== YEAR-WISE FINAL TEST RESULTS =====")
display(yearly_test_results.round(2))

peak_idx = test_predictions["Actual"].idxmax()
peak = test_predictions.loc[peak_idx]

print("\n===== LARGEST MONTHLY OUTBREAK =====")
print("Date:", peak["Date"].strftime("%Y-%m"))
print("Observed cases:", int(peak["Actual"]))

peak_comparison = pd.DataFrame({
    "Model": models,
    "Predicted_Cases": [
        peak[model] for model in models
    ],
    "Error": [
        peak[model] - peak["Actual"] for model in models
    ]
})

display(peak_comparison.round(2))

yearly_test_results.to_csv(
    OUTPUT_DIR / "yearwise_final_test_results.csv",
    index=False
)

peak_comparison.to_csv(
    OUTPUT_DIR / "outbreak_peak_comparison.csv",
    index=False
)

outbreak_2023 = test_predictions[
    test_predictions["Year"] == 2023
]

fig, ax = plt.subplots(figsize=(11, 5.5))

styles = {
    "Actual": ("#17324D", "-", 2.8),
    "Previous Month": ("#9AA4AF", ":", 2),
    "Random Forest": ("#16867D", "--", 2),
    "Random Forest + PCA": ("#CE8640", "-.", 2)
}

for model, (color, style, width) in styles.items():
    ax.plot(
        outbreak_2023["Date"],
        outbreak_2023[model],
        color=color,
        linestyle=style,
        linewidth=width,
        marker="o" if model == "Actual" else None,
        label="Observed cases" if model == "Actual" else model
    )

ax.set_title(
    "Forecasting the 2023 Dengue Outbreak",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Month")
ax.set_ylabel("Reported dengue cases")
ax.set_ylim(bottom=0)
ax.grid(axis="y", alpha=0.2)
ax.legend(frameon=False)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_14_2023_Outbreak_Forecasts.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_14_2023_Outbreak_Forecasts.pdf",
    bbox_inches="tight"
)

plt.show()

print("Outbreak-year diagnostics completed!")

from pathlib import Path

print("Current folder:", Path.cwd())

for file in Path.cwd().glob("*.xlsx"):
    print("Excel file:", file.name)


# -----------------------------------------------------------------
# 23  Seasonal-naive baseline
# -----------------------------------------------------------------

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

filename = "Bangladesh_Dengue_2000_2025_Data.xlsx"
candidates = [Path.cwd() / "Data" / filename, Path.cwd() / filename]
file_path = next((p for p in candidates if p.is_file()), None)

if file_path is None:
    raise FileNotFoundError(
        f"Cannot find {filename}. Put it in Data/ or the current working folder."
    )
print("Dataset found:", file_path)

df = pd.read_excel(file_path, sheet_name="Research_Data")
df = df.sort_values(["Year", "Month_Number"]).reset_index(drop=True)

df["Seasonal_Naive"] = df["Cases"].shift(12)

test = df[df["Year"].between(2023, 2025)].copy()

actual = test["Cases"].to_numpy()
predicted = test["Seasonal_Naive"].to_numpy()

assert len(test) == 36, "Expected 36 test months."
assert not np.isnan(predicted).any(), "Missing predictions found."

mae = mean_absolute_error(actual, predicted)
rmse = np.sqrt(mean_squared_error(actual, predicted))
log_mae = np.mean(np.abs(np.log1p(actual) - np.log1p(predicted)))
bias = np.mean(predicted - actual)

print("\nSEASONAL NAIVE BASELINE — 2023–2025")
print("-----------------------------------")
print(f"MAE     : {mae:,.2f}")
print(f"RMSE    : {rmse:,.2f}")
print(f"Log MAE : {log_mae:.3f}")
print(f"Bias    : {bias:,.2f}")


# -----------------------------------------------------------------
# 24  Moving-block bootstrap
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

bootstrap_data = test_predictions.copy()

models = [
    "Random Forest",
    "Random Forest + PCA",
    "XGBoost",
    "XGBoost + PCA"
]

y_true = bootstrap_data["Actual"].to_numpy()
baseline = bootstrap_data["Previous Month"].to_numpy()

n = len(y_true)

B = 2000
block_length = 6
rng = np.random.default_rng(42)

bootstrap_indices = []

for _ in range(B):
    starts = rng.integers(
        0,
        n - block_length + 1,
        size=int(np.ceil(n / block_length))
    )

    indices = np.concatenate([
        np.arange(start, start + block_length)
        for start in starts
    ])[:n]

    bootstrap_indices.append(indices)

results = []

baseline_error = np.abs(y_true - baseline)

for model in models:

    model_pred = bootstrap_data[model].to_numpy()
    model_error = np.abs(y_true - model_pred)

    error_difference = model_error - baseline_error

    observed_difference = error_difference.mean()

    boot_differences = np.array([
        error_difference[idx].mean()
        for idx in bootstrap_indices
    ])

    ci_low, ci_high = np.percentile(
        boot_differences, [2.5, 97.5]
    )

    results.append({
        "Model": model,
        "MAE_Difference": observed_difference,
        "CI_Lower": ci_low,
        "CI_Upper": ci_high
    })

bootstrap_results = pd.DataFrame(results)

print("===== BLOCK BOOTSTRAP RESULTS =====")
display(bootstrap_results.round(2))

fig, ax = plt.subplots(figsize=(10, 5.5))

for i, row in bootstrap_results.iterrows():

    ax.plot(
        [row["CI_Lower"], row["CI_Upper"]],
        [i, i],
        linewidth=3
    )

    ax.scatter(
        row["MAE_Difference"], i,
        s=80, zorder=3
    )

ax.axvline(
    0,
    color="black",
    linestyle="--",
    linewidth=1.5
)

ax.set_yticks(range(len(bootstrap_results)))
ax.set_yticklabels(bootstrap_results["Model"])

ax.set_title(
    "Bootstrap Uncertainty in Forecasting Performance",
    fontweight="bold",
    fontsize=14
)

ax.set_xlabel(
    "MAE Difference vs Previous-Month Baseline (Cases)"
)

ax.grid(axis="x", alpha=0.2)
ax.invert_yaxis()

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Figure_15_Bootstrap_MAE_Comparison.png",
    dpi=400,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "Figure_15_Bootstrap_MAE_Comparison.pdf",
    bbox_inches="tight"
)

plt.show()

bootstrap_results.to_csv(
    OUTPUT_DIR / "bootstrap_mae_results.csv",
    index=False
)

print("Bootstrap analysis completed!")


# -----------------------------------------------------------------
# 25  Bootstrap block-length sensitivity
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

y_true = test_predictions["Actual"].to_numpy()
baseline = test_predictions["Previous Month"].to_numpy()

models = [
    "Random Forest",
    "Random Forest + PCA",
    "XGBoost",
    "XGBoost + PCA"
]

n = len(y_true)
B = 2000

block_lengths = [3, 6, 12]

baseline_error = np.abs(y_true - baseline)

sensitivity_results = []

for block_length in block_lengths:

    rng = np.random.default_rng(42)

    bootstrap_indices = []

    for _ in range(B):

        starts = rng.integers(
            0,
            n - block_length + 1,
            size=int(np.ceil(n / block_length))
        )

        indices = np.concatenate([
            np.arange(start, start + block_length)
            for start in starts
        ])[:n]

        bootstrap_indices.append(indices)

    for model in models:

        model_pred = test_predictions[model].to_numpy()
        model_error = np.abs(y_true - model_pred)

        difference = model_error - baseline_error

        observed = difference.mean()

        boot_values = np.array([
            difference[idx].mean()
            for idx in bootstrap_indices
        ])

        ci_low, ci_high = np.percentile(
            boot_values, [2.5, 97.5]
        )

        sensitivity_results.append({
            "Block_Length": block_length,
            "Model": model,
            "MAE_Difference": observed,
            "CI_Lower": ci_low,
            "CI_Upper": ci_high,
            "Includes_Zero": ci_low <= 0 <= ci_high
        })

sensitivity_df = pd.DataFrame(sensitivity_results)

print("===== BOOTSTRAP SENSITIVITY RESULTS =====")
display(sensitivity_df.round(2))

sensitivity_df.to_csv(
    OUTPUT_DIR / "bootstrap_sensitivity_results.csv",
    index=False
)

fig, axes = plt.subplots(
    1, 3,
    figsize=(16, 6),
    sharey=True
)

for ax, block_length in zip(axes, block_lengths):

    subset = sensitivity_df[
        sensitivity_df["Block_Length"] == block_length
    ].reset_index(drop=True)

    for i, row in subset.iterrows():

        ax.plot(
            [row["CI_Lower"], row["CI_Upper"]],
            [i, i],
            linewidth=2.5,
            marker="|",
            markersize=10
        )

        ax.scatter(
            row["MAE_Difference"], i,
            s=65,
            zorder=3
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1
    )

    ax.set_title(
        f"Block Length = {block_length} Months"
    )

    ax.set_xlabel("MAE Difference (Cases)")
    ax.grid(axis="x", alpha=0.2)
    ax.set_yticks(range(len(models)))

    if ax == axes[0]:
        ax.set_yticklabels(models)

    ax.invert_yaxis()

fig.suptitle(
    "Bootstrap Sensitivity to Block Length",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Figure_16_Bootstrap_Sensitivity.png",
    dpi=400,
    bbox_inches="tight"
)

plt.savefig(
    OUTPUT_DIR / "Figure_16_Bootstrap_Sensitivity.pdf",
    bbox_inches="tight"
)

plt.show()

print("Bootstrap sensitivity analysis completed!")


# -----------------------------------------------------------------
# 26  Figure 7: observed versus predicted cases
# -----------------------------------------------------------------

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
import xgboost

file_name = "Bangladesh_Dengue_2000_2025_Data.xlsx"

search_folders = [
    Path.cwd() / "Data",
    Path.cwd()
]

file_path = next(
    (
        folder / file_name
        for folder in search_folders
        if (folder / file_name).exists()
    ),
    None
)

if file_path is None:
    raise FileNotFoundError(
        f"Cannot find {file_name}.\n"
        "Put the Excel file inside Data/ or the current working folder.\n"
        f"Current working folder: {Path.cwd()}"
    )

df = pd.read_excel(
    file_path,
    sheet_name="Research_Data"
)

df["date"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month_Number"],
        day=1
    )
)

df = df.sort_values("date").reset_index(drop=True)

assert len(df) == 312, "Expected 312 monthly observations."
assert df["Cases"].sum() == 769500, "Dataset total differs."

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)
print("XGBoost version:", xgboost.__version__)

climate_features = [
    "T2M_C",
    "T2M_MIN_C",
    "T2M_MAX_C",
    "RH2M_pct",
    "PRECTOTCORR_SUM_mm_month",
    "PS_kPa",
    "WS50M_MAX_m_s",
    "QV2M_g_kg",
    "WS2M_m_s"
]

data = df.copy()

data["Cases_lag1"] = data["Cases"].shift(1)
data["Cases_lag12"] = data["Cases"].shift(12)

climate_lags = []

for feature in climate_features:
    new_col = feature + "_lag1"
    data[new_col] = data[feature].shift(1)
    climate_lags.append(new_col)

data["month_sin"] = np.sin(
    2 * np.pi * data["Month_Number"] / 12
)

data["month_cos"] = np.cos(
    2 * np.pi * data["Month_Number"] / 12
)

feature_cols = climate_lags + [
    "Cases_lag1",
    "Cases_lag12",
    "month_sin",
    "month_cos"
]

data = data.dropna(
    subset=feature_cols + ["Cases"]
).reset_index(drop=True)

print("Feature engineering completed.")

train_data = data[
    data["date"] < "2023-01-01"
].copy()

test_data = data[
    (data["date"] >= "2023-01-01") &
    (data["date"] <= "2025-12-01")
].copy()

X_train = train_data[feature_cols]
y_train = train_data["Cases"]

X_test = test_data[feature_cols]
y_test = test_data["Cases"]

assert len(train_data) == 264
assert len(test_data) == 36

print("\nTraining months:", len(train_data))
print("Testing months:", len(test_data))

rf_model = RandomForestRegressor(
    n_estimators=300,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

pred_rf = np.maximum(
    rf_model.predict(X_test),
    0
)

xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=2,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    reg_lambda=5,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=4
)

xgb_model.fit(X_train, y_train)

pred_xgb = np.maximum(
    xgb_model.predict(X_test),
    0
)

pred_baseline = test_data[
    "Cases_lag1"
].to_numpy()

plot_df = pd.DataFrame({
    "Date": test_data["date"].to_numpy(),
    "Actual": y_test.to_numpy(),
    "Previous Month": pred_baseline,
    "Random Forest": pred_rf,
    "XGBoost": pred_xgb
})

plot_df = plot_df.sort_values(
    "Date"
).reset_index(drop=True)

print("\nPrediction data prepared.")
print(plot_df.head())

sep = plot_df[
    plot_df["Date"] == "2023-09-01"
].iloc[0]

expected_peak = {
    "Actual": 79598,
    "Previous Month": 71976,
    "Random Forest": 22508.54,
    "XGBoost": 20264.54
}

expected_mae = {
    "Previous Month": 7373.444,
    "Random Forest": 9382.196,
    "XGBoost": 9316.520
}

problems = []

print("\n===== SEPTEMBER 2023 CHECK =====")

for model, expected in expected_peak.items():

    actual = float(sep[model])

    match = np.isclose(
        actual,
        expected,
        atol=1.0
    )

    print(
        f"{model}: {actual:,.2f} "
        f"| Expected: {expected:,.2f} "
        f"| {'OK' if match else 'MISMATCH'}"
    )

    if not match:
        problems.append(model + " peak")

print("\n===== TEST MAE CHECK =====")

for model, expected in expected_mae.items():

    mae = mean_absolute_error(
        plot_df["Actual"],
        plot_df[model]
    )

    match = np.isclose(
        mae,
        expected,
        atol=1.0
    )

    print(
        f"{model}: {mae:,.3f} "
        f"| Expected: {expected:,.3f} "
        f"| {'OK' if match else 'MISMATCH'}"
    )

    if not match:
        problems.append(model + " MAE")

if problems:
    raise RuntimeError(
        "MODEL RESULTS DO NOT MATCH THE CORRECTED DATASET REFERENCE.\n"
        "Mismatches: " + ", ".join(problems)
    )

print("\nCorrected dataset reference checks passed.")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11
})

fig, ax = plt.subplots(
    figsize=(13.6, 7)
)

fig.subplots_adjust(
    left=0.09,
    right=0.985,
    bottom=0.18,
    top=0.79
)

ax.axvspan(
    pd.Timestamp("2025-01-01"),
    pd.Timestamp("2025-12-31"),
    color="#CBD2DA",
    alpha=0.34,
    zorder=0
)

ax.plot(
    plot_df["Date"],
    plot_df["Actual"],
    color="#17324D",
    linewidth=2.9,
    linestyle="-",
    marker="o",
    markersize=3.7,
    label="Observed cases",
    zorder=4
)

ax.plot(
    plot_df["Date"],
    plot_df["Previous Month"],
    color="#96A4B3",
    linewidth=2.25,
    linestyle=":",
    label="Previous-month baseline"
)

ax.plot(
    plot_df["Date"],
    plot_df["Random Forest"],
    color="#16867D",
    linewidth=2.25,
    linestyle="--",
    label="Random Forest"
)

ax.plot(
    plot_df["Date"],
    plot_df["XGBoost"],
    color="#D48636",
    linewidth=2.25,
    linestyle="-.",
    label="XGBoost"
)

ax.annotate(
    "Sep 2023: 79,598 cases",
    xy=(
        pd.Timestamp("2023-09-01"),
        79598
    ),
    xytext=(
        pd.Timestamp("2023-10-20"),
        85500
    ),
    fontsize=10,
    color="#17324D",
    arrowprops=dict(
        arrowstyle="->",
        linewidth=1.1,
        color="#17324D"
    )
)

ax.set_xlim(
    pd.Timestamp("2023-01-01"),
    pd.Timestamp("2025-12-31")
)

ax.set_ylim(0, 94500)

ticks = list(
    pd.date_range(
        "2023-01-01",
        "2025-10-01",
        freq="QS-JAN"
    )
)

ticks.append(
    pd.Timestamp("2025-12-01")
)

ax.set_xticks(ticks)

ax.xaxis.set_major_formatter(
    mdates.DateFormatter("%b %Y")
)

plt.setp(
    ax.get_xticklabels(),
    rotation=35,
    ha="right"
)

ax.yaxis.set_major_formatter(
    StrMethodFormatter("{x:,.0f}")
)

ax.set_xlabel(
    "Month",
    fontsize=12
)

ax.set_ylabel(
    "Reported dengue cases",
    fontsize=12
)

ax.grid(
    axis="y",
    linestyle="--",
    alpha=0.22
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    loc="upper right",
    ncol=2,
    frameon=False,
    fontsize=10
)

fig.text(
    0.09,
    0.94,
    "Observed vs Predicted Monthly Dengue Cases",
    fontsize=18,
    fontweight="bold",
    color="#17324D"
)

fig.text(
    0.09,
    0.90,
    "Bangladesh | One-month-ahead forecasts | 2023–2025 test period",
    fontsize=11,
    color="#58697C"
)

fig.text(
    0.09,
    0.045,
    "Shading denotes 2025, which includes provisional or derived surveillance observations.",
    fontsize=9,
    color="#58697C"
)

output_dir = Path("outputs")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

plot_df.to_csv(
    output_dir / "Fig7_Test_2023_2025_predictions.csv",
    index=False
)

fig.savefig(
    output_dir / "Fig7_Test_2023_2025_FINAL.png",
    dpi=400,
    bbox_inches="tight",
    facecolor="white"
)

fig.savefig(
    output_dir / "Fig7_Test_2023_2025_FINAL.pdf",
    bbox_inches="tight",
    facecolor="white"
)

plt.show()

print("\nFIGURE 7 CREATED SUCCESSFULLY")
print("Output folder:", output_dir.resolve())

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from pathlib import Path

csv_path = Path("outputs/Fig7_Test_2023_2025_predictions.csv")

if not csv_path.exists():
    raise FileNotFoundError(
        f"Could not find: {csv_path}\n"
        "Make sure the CSV file exists in the outputs folder."
    )

plot_df = pd.read_csv(csv_path)

plot_df = plot_df.loc[:, ~plot_df.columns.str.contains("^Unnamed")]

required_cols = ["Date", "Actual", "Previous Month", "Random Forest", "XGBoost"]
missing = [col for col in required_cols if col not in plot_df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(plot_df.columns)}"
    )

plot_df["Date"] = pd.to_datetime(plot_df["Date"])
plot_df = plot_df.sort_values("Date").reset_index(drop=True)

plot_df = plot_df[
    (plot_df["Date"] >= pd.Timestamp("2023-01-01")) &
    (plot_df["Date"] <= pd.Timestamp("2025-12-01"))
].copy()

assert len(plot_df) == 36, f"Expected 36 months, got {len(plot_df)}"
assert plot_df["Date"].min() == pd.Timestamp("2023-01-01"), "Start date should be 2023-01-01"
assert plot_df["Date"].max() == pd.Timestamp("2025-12-01"), "End date should be 2025-12-01"

peak_row = plot_df.loc[plot_df["Date"] == pd.Timestamp("2023-09-01")]

if peak_row.empty:
    raise ValueError("September 2023 row not found in the data.")

peak = peak_row.iloc[0]

print("September 2023 values:")
print(peak.to_string())

expected_values = {
    "Actual": 79598,
    "Previous Month": 71976,
    "Random Forest": 22508.54,
    "XGBoost": 20264.54
}

print("\nComparison with manuscript values:")
for key, expected in expected_values.items():
    observed = peak[key]
    print(f"{key:15s} observed = {observed:,.2f} | manuscript = {expected:,.2f}")

fig, ax = plt.subplots(figsize=(13.5, 6.8))

styles = {
    "Actual": {
        "color": "#17324D",
        "linestyle": "-",
        "linewidth": 2.8,
        "label": "Observed cases",
        "marker": "o",
        "markersize": 3.5,
        "zorder": 4
    },
    "Previous Month": {
        "color": "#94A3B8",
        "linestyle": ":",
        "linewidth": 2.3,
        "label": "Previous-month baseline",
        "marker": None,
        "zorder": 2
    },
    "Random Forest": {
        "color": "#16867D",
        "linestyle": "--",
        "linewidth": 2.3,
        "label": "Random Forest",
        "marker": None,
        "zorder": 3
    },
    "XGBoost": {
        "color": "#D3832F",
        "linestyle": "-.",
        "linewidth": 2.3,
        "label": "XGBoost",
        "marker": None,
        "zorder": 3
    }
}

for col, s in styles.items():
    ax.plot(
        plot_df["Date"],
        plot_df[col],
        color=s["color"],
        linestyle=s["linestyle"],
        linewidth=s["linewidth"],
        marker=s["marker"],
        markersize=s.get("markersize", None),
        label=s["label"],
        zorder=s["zorder"]
    )

ax.axvspan(
    pd.Timestamp("2025-01-01"),
    pd.Timestamp("2025-12-31"),
    color="#9AA4AF",
    alpha=0.14,
    zorder=0
)

ax.set_title(
    "Observed vs Predicted Monthly Dengue Cases",
    fontsize=20,
    fontweight="bold",
    pad=18
)

ax.text(
    0.005, 1.015,
    "Bangladesh | One-month-ahead forecasts | 2023–2025 test period",
    transform=ax.transAxes,
    fontsize=12.5,
    color="#556B84",
    ha="left",
    va="bottom"
)

ax.annotate(
    "Sep 2023: 79,598 cases",
    xy=(peak["Date"], peak["Actual"]),
    xytext=(20, 12),
    textcoords="offset points",
    fontsize=12,
    color="#222222",
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
    arrowprops=dict(arrowstyle="->", lw=1.2, color="#222222")
)

ax.set_xlabel("Month", fontsize=15)
ax.set_ylabel("Reported dengue cases", fontsize=15)

ax.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
plt.xticks(rotation=35, ha="right", fontsize=11)

ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.tick_params(axis="y", labelsize=11)
ax.set_ylim(bottom=0)

ax.grid(axis="y", linestyle="--", alpha=0.25)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax.legend(
    frameon=False,
    ncol=2,
    loc="upper right",
    fontsize=12
)

plt.tight_layout()

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

png_path = output_dir / "Fig7_Test_2023_2025_CORRECTED.png"
pdf_path = output_dir / "Fig7_Test_2023_2025_CORRECTED.pdf"

fig.savefig(png_path, dpi=400, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")

plt.show()

print("\nSaved files:")
print(png_path)
print(pdf_path)


# -----------------------------------------------------------------
# 27  SHAP: model explanation
# -----------------------------------------------------------------

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

rf_model = final_models["Random Forest"]

explainer = shap.TreeExplainer(rf_model)
shap_values = explainer(X_test)

importance = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean_Absolute_SHAP": np.abs(
        shap_values.values
    ).mean(axis=0)
})

importance = importance.sort_values(
    "Mean_Absolute_SHAP",
    ascending=False
)

print("===== SHAP FEATURE IMPORTANCE =====")
display(importance.round(3))

shap.plots.beeswarm(
    shap_values,
    max_display=13,
    show=False
)

plt.title(
    "SHAP Summary: Random Forest Dengue Forecasting"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "Figure_17_SHAP_Summary.png",
    dpi=400,
    bbox_inches="tight"
)

plt.show()

importance.to_csv(
    OUTPUT_DIR / "shap_feature_importance.csv",
    index=False
)

print("SHAP analysis completed!")


# -----------------------------------------------------------------
# 28  SHAP: climate features
# -----------------------------------------------------------------

import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

climate_importance = importance[
    importance["Feature"].isin(climate_lag_cols)
].copy()

labels = {
    "T2M_C_lag1": "Average temperature",
    "T2M_MIN_C_lag1": "Minimum temperature",
    "T2M_MAX_C_lag1": "Maximum temperature",
    "RH2M_pct_lag1": "Relative humidity",
    "PRECTOTCORR_SUM_mm_month_lag1": "Precipitation",
    "PS_kPa_lag1": "Surface pressure",
    "WS50M_MAX_m_s_lag1": "Maximum wind speed (50 m)",
    "QV2M_g_kg_lag1": "Specific humidity",
    "WS2M_m_s_lag1": "Wind speed (2 m)"
}

climate_importance["Label"] = (
    climate_importance["Feature"].map(labels)
)

climate_importance = climate_importance.sort_values(
    "Mean_Absolute_SHAP"
)

fig, ax = plt.subplots(figsize=(10, 6))

bars = ax.barh(
    climate_importance["Label"],
    climate_importance["Mean_Absolute_SHAP"],
    color="#16867D",
    height=0.65
)

for bar in bars:
    value = bar.get_width()
    ax.text(
        value + 3,
        bar.get_y() + bar.get_height() / 2,
        f"{value:.1f}",
        va="center",
        fontsize=10
    )

ax.set_xlim(
    0,
    climate_importance["Mean_Absolute_SHAP"].max() * 1.2
)

ax.set_title(
    "Meteorological Predictor Importance",
    fontsize=15,
    fontweight="bold",
    loc="left",
    pad=20
)

ax.set_xlabel("Mean absolute SHAP value (predicted cases)")
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_18_Climate_SHAP.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_18_Climate_SHAP.pdf",
    bbox_inches="tight"
)

plt.show()


# -----------------------------------------------------------------
# 29  Feature ablation
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

case_features = [
    "Cases_lag1",
    "Cases_lag12",
    "month_sin",
    "month_cos"
]

climate_only_features = climate_lag_cols + [
    "month_sin",
    "month_cos"
]

full_features = case_features + climate_lag_cols

feature_sets = {
    "Cases + Season": case_features,
    "Climate + Season": climate_only_features,
    "Cases + Climate + Season": full_features
}

ablation_results = []
ablation_predictions = {}

for name, features in feature_sets.items():

    model = clone(rf_original)

    model.fit(
        train[features],
        train["Cases"]
    )

    pred = np.maximum(
        model.predict(val[features]), 0
    )

    ablation_predictions[name] = pred

    actual = val["Cases"].to_numpy()

    ablation_results.append({
        "Model": name,
        "MAE": mean_absolute_error(actual, pred),
        "RMSE": np.sqrt(
            mean_squared_error(actual, pred)
        ),
        "Log_MAE": mean_absolute_error(
            np.log1p(actual),
            np.log1p(pred)
        )
    })

baseline_pred = val["Cases_lag1"].to_numpy()
actual = val["Cases"].to_numpy()

ablation_results.append({
    "Model": "Previous Month",
    "MAE": mean_absolute_error(actual, baseline_pred),
    "RMSE": np.sqrt(
        mean_squared_error(actual, baseline_pred)
    ),
    "Log_MAE": mean_absolute_error(
        np.log1p(actual),
        np.log1p(baseline_pred)
    )
})

ablation_df = (
    pd.DataFrame(ablation_results)
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("===== FEATURE ABLATION RESULTS =====")
display(ablation_df.round(3))

fig, ax = plt.subplots(figsize=(10, 5.5))

colors = [
    "#16867D" if i == 0 else "#9BAABD"
    for i in range(len(ablation_df))
]

bars = ax.barh(
    ablation_df["Model"],
    ablation_df["MAE"],
    color=colors,
    height=0.58
)

ax.invert_yaxis()

for bar in bars:
    value = bar.get_width()
    ax.text(
        value + 20,
        bar.get_y() + bar.get_height() / 2,
        f"{value:,.0f}",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

ax.set_xlim(0, ablation_df["MAE"].max() * 1.25)

ax.set_title(
    "Feature Ablation: Random Forest Validation",
    fontsize=15,
    fontweight="bold",
    loc="left",
    pad=15
)

ax.set_xlabel("Mean Absolute Error (Cases)")
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_19_Feature_Ablation.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_19_Feature_Ablation.pdf",
    bbox_inches="tight"
)

plt.show()

ablation_df.to_csv(
    OUTPUT_DIR / "feature_ablation_results.csv",
    index=False
)

print("Feature ablation analysis completed!")


# -----------------------------------------------------------------
# 30  Random Forest reproducibility check
# -----------------------------------------------------------------

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error

original_feature_order = climate_lag_cols + other_features

rf_check = clone(rf_original)

rf_check.fit(
    train[original_feature_order],
    train["Cases"]
)

pred_check = np.maximum(
    rf_check.predict(val[original_feature_order]),
    0
)

mae_original = mean_absolute_error(
    val["Cases"],
    pred_original
)

mae_check = mean_absolute_error(
    val["Cases"],
    pred_check
)

print("Original Random Forest MAE:", round(mae_original, 3))
print("Reproduced Random Forest MAE:", round(mae_check, 3))

print(
    "Predictions identical:",
    np.allclose(pred_check, pred_original)
)


# -----------------------------------------------------------------
# 31  Count distribution and overdispersion
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bayes_train = ml_data.loc[
    ml_data["date"] < "2023-01-01"
].copy()

cases = bayes_train["Cases"].astype(float)

mean_cases = cases.mean()
variance_cases = cases.var(ddof=1)
dispersion_ratio = variance_cases / mean_cases

zero_months = int((cases == 0).sum())
zero_percentage = 100 * zero_months / len(cases)

summary = pd.DataFrame({
    "Statistic": [
        "Observations",
        "Mean cases",
        "Median cases",
        "Variance",
        "Variance-to-mean ratio",
        "Minimum cases",
        "Maximum cases",
        "Zero-case months",
        "Zero-case percentage"
    ],
    "Value": [
        len(cases),
        mean_cases,
        cases.median(),
        variance_cases,
        dispersion_ratio,
        cases.min(),
        cases.max(),
        zero_months,
        zero_percentage
    ]
})

print("===== COUNT DISTRIBUTION SUMMARY =====")
display(summary.round(3))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(
    cases,
    bins=35,
    color="#16867D",
    edgecolor="white"
)

axes[0].set_title("Distribution of Monthly Dengue Cases")
axes[0].set_xlabel("Monthly cases")
axes[0].set_ylabel("Frequency")
axes[0].grid(axis="y", alpha=0.2)

axes[1].hist(
    np.log1p(cases),
    bins=35,
    color="#456A9A",
    edgecolor="white"
)

axes[1].set_title("Log-Transformed Dengue Cases")
axes[1].set_xlabel("log(1 + monthly cases)")
axes[1].set_ylabel("Frequency")
axes[1].grid(axis="y", alpha=0.2)

fig.suptitle(
    "Dengue Case Distribution: Pre-2023 Modelling Period",
    fontsize=15,
    fontweight="bold"
)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_20_Dengue_Count_Distribution.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_20_Dengue_Count_Distribution.pdf",
    bbox_inches="tight"
)

plt.show()

print("\nCount distribution analysis completed.")


# -----------------------------------------------------------------
# 32  Bayesian data preparation
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

from sklearn.preprocessing import StandardScaler

bayes_train = train.copy()
bayes_val = val.copy()

bayes_features = [
    "log_cases_lag1",
    "T2M_C_lag1",
    "RH2M_pct_lag1",
    "PRECTOTCORR_SUM_mm_month_lag1",
    "month_sin",
    "month_cos"
]

bayes_train["log_cases_lag1"] = np.log1p(
    bayes_train["Cases_lag1"]
)

bayes_val["log_cases_lag1"] = np.log1p(
    bayes_val["Cases_lag1"]
)

bayes_scaler = StandardScaler()

X_bayes_train = bayes_scaler.fit_transform(
    bayes_train[bayes_features]
)

X_bayes_val = bayes_scaler.transform(
    bayes_val[bayes_features]
)

y_bayes_train = (
    bayes_train["Cases"].to_numpy(dtype=np.int64)
)

y_bayes_val = (
    bayes_val["Cases"].to_numpy(dtype=np.int64)
)

print("===== BAYESIAN DATA PREPARATION =====")

print("Training observations:", len(y_bayes_train))
print("Validation observations:", len(y_bayes_val))
print("Number of predictors:", X_bayes_train.shape[1])

print("\nPredictors:")
print(bayes_features)

print("\nPyMC version:", pm.__version__)


# -----------------------------------------------------------------
# 33  Bayesian Negative Binomial model
# -----------------------------------------------------------------

import pymc as pm
import arviz as az
import numpy as np

with pm.Model() as bayesian_nb_model:

    intercept = pm.Normal(
        "intercept",
        mu=np.log(y_bayes_train.mean() + 1),
        sigma=2
    )

    beta = pm.Normal(
        "beta",
        mu=0,
        sigma=0.75,
        shape=X_bayes_train.shape[1]
    )

    alpha = pm.Exponential(
        "alpha",
        lam=1.0
    )

    log_mu = (
        intercept
        + pm.math.dot(X_bayes_train, beta)
    )

    mu = pm.math.exp(log_mu)

    observed_cases = pm.NegativeBinomial(
        "observed_cases",
        mu=mu,
        alpha=alpha,
        observed=y_bayes_train
    )

    trace_nb = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=4,
        target_accept=0.95,
        random_seed=42,
        return_inferencedata=True
    )

print("Bayesian model sampling completed!")

posterior_summary = az.summary(
    trace_nb,
    var_names=["intercept", "beta", "alpha"]
)

display(posterior_summary.round(3))

for name, obj in list(globals().items()):
    if hasattr(obj, "sample_stats"):
        if "diverging" in obj.sample_stats:
            print(
                f"{name} — Total divergences:",
                int(obj.sample_stats["diverging"].sum().item())
            )


# -----------------------------------------------------------------
# 34  Bayesian model diagnostics
# -----------------------------------------------------------------

import arviz as az

summary = az.summary(
    trace_nb,
    var_names=["intercept", "beta", "alpha"],
    round_to=3
)

print("===== BAYESIAN POSTERIOR SUMMARY =====")
print(summary.to_string())

print("\n===== MODEL DIAGNOSTICS =====")

print(
    "Maximum R-hat:",
    round(summary["r_hat"].max(), 3)
)

print(
    "Minimum Bulk ESS:",
    round(summary["ess_bulk"].min(), 0)
)

print(
    "Total divergences:",
    int(trace_nb.sample_stats["diverging"].sum())
)

print("\nBayesian diagnostics completed!")

print("NB stored chains:", trace_nb.posterior.sizes["chain"])
print("NB stored draws:", trace_nb.posterior.sizes["draw"])

print("NB Chains:", trace_nb.posterior.sizes["chain"])
print("NB Draws:", trace_nb.posterior.sizes["draw"])
print(
    "Divergences:",
    int(trace_nb.sample_stats["diverging"].sum())
)

import numpy as np
import pandas as pd
import arviz as az

summary = az.summary(
    trace_nb,
    var_names=["intercept", "beta", "alpha"]
)

intervals = {}

for var in ["intercept", "alpha"]:

    samples = trace_nb.posterior[var].values.flatten()

    intervals[var] = np.quantile(
        samples, [0.025, 0.975]
    )

beta_samples = trace_nb.posterior["beta"].values

beta_samples = beta_samples.reshape(
    -1, beta_samples.shape[-1]
)

for i in range(beta_samples.shape[1]):

    intervals[f"beta[{i}]"] = np.quantile(
        beta_samples[:, i],
        [0.025, 0.975]
    )

summary["CI_95_Lower"] = pd.Series({
    name: values[0]
    for name, values in intervals.items()
})

summary["CI_95_Upper"] = pd.Series({
    name: values[1]
    for name, values in intervals.items()
})

print("===== BAYESIAN POSTERIOR SUMMARY (95%) =====")

display(
    summary[
        [
            "mean",
            "sd",
            "CI_95_Lower",
            "CI_95_Upper",
            "ess_bulk",
            "r_hat"
        ]
    ].round(3)
)

print("\n===== MODEL CONVERGENCE =====")

print(
    "Maximum R-hat:",
    round(summary["r_hat"].max(), 3)
)

print(
    "Minimum Bulk ESS:",
    round(summary["ess_bulk"].min(), 0)
)

divergences = int(
    trace_nb.sample_stats["diverging"].sum()
)

print("Total divergences:", divergences)

if summary["r_hat"].max() < 1.01 and divergences == 0:
    print("Initial convergence checks passed.")
else:
    print("Further convergence investigation required.")


# -----------------------------------------------------------------
# 35  Posterior predictive checks
# -----------------------------------------------------------------

import pymc as pm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

with bayesian_nb_model:
    ppc_nb = pm.sample_posterior_predictive(
        trace_nb,
        var_names=["observed_cases"],
        random_seed=42,
        progressbar=False
    )

y_observed = np.asarray(y_bayes_train)

y_replicated = (
    ppc_nb.posterior_predictive["observed_cases"]
    .values
    .reshape(-1, len(y_observed))
)

print("Posterior predictive shape:", y_replicated.shape)

observed_stats = {
    "Mean monthly cases": y_observed.mean(),
    "Median monthly cases": np.median(y_observed),
    "Zero-case months": np.sum(y_observed == 0),
    "Maximum monthly cases": y_observed.max()
}

replicated_stats = {
    "Mean monthly cases": y_replicated.mean(axis=1),
    "Median monthly cases": np.median(y_replicated, axis=1),
    "Zero-case months": np.sum(y_replicated == 0, axis=1),
    "Maximum monthly cases": y_replicated.max(axis=1)
}

results = []

for statistic in observed_stats:

    values = replicated_stats[statistic]

    results.append({
        "Statistic": statistic,
        "Observed": observed_stats[statistic],
        "Predictive_Median": np.median(values),
        "Predictive_95_Lower": np.quantile(values, 0.025),
        "Predictive_95_Upper": np.quantile(values, 0.975)
    })

ppc_results = pd.DataFrame(results)

print("\n===== POSTERIOR PREDICTIVE CHECK RESULTS =====")
display(ppc_results.round(2))

pred_median = np.median(y_replicated, axis=0)

pred_lower = np.quantile(y_replicated, 0.05, axis=0)
pred_upper = np.quantile(y_replicated, 0.95, axis=0)

dates = pd.to_datetime(bayes_train["date"])

fig, ax = plt.subplots(figsize=(13, 6))

ax.fill_between(
    dates,
    np.log1p(pred_lower),
    np.log1p(pred_upper),
    color="#16867D",
    alpha=0.18,
    label="90% posterior predictive interval"
)

ax.plot(
    dates,
    np.log1p(y_observed),
    color="#17324D",
    linewidth=1.7,
    label="Observed cases"
)

ax.plot(
    dates,
    np.log1p(pred_median),
    color="#16867D",
    linestyle="--",
    linewidth=1.6,
    label="Posterior predictive median"
)

ax.set_title(
    "Bayesian Negative Binomial: Posterior Predictive Check",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Year")
ax.set_ylabel("log(1 + monthly dengue cases)")
ax.legend(frameon=False)
ax.grid(alpha=0.2)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_21_Bayesian_PPC.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_21_Bayesian_PPC.pdf",
    bbox_inches="tight"
)

plt.show()

ppc_results.to_csv(
    OUTPUT_DIR / "bayesian_ppc_results.csv",
    index=False
)

print("\nPosterior predictive checking completed!")


# -----------------------------------------------------------------
# 36  Model adequacy
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

observed = np.asarray(y_observed)
replicated = np.asarray(y_replicated)

lower_90 = np.quantile(replicated, 0.05, axis=0)
upper_90 = np.quantile(replicated, 0.95, axis=0)

covered = (
    (observed >= lower_90) &
    (observed <= upper_90)
)

coverage_90 = covered.mean() * 100

zero_probability = (replicated == 0).mean(axis=0)

print("===== POSTERIOR PREDICTIVE DIAGNOSTICS =====")
print("Observed zero months:", (observed == 0).sum())
print("Expected zero months:", round(zero_probability.sum(), 2))
print("In-sample 90% interval coverage:", round(coverage_90, 2), "%")
print("Months outside interval:", int((~covered).sum()))

diagnostics = pd.DataFrame({
    "Year": bayes_train["date"].dt.year.to_numpy(),
    "Observed_Zero": (observed == 0).astype(int),
    "Expected_Zero": zero_probability,
    "Covered": covered
})

yearly = diagnostics.groupby("Year", as_index=False).agg(
    Observed_Zeros=("Observed_Zero", "sum"),
    Expected_Zeros=("Expected_Zero", "sum"),
    Coverage=("Covered", "mean")
)

yearly["Coverage"] *= 100

print("\n===== YEAR-WISE DIAGNOSTICS =====")
display(yearly.round(2))

fig, ax = plt.subplots(figsize=(11, 5.5))

ax.plot(
    yearly["Year"],
    yearly["Observed_Zeros"],
    marker="o",
    linewidth=2,
    label="Observed zero-case months"
)

ax.plot(
    yearly["Year"],
    yearly["Expected_Zeros"],
    marker="s",
    linestyle="--",
    linewidth=2,
    label="Model-expected zero-case months"
)

ax.set_title(
    "Observed vs Model-Expected Zero-Case Months",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Year")
ax.set_ylabel("Number of zero-case months")
ax.set_ylim(0, 12.5)
ax.set_xticks(yearly["Year"][::2])
ax.legend(frameon=False)
ax.grid(alpha=0.2)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_22_Bayesian_Zero_Count_Diagnostics.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_22_Bayesian_Zero_Count_Diagnostics.pdf",
    bbox_inches="tight"
)

plt.show()

yearly.to_csv(
    OUTPUT_DIR / "bayesian_yearwise_diagnostics.csv",
    index=False
)

print("\nBayesian adequacy diagnostics completed!")


# -----------------------------------------------------------------
# 37  Bayesian validation
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

posterior = trace_nb.posterior

intercept_samples = posterior["intercept"].values.reshape(-1)
alpha_samples = posterior["alpha"].values.reshape(-1)

beta_samples = posterior["beta"].values.reshape(
    -1, len(bayes_features)
)

log_mu_samples = (
    intercept_samples[:, None]
    + beta_samples @ X_bayes_val.T
)

mu_samples = np.exp(log_mu_samples)

alpha_matrix = alpha_samples[:, None]

prob = alpha_matrix / (alpha_matrix + mu_samples)

rng = np.random.default_rng(42)

predictive_samples = rng.negative_binomial(
    n=alpha_matrix,
    p=prob
)

bayes_median = np.median(predictive_samples, axis=0)

bayes_lower = np.quantile(
    predictive_samples, 0.05, axis=0
)

bayes_upper = np.quantile(
    predictive_samples, 0.95, axis=0
)

actual = np.asarray(y_bayes_val)

def validation_metrics(name, predicted):
    predicted = np.maximum(
        np.asarray(predicted), 0
    )

    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "Log_MAE": mean_absolute_error(
            np.log1p(actual),
            np.log1p(predicted)
        )
    }

results = pd.DataFrame([
    validation_metrics(
        "Bayesian Negative Binomial",
        bayes_median
    ),
    validation_metrics(
        "Random Forest",
        pred_original
    ),
    validation_metrics(
        "Previous Month",
        bayes_val["Cases_lag1"]
    )
]).sort_values("MAE")

coverage = np.mean(
    (actual >= bayes_lower) &
    (actual <= bayes_upper)
) * 100

mean_width = np.mean(bayes_upper - bayes_lower)

print("===== BAYESIAN VALIDATION RESULTS =====")
display(results.round(3))

print("\n===== PREDICTIVE UNCERTAINTY =====")
print("Nominal interval coverage: 90%")
print("Observed validation coverage:", round(coverage, 2), "%")
print("Mean interval width:", round(mean_width, 2), "cases")

dates = pd.to_datetime(bayes_val["date"])

fig, ax = plt.subplots(figsize=(12, 6))

ax.fill_between(
    dates,
    bayes_lower,
    bayes_upper,
    alpha=0.22,
    color="#16867D",
    label="Bayesian 90% predictive interval"
)

ax.plot(
    dates,
    actual,
    color="#17324D",
    linewidth=2.4,
    marker="o",
    markersize=3,
    label="Observed cases"
)

ax.plot(
    dates,
    bayes_median,
    color="#16867D",
    linestyle="--",
    linewidth=2,
    label="Bayesian predictive median"
)

ax.plot(
    dates,
    pred_original,
    color="#CC873D",
    linestyle=":",
    linewidth=1.8,
    label="Random Forest"
)

ax.set_title(
    "Bayesian Dengue Forecasting: Validation 2020–2022",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Month")
ax.set_ylabel("Monthly reported dengue cases")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.2)
ax.legend(frameon=False)

fig.autofmt_xdate()
fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_23_Bayesian_Validation.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_23_Bayesian_Validation.pdf",
    bbox_inches="tight"
)

plt.show()

results.to_csv(
    OUTPUT_DIR / "bayesian_validation_results.csv",
    index=False
)

print("Bayesian validation completed!")


# -----------------------------------------------------------------
# 38  Bayesian errors by year
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bayesian_diagnostics = pd.DataFrame({
    "Date": pd.to_datetime(bayes_val["date"]).to_numpy(),
    "Actual": np.asarray(y_bayes_val),
    "Bayesian": np.asarray(bayes_median),
    "Random_Forest": np.asarray(pred_original),
    "Previous_Month": np.asarray(
        bayes_val["Cases_lag1"]
    ),
    "Lower_90": np.asarray(bayes_lower),
    "Upper_90": np.asarray(bayes_upper)
})

bayesian_diagnostics["Year"] = (
    bayesian_diagnostics["Date"].dt.year
)

results = []

for year, group in bayesian_diagnostics.groupby("Year"):

    actual = group["Actual"].to_numpy()

    coverage = np.mean(
        (actual >= group["Lower_90"].to_numpy()) &
        (actual <= group["Upper_90"].to_numpy())
    ) * 100

    width = (
        group["Upper_90"] - group["Lower_90"]
    ).mean()

    for model in [
        "Bayesian",
        "Random_Forest",
        "Previous_Month"
    ]:

        pred = group[model].to_numpy()

        results.append({
            "Year": year,
            "Model": model,
            "MAE": mean_absolute_error(actual, pred),
            "RMSE": np.sqrt(
                mean_squared_error(actual, pred)
            ),
            "Bias": np.mean(pred - actual),
            "Bayesian_Coverage_90": (
                coverage if model == "Bayesian" else np.nan
            ),
            "Bayesian_Interval_Width": (
                width if model == "Bayesian" else np.nan
            )
        })

yearwise_bayesian = pd.DataFrame(results)

print("===== BAYESIAN YEAR-WISE VALIDATION =====")
display(yearwise_bayesian.round(2))

print("\n===== BAYESIAN ANNUAL DIAGNOSTICS =====")

display(
    yearwise_bayesian[
        yearwise_bayesian["Model"] == "Bayesian"
    ].round(2)
)

peak_idx = bayesian_diagnostics["Actual"].idxmax()
peak = bayesian_diagnostics.loc[peak_idx]

print("\n===== LARGEST VALIDATION OUTBREAK =====")
print("Month:", peak["Date"].strftime("%Y-%m"))
print("Observed:", int(peak["Actual"]))
print("Bayesian median:", round(peak["Bayesian"], 1))
print("Random Forest:", round(peak["Random_Forest"], 1))
print(
    "Bayesian 90% interval:",
    round(peak["Lower_90"], 1),
    "to",
    round(peak["Upper_90"], 1)
)

yearwise_bayesian.to_csv(
    OUTPUT_DIR / "bayesian_yearwise_validation.csv",
    index=False
)

plot_data = yearwise_bayesian.pivot(
    index="Year",
    columns="Model",
    values="MAE"
)

fig, ax = plt.subplots(figsize=(10, 5.5))

plot_data[
    ["Previous_Month", "Random_Forest", "Bayesian"]
].plot(
    kind="bar",
    ax=ax,
    width=0.7
)

ax.set_title(
    "Year-wise Bayesian and ML Validation Errors",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Validation Year")
ax.set_ylabel("Mean Absolute Error (Cases)")
ax.tick_params(axis="x", rotation=0)
ax.legend(frameon=False, title="Model")
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_24_Bayesian_Yearwise_Validation.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_24_Bayesian_Yearwise_Validation.pdf",
    bbox_inches="tight"
)

plt.show()

print("Year-wise Bayesian diagnostics completed!")


# -----------------------------------------------------------------
# 39  Zero-Inflated Negative Binomial model
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az

year_fraction = (
    bayes_train["date"].dt.year
    + (bayes_train["date"].dt.month - 1) / 12
)

time_decades = (
    (year_fraction - 2010) / 10
).to_numpy(dtype=float)

with pm.Model() as zinb_model:

    intercept = pm.Normal(
        "intercept",
        mu=np.log(y_bayes_train.mean() + 1),
        sigma=2
    )

    beta = pm.Normal(
        "beta",
        mu=0,
        sigma=0.75,
        shape=X_bayes_train.shape[1]
    )

    alpha = pm.Exponential(
        "alpha",
        lam=1
    )

    zi_intercept = pm.Normal(
        "zi_intercept",
        mu=-2,
        sigma=1.5
    )

    zi_trend = pm.Normal(
        "zi_trend",
        mu=0,
        sigma=1
    )

    zero_prob = pm.math.sigmoid(
        zi_intercept + zi_trend * time_decades
    )

    psi = 1 - zero_prob

    log_mu = (
        intercept
        + pm.math.dot(X_bayes_train, beta)
    )

    mu = pm.math.exp(log_mu)

    observed = pm.ZeroInflatedNegativeBinomial(
        "observed",
        psi=psi,
        mu=mu,
        alpha=alpha,
        observed=y_bayes_train
    )

    trace_zinb = pm.sample(
        draws=1000,
        tune=1000,
        chains=4,
        cores=2,
        target_accept=0.95,
        random_seed=42,
        progressbar=False,
        return_inferencedata=True
    )

zinb_summary = az.summary(
    trace_zinb,
    var_names=[
        "intercept",
        "beta",
        "alpha",
        "zi_intercept",
        "zi_trend"
    ]
)

print("===== ZINB POSTERIOR SUMMARY =====")
display(zinb_summary.round(3))

print("\n===== ZINB CONVERGENCE =====")

print(
    "Maximum R-hat:",
    round(zinb_summary["r_hat"].max(), 3)
)

print(
    "Minimum Bulk ESS:",
    round(zinb_summary["ess_bulk"].min(), 0)
)

print(
    "Total divergences:",
    int(trace_zinb.sample_stats["diverging"].sum())
)

print("\nZINB model fitting completed.")


# -----------------------------------------------------------------
# 40  NB versus ZINB posterior predictive checks
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with zinb_model:
    ppc_zinb = pm.sample_posterior_predictive(
        trace_zinb,
        var_names=["observed"],
        random_seed=42,
        progressbar=False
    )

observed = np.asarray(y_bayes_train)

zinb_rep = (
    ppc_zinb.posterior_predictive["observed"]
    .values
    .reshape(-1, len(observed))
)

nb_rep = np.asarray(y_replicated)

assert nb_rep.shape[1] == len(observed)
assert zinb_rep.shape[1] == len(observed)

print("NB predictive shape:", nb_rep.shape)
print("ZINB predictive shape:", zinb_rep.shape)

def predictive_statistics(samples):
    return {
        "Mean Cases": samples.mean(axis=1),
        "Median Cases": np.median(samples, axis=1),
        "Zero Months": (samples == 0).sum(axis=1),
        "Maximum Cases": samples.max(axis=1)
    }

nb_stats = predictive_statistics(nb_rep)
zinb_stats = predictive_statistics(zinb_rep)

observed_stats = {
    "Mean Cases": observed.mean(),
    "Median Cases": np.median(observed),
    "Zero Months": (observed == 0).sum(),
    "Maximum Cases": observed.max()
}

results = []

for model_name, stats in [
    ("Negative Binomial", nb_stats),
    ("Zero-Inflated NB", zinb_stats)
]:
    for statistic, values in stats.items():
        results.append({
            "Model": model_name,
            "Statistic": statistic,
            "Observed": observed_stats[statistic],
            "Predictive_Median": np.median(values),
            "CI_95_Lower": np.quantile(values, 0.025),
            "CI_95_Upper": np.quantile(values, 0.975)
        })

ppc_comparison = pd.DataFrame(results)

print("===== NB VS ZINB POSTERIOR PREDICTIVE CHECK =====")
display(ppc_comparison.round(2))

nb_zeros = nb_stats["Zero Months"]
zinb_zeros = zinb_stats["Zero Months"]
observed_zeros = observed_stats["Zero Months"]

fig, ax = plt.subplots(figsize=(10, 5.5))

ax.hist(
    nb_zeros,
    bins=np.arange(-0.5, 111.5, 2),
    alpha=0.55,
    label="Negative Binomial",
    color="#7895B2",
    density=True
)

ax.hist(
    zinb_zeros,
    bins=np.arange(-0.5, 111.5, 2),
    alpha=0.55,
    label="Zero-Inflated NB",
    color="#16867D",
    density=True
)

ax.axvline(
    observed_zeros,
    color="#B33A3A",
    linewidth=2.5,
    linestyle="--",
    label=f"Observed zeros ({observed_zeros})"
)

ax.set_title(
    "Posterior Predictive Comparison of Zero-Case Months",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Number of zero-case months")
ax.set_ylabel("Predictive density")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)

plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_25_NB_vs_ZINB_Zero_Counts.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_25_NB_vs_ZINB_Zero_Counts.pdf",
    bbox_inches="tight"
)

plt.show()

ppc_comparison.to_csv(
    OUTPUT_DIR / "NB_vs_ZINB_PPC.csv",
    index=False
)

print("NB vs ZINB comparison completed!")


# -----------------------------------------------------------------
# 41  NB versus ZINB validation
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.special import expit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

posterior = trace_zinb.posterior

intercept = posterior["intercept"].values.reshape(-1)
alpha = posterior["alpha"].values.reshape(-1)

beta = posterior["beta"].values.reshape(
    -1, len(bayes_features)
)

zi_intercept = posterior["zi_intercept"].values.reshape(-1)
zi_trend = posterior["zi_trend"].values.reshape(-1)

dates_val = pd.to_datetime(bayes_val["date"])

year_fraction = (
    dates_val.dt.year.to_numpy()
    + (dates_val.dt.month.to_numpy() - 1) / 12
)

time_val = (year_fraction - 2010) / 10

log_mu = (
    intercept[:, None]
    + beta @ X_bayes_val.T
)

mu = np.exp(log_mu)

zero_probability = expit(
    zi_intercept[:, None]
    + zi_trend[:, None] * time_val[None, :]
)

psi = 1 - zero_probability

rng = np.random.default_rng(42)

alpha_matrix = alpha[:, None]

nb_probability = (
    alpha_matrix / (alpha_matrix + mu)
)

nb_draws = rng.negative_binomial(
    n=alpha_matrix,
    p=nb_probability
)

zinb_draws = np.where(
    rng.random(nb_draws.shape) < psi,
    nb_draws,
    0
)

zinb_median = np.median(zinb_draws, axis=0)

zinb_lower = np.quantile(
    zinb_draws, 0.05, axis=0
)

zinb_upper = np.quantile(
    zinb_draws, 0.95, axis=0
)

actual = np.asarray(y_bayes_val)

def bayesian_metrics(name, predicted, lower, upper):

    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": np.sqrt(
            mean_squared_error(actual, predicted)
        ),
        "Log_MAE": mean_absolute_error(
            np.log1p(actual),
            np.log1p(predicted)
        ),
        "Coverage_90_%": np.mean(
            (actual >= lower) &
            (actual <= upper)
        ) * 100,
        "Mean_Width": np.mean(upper - lower)
    }

bayesian_comparison = pd.DataFrame([
    bayesian_metrics(
        "Negative Binomial",
        bayes_median,
        bayes_lower,
        bayes_upper
    ),
    bayesian_metrics(
        "Zero-Inflated NB",
        zinb_median,
        zinb_lower,
        zinb_upper
    )
])

print("===== BAYESIAN FORECASTING COMPARISON =====")
display(bayesian_comparison.round(3))

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(
    dates_val,
    np.log1p(actual),
    label="Observed cases",
    color="#17324D",
    linewidth=2.7
)

ax.plot(
    dates_val,
    np.log1p(bayes_median),
    label="Negative Binomial",
    color="#16867D",
    linestyle="--",
    linewidth=2
)

ax.plot(
    dates_val,
    np.log1p(zinb_median),
    label="Zero-Inflated NB",
    color="#C77C35",
    linestyle="-.",
    linewidth=2
)

ax.set_title(
    "Bayesian Forecast Comparison (2020–2022)",
    fontsize=15,
    fontweight="bold"
)

ax.set_xlabel("Month")
ax.set_ylabel("log(1 + reported dengue cases)")
ax.legend(frameon=False)
ax.grid(alpha=0.2)

fig.autofmt_xdate()
plt.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_26_NB_vs_ZINB_Validation.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_26_NB_vs_ZINB_Validation.pdf",
    bbox_inches="tight"
)

plt.show()

bayesian_comparison.to_csv(
    OUTPUT_DIR / "NB_vs_ZINB_validation.csv",
    index=False
)

print("Bayesian forecasting comparison completed!")


# -----------------------------------------------------------------
# 42  Bayesian model comparison by year
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

comparison_data = pd.DataFrame({
    "Date": pd.to_datetime(bayes_val["date"]).to_numpy(),
    "Actual": np.asarray(y_bayes_val),
    "NB": np.asarray(bayes_median),
    "ZINB": np.asarray(zinb_median),
    "NB_Lower": np.asarray(bayes_lower),
    "NB_Upper": np.asarray(bayes_upper),
    "ZINB_Lower": np.asarray(zinb_lower),
    "ZINB_Upper": np.asarray(zinb_upper)
})

comparison_data["Year"] = comparison_data["Date"].dt.year

results = []

for year, group in comparison_data.groupby("Year"):

    actual = group["Actual"].to_numpy()

    for model, lower_col, upper_col in [
        ("NB", "NB_Lower", "NB_Upper"),
        ("ZINB", "ZINB_Lower", "ZINB_Upper")
    ]:

        predicted = group[model].to_numpy()
        lower = group[lower_col].to_numpy()
        upper = group[upper_col].to_numpy()

        results.append({
            "Year": year,
            "Model": model,
            "MAE": mean_absolute_error(actual, predicted),
            "RMSE": np.sqrt(
                mean_squared_error(actual, predicted)
            ),
            "Bias": np.mean(predicted - actual),
            "Coverage_90": 100 * np.mean(
                (actual >= lower) & (actual <= upper)
            ),
            "Interval_Width": np.mean(upper - lower)
        })

yearly_comparison = pd.DataFrame(results)

print("===== YEAR-WISE NB VS ZINB COMPARISON =====")
display(yearly_comparison.round(2))

mae_table = yearly_comparison.pivot(
    index="Year",
    columns="Model",
    values="MAE"
)

fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(mae_table))
width = 0.34

bars_nb = ax.bar(
    x - width / 2,
    mae_table["NB"],
    width,
    label="Negative Binomial",
    color="#7895B2"
)

bars_zinb = ax.bar(
    x + width / 2,
    mae_table["ZINB"],
    width,
    label="Zero-Inflated NB",
    color="#16867D"
)

for bars in [bars_nb, bars_zinb]:
    for bar in bars:
        height = bar.get_height()

        ax.annotate(
            f"{height:,.0f}",
            (bar.get_x() + bar.get_width()/2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=10
        )

ax.set_xticks(x)
ax.set_xticklabels(mae_table.index.astype(str))

ax.set_title(
    "Year-wise Bayesian Forecasting Performance",
    fontsize=15,
    fontweight="bold",
    loc="left"
)

ax.set_xlabel("Validation Year")
ax.set_ylabel("Mean Absolute Error (Cases)")
ax.legend(frameon=False)

ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_27_Bayesian_Yearwise_Comparison.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_27_Bayesian_Yearwise_Comparison.pdf",
    bbox_inches="tight"
)

plt.show()

yearly_comparison.to_csv(
    OUTPUT_DIR / "bayesian_yearwise_comparison.csv",
    index=False
)

print("Year-wise Bayesian comparison completed!")


# -----------------------------------------------------------------
# 43  Bayesian bootstrap comparison
# -----------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

actual = comparison_data["Actual"].to_numpy()
nb_pred = comparison_data["NB"].to_numpy()
zinb_pred = comparison_data["ZINB"].to_numpy()

error_diff = (
    np.abs(actual - zinb_pred)
    - np.abs(actual - nb_pred)
)

observed_diff = error_diff.mean()

B = 5000
block_lengths = [3, 6, 12]
n = len(error_diff)

results = []

for block in block_lengths:

    rng = np.random.default_rng(42)
    bootstrap_values = []

    for _ in range(B):

        starts = rng.integers(
            0,
            n - block + 1,
            size=int(np.ceil(n / block))
        )

        indices = np.concatenate([
            np.arange(s, s + block)
            for s in starts
        ])[:n]

        bootstrap_values.append(
            error_diff[indices].mean()
        )

    lower, upper = np.percentile(
        bootstrap_values,
        [2.5, 97.5]
    )

    results.append({
        "Block_Length": block,
        "MAE_Difference": observed_diff,
        "CI_Lower": lower,
        "CI_Upper": upper,
        "Includes_Zero": lower <= 0 <= upper
    })

bayesian_bootstrap = pd.DataFrame(results)

print("===== BAYESIAN BOOTSTRAP COMPARISON =====")
display(bayesian_bootstrap.round(2))

fig, ax = plt.subplots(figsize=(9, 5))

y_positions = np.arange(len(block_lengths))

for i, row in bayesian_bootstrap.iterrows():

    ax.plot(
        [row["CI_Lower"], row["CI_Upper"]],
        [i, i],
        linewidth=3,
        color="#16867D"
    )

    ax.scatter(
        row["MAE_Difference"],
        i,
        s=90,
        color="#17324D",
        zorder=3
    )

ax.axvline(
    0,
    color="#9A3D3D",
    linestyle="--",
    linewidth=1.5,
    label="No difference"
)

ax.set_yticks(y_positions)
ax.set_yticklabels([
    f"{b}-month blocks" for b in block_lengths
])

ax.invert_yaxis()

ax.set_title(
    "Uncertainty in Bayesian Model Performance",
    fontsize=15,
    fontweight="bold",
    loc="left"
)

ax.set_xlabel(
    "MAE Difference: ZINB minus NB (cases)"
)

ax.grid(axis="x", alpha=0.2)
ax.legend(frameon=False)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "Figure_28_Bayesian_Bootstrap.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_28_Bayesian_Bootstrap.pdf",
    bbox_inches="tight"
)

plt.show()

bayesian_bootstrap.to_csv(
    OUTPUT_DIR / "bayesian_bootstrap_comparison.csv",
    index=False
)

print("Bayesian bootstrap comparison completed!")


# -----------------------------------------------------------------
# 44  Consolidated model performance
# -----------------------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

ml_summary = all_results[
    ["Model", "MAE", "RMSE", "Log_MAE"]
].copy()

bayes_summary = bayesian_comparison[
    ["Model", "MAE", "RMSE", "Log_MAE"]
].copy()

final_comparison = pd.concat(
    [ml_summary, bayes_summary],
    ignore_index=True
)

final_comparison = (
    final_comparison
    .sort_values("MAE")
    .reset_index(drop=True)
)

print("===== ALL MODEL VALIDATION COMPARISON =====")
display(final_comparison.round(2))

fig, ax = plt.subplots(figsize=(11, 6.5))

best_model = final_comparison.iloc[0]["Model"]

colors = [
    "#16867D" if model == best_model else "#9BAABD"
    for model in final_comparison["Model"]
]

bars = ax.barh(
    final_comparison["Model"],
    final_comparison["MAE"],
    color=colors,
    height=0.6
)

ax.invert_yaxis()

for bar in bars:
    value = bar.get_width()

    ax.text(
        value + 25,
        bar.get_y() + bar.get_height() / 2,
        f"{value:,.0f}",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

ax.set_xlim(
    0,
    final_comparison["MAE"].max() * 1.25
)

ax.set_title(
    "Comparison of Dengue Forecasting Models",
    fontsize=16,
    fontweight="bold",
    loc="left",
    pad=20
)

ax.set_xlabel(
    "Mean Absolute Error (Reported Cases)"
)

ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.text(
    0.13, 0.015,
    "One-month-ahead forecasting | Development validation: 2020–2022",
    fontsize=9
)

plt.tight_layout(rect=[0, 0.04, 1, 1])

fig.savefig(
    OUTPUT_DIR / "Figure_29_All_Model_Comparison.png",
    dpi=400,
    bbox_inches="tight"
)

fig.savefig(
    OUTPUT_DIR / "Figure_29_All_Model_Comparison.pdf",
    bbox_inches="tight"
)

plt.show()

final_comparison.to_csv(
    OUTPUT_DIR / "all_model_validation_results.csv",
    index=False
)

print("Consolidated comparison completed!")

# S1_TABLE — ANNUAL DENGUE DATA WITH PROVENANCE

import pandas as pd
from pathlib import Path

public_data = df_updated.copy()

audited_path = Path(
    "Data/Bangladesh_Dengue_2000_2025_UPDATED_AUDITED.xlsx"
)

if audited_path.is_file():
    audited_data = pd.read_excel(
        audited_path,
        sheet_name="Research_Data"
    )

    def unique_sources(values):
        return " | ".join(
            dict.fromkeys(values.dropna().astype(str))
        )

    required_columns = [
        "Year",
        "Cases",
        "Deaths",
        "Dengue_Record_Status",
        "Cases_Source",
        "Deaths_Source"
    ]

    missing = [
        col for col in required_columns
        if col not in audited_data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in audited dataset: {missing}"
        )

    # Check that audited and public data agree on every month.
    public_check = public_data[
        ["Year", "Month_Number", "Cases", "Deaths"]
    ].sort_values(["Year", "Month_Number"]).reset_index(drop=True)

    audited_check = audited_data[
        ["Year", "Month_Number", "Cases", "Deaths"]
    ].sort_values(["Year", "Month_Number"]).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        public_check,
        audited_check,
        check_dtype=False
    )

    # Build annual S1 table with provenance.
    annual_table = (
        audited_data.groupby("Year")
        .agg(
            Cases=("Cases", "sum"),
            Deaths=("Deaths", "sum"),
            Months=("Cases", "size"),
            Record_Status=(
                "Dengue_Record_Status",
                unique_sources
            ),
            Cases_Source=("Cases_Source", unique_sources),
            Deaths_Source=("Deaths_Source", unique_sources)
        )
        .reset_index()
    )

    assert len(annual_table) == 26
    assert (annual_table["Months"] == 12).all()

    # Save S1 table.
    output_path = Path(
        "outputs/S1_Table_Annual_Dengue_Provenance_FINAL.xlsx"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    annual_table.to_excel(
        output_path,
        index=False
    )

    print(annual_table.to_string(index=False))

    print("\nS1 Table saved successfully!")
    print("Saved to:", output_path)
    print("Provenance columns included.")
else:
    print("OPTIONAL S1 PROVENANCE EXPORT SKIPPED: audited source workbook not found.")
    print("Expected:", audited_path)
    print("Other analyses and tables remain available; source-level provenance must be documented separately.")


# -----------------------------------------------------------------
# 45  Supplementary Table S3: reproducible posterior summaries and
#     independently reported overall/year-wise validation metrics.
#     Reuses the already fitted NB/ZINB traces and predictions above;
#     no unnecessary second MCMC fitting.
# -----------------------------------------------------------------

_s3_dir = Path("outputs")
_s3_dir.mkdir(parents=True, exist_ok=True)

_s3_parameters = []
_s3_diagnostics = []
for _model_name, _trace, _vars in [
    ("NB", trace_nb, ["intercept", "beta", "alpha"]),
    ("ZINB", trace_zinb, ["intercept", "beta", "zi_intercept", "zi_trend", "alpha"]),
]:
    _summary = az.summary(_trace, var_names=_vars)
    _s3_diagnostics.append({
        "Model": _model_name,
        "Chains": int(_trace.posterior.sizes["chain"]),
        "Draws_per_chain": int(_trace.posterior.sizes["draw"]),
        "Max_Rhat": float(_summary["r_hat"].max()),
        "Min_bulk_ESS": float(_summary["ess_bulk"].min()),
        "Min_tail_ESS": float(_summary["ess_tail"].min()),
        "Divergences": int(_trace.sample_stats["diverging"].sum()),
    })
    for _var in _vars:
        _values = _trace.posterior[_var].values
        _flat = _values.reshape(-1, _values.shape[-1]) if _var == "beta" else _values.reshape(-1, 1)
        for _idx in range(_flat.shape[1]):
            _key = f"beta[{_idx}]" if _var == "beta" else _var
            _q = np.quantile(_flat[:, _idx], [0.025, 0.055, 0.945, 0.975])
            _s3_parameters.append({
                "Model": _model_name, "Parameter": _key,
                "Mean": float(_flat[:, _idx].mean()),
                "SD": float(_flat[:, _idx].std(ddof=1)),
                "CI_89_lower": float(_q[1]), "CI_89_upper": float(_q[2]),
                "CI_95_lower": float(_q[0]), "CI_95_upper": float(_q[3]),
                "Bulk_ESS": float(_summary.loc[_key, "ess_bulk"]),
                "Tail_ESS": float(_summary.loc[_key, "ess_tail"]),
                "Rhat": float(_summary.loc[_key, "r_hat"]),
            })

pd.DataFrame(_s3_diagnostics).to_csv(_s3_dir / "S3_MCMC_diagnostics.csv", index=False)
pd.DataFrame(_s3_parameters).to_csv(_s3_dir / "S3_posterior_89_95_equal_tailed.csv", index=False)

_s3_rows = []
_s3_monthly = []
for _name, _med_col, _low_col, _up_col in [
    ("NB", "NB", "NB_Lower", "NB_Upper"),
    ("ZINB", "ZINB", "ZINB_Lower", "ZINB_Upper"),
]:
    _d = comparison_data.copy()
    for _period, _g in [("2020-2022", _d)] + [
        (str(_year), _d.loc[_d["Date"].dt.year == _year])
        for _year in (2020, 2021, 2022)
    ]:
        _y = _g["Actual"].to_numpy()
        _pred = _g[_med_col].to_numpy()
        _lo = _g[_low_col].to_numpy()
        _hi = _g[_up_col].to_numpy()
        _s3_rows.append({
            "Period": _period, "Model": _name, "Months": len(_g),
            "MAE": mean_absolute_error(_y, _pred),
            "RMSE": np.sqrt(mean_squared_error(_y, _pred)),
            "Coverage_percent": 100 * np.mean((_y >= _lo) & (_y <= _hi)),
            "Mean_interval_width": np.mean(_hi - _lo),
        })
    for _, _row in _d.iterrows():
        _s3_monthly.append({
            "Date": _row["Date"], "Model": _name,
            "Actual": _row["Actual"],
            "Predicted_Median": _row[_med_col],
            "Lower_90": _row[_low_col], "Upper_90": _row[_up_col],
        })

_s3_validation = pd.DataFrame(_s3_rows)
_s3_validation.to_csv(_s3_dir / "S3_yearly_independent_check.csv", index=False)
pd.DataFrame(_s3_monthly).to_csv(_s3_dir / "S3_monthly_validation_predictions.csv", index=False)

# Catch accidental changes to published Table 3 / supplementary Table S3.
_s3_expected = {
    ("2020-2022", "NB"): (2243.86, 5157.95, 88.89, 3951.42),
    ("2020-2022", "ZINB"): (2170.10, 5033.03, 86.11, 3229.30),
    ("2020", "NB"): (99.17, 142.04, 100.00, 816.26),
    ("2020", "ZINB"): (99.92, 132.62, 100.00, 756.32),
    ("2021", "NB"): (1909.38, 3011.52, 91.67, 5374.89),
    ("2021", "ZINB"): (1811.96, 2875.19, 91.67, 4412.94),
    ("2022", "NB"): (4723.04, 8409.75, 75.00, 5663.11),
    ("2022", "ZINB"): (4598.42, 8228.61, 66.67, 4518.64),
}
for _row in _s3_validation.itertuples(index=False):
    _expected = np.array(_s3_expected[(_row.Period, _row.Model)])
    _actual = np.array([_row.MAE, _row.RMSE, _row.Coverage_percent, _row.Mean_interval_width])
    if not np.allclose(_actual, _expected, atol=0.02, rtol=0):
        raise AssertionError(f"S3 mismatch for {_row.Period}/{_row.Model}: {_actual} vs {_expected}")

print("\n===== S3 OVERALL AND YEAR-WISE VALIDATION: VERIFIED =====")
print(_s3_validation.round(2).to_string(index=False))
print("Saved S3 diagnostics, 89%/95% posterior intervals, validation metrics and monthly predictions to outputs/.")
