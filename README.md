# FX Risk Analytics: Dynamic Hedge Ratio Estimation for INR Currency Futures

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red)
![Power BI](https://img.shields.io/badge/PowerBI-Dashboard-yellow)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![Models](https://img.shields.io/badge/Models-OLS%20%7C%20VECM%20%7C%20GJR--DCC%20%7C%20BEKK-purple)
![Data](https://img.shields.io/badge/Data-2015--2024-lightgrey)

## Project Overview

Indian corporates with foreign currency exposure face significant FX risk - a $1M USD receivable can lose lakhs of rupees in value before settlement if the exchange rate moves unfavorably. The standard question for any corporate treasurer or FX desk is: how much of this exposure should be hedged using currency futures, and does the answer change over time?

This project estimates the **optimal minimum-variance hedge ratio** for USD/INR and EUR/INR currency futures across a 10-year sample (January 2015 - December 2024), comparing four progressively sophisticated models: a simple OLS baseline, a VECM capturing the long-run cointegrating relationship between spot and futures, a GJR-DCC-GARCH model estimating daily time-varying hedge ratios, and a diagonal BEKK-GARCH model jointly estimating the full covariance matrix.

The central research question is whether dynamic GARCH-based models - which update the hedge ratio daily based on current market conditions - outperform the simpler constant OLS estimate out-of-sample. Every model and formula is grounded in peer-reviewed academic literature. The project delivers findings through an interactive Streamlit advisory dashboard and a Power BI analytics report, both allowing users to input their firm's FX exposure and receive a live hedge recommendation with accompanying risk metrics.

---

## Live Links

### Streamlit Application
[Live Streamlit app](https://fx-hedge-ratio-analytics-6pxjbmgksolgavdem7dtny.streamlit.app/)

### Power BI Dashboard


---

## Datasets

### Dataset Sources
- **Spot rates and futures prices** - Investing.com (USD/INR and EUR/INR daily OHLC)
- **VIX (Global Fear Index)** - FRED, Federal Reserve Bank of St. Louis
- **NSE currency futures settlement prices** - Investing.com futures historical data

### Dataset Description

| Property | Detail |
|---|---|
| Currency pairs | USD/INR and EUR/INR (spot + futures) |
| Frequency | Daily |
| Raw period | January 2015 - December 2024 |
| Price observations | 2,403 (after merging on common trading dates) |
| Return observations | 2,402 (log returns computed from prices) |
| Training sample | 2,161 observations (Jan 2015 - Dec 2023) |
| Test sample | 241 observations (Jan 2024 - Dec 2024) |
| Control variable | VIX (daily closing level) |
| Data type | Time series, daily frequency |

All five raw files are stored in `Data/Original/`. Cleaned outputs are stored in `Data/Results/`.

**Preprocessing steps:**
- Standardized date formats across all Investing.com and FRED sources
- Merged five files on common trading dates using inner join (eliminates date mismatches between spot, futures, and VIX)
- Forward-filled 51 missing VIX observations on holidays
- Computed log returns: r_t = ln(P_t / P_{t-1})
- Split into training (2015-2023) and test (2024) samples before any model estimation

### Key Features

| Feature | Description |
|---|---|
| USDINR_Spot | Daily USD/INR spot exchange rate (INR per USD) |
| EURINR_Spot | Daily EUR/INR spot exchange rate (INR per EUR) |
| USDINR_Futures | Daily USD/INR nearest-contract futures settlement price |
| EURINR_Futures | Daily EUR/INR nearest-contract futures settlement price |
| VIX | CBOE Volatility Index - proxy for global risk sentiment |
| r_USDINR_Spot | Log return of USD/INR spot rate |
| r_EURINR_Spot | Log return of EUR/INR spot rate |
| r_USDINR_Futures | Log return of USD/INR futures price |
| r_EURINR_Futures | Log return of EUR/INR futures price |

---

## Outputs and Analysis

### Exploratory Data Analysis

**Price levels:**
- USD/INR appreciated from ~62 to ~85 over the sample (INR depreciation of ~37%)
- EUR/INR moved from ~67 to ~91 over the same period
- Both series show persistent upward trends confirming non-stationarity in levels

**Log returns:**
- USD/INR mean daily return: 0.000134, std dev: 0.003227
- EUR/INR mean daily return: 0.000098, std dev: 0.005376
- EUR/INR is approximately 1.7x more volatile than USD/INR on a daily basis
- Clear volatility clustering is visible across both return series - large moves cluster together, particularly during COVID-19 (March-June 2020) where VIX spiked to ~80

**Distributional properties:**
- Kurtosis of 6.74 (USD/INR) and 5.98 (EUR/INR) - both substantially above the normal value of 3
- Positive skewness in both pairs - large INR depreciation events (positive returns in USD/INR) are more common than appreciation events of equal size
- Both pairs fail the Jarque-Bera normality test at the 1% level, confirming fat-tailed distributions

### Statistical Diagnostics

All model choices are justified by formal statistical tests on the data - not assumed.

| Test | Result | Implication |
|---|---|---|
| ADF - price levels | p = 0.876-0.892, fail to reject | Prices are non-stationary (I(1)) - use returns |
| ADF - log returns | p = 0.000 for all series | Returns are stationary (I(0)) - models are valid |
| Johansen - USD/INR | Trace statistic = 525 vs CV 15.49 | Strong cointegration - VECM is required |
| Johansen - EUR/INR | Trace statistic = 843 vs CV 15.49 | Strong cointegration - VECM is required |
| ARCH-LM test | LM = 109-201, p = 0.000 for all | Volatility clustering confirmed - GARCH justified |
| Jarque-Bera | Kurtosis 5.9-6.7, p = 0.000 | Fat tails - Student-t distribution used in GARCH |
| Breusch-Pagan | Rejected for both pairs | Heteroskedasticity in OLS residuals - confirms GARCH needed |

### Model Results

**Hedge ratios estimated:**

| Model | Paper | USD/INR h* | EUR/INR h* |
|---|---|---|---|
| OLS | Ederington (1979) | 0.789 | 0.545 |
| VECM | Kroner and Sultan (1993) | 0.929 | 0.939 |
| GJR-DCC-GARCH | Glosten et al. (1993) + Engle (2002) | Dynamic (mean 0.776) | Dynamic (mean 0.512) |
| BEKK-GARCH | Engle and Kroner (1995) | Dynamic (mean 0.801) | Dynamic (mean 0.529) |

**Out-of-sample hedging effectiveness (2024 test period):**

| Model | USD/INR OOS HE | EUR/INR OOS HE |
|---|---|---|
| OLS | **62.29%** | **11.53%** |
| VECM | 60.22% | -16.53% |
| GJR-DCC-GARCH | 61.71% | 13.04% |
| BEKK-GARCH | 61.25% | 11.58% |

HE = 1 - Var(hedged portfolio) / Var(unhedged portfolio). Source: Ederington (1979).

**Risk metrics on $1M exposure (95% confidence, 30-day horizon):**

| Metric | USD/INR | EUR/INR |
|---|---|---|
| VaR (unhedged) | Rs 9.90 lakhs | Rs 40.97 lakhs |
| VaR (hedged) | Rs 3.76 lakhs | Rs 36.25 lakhs |
| CVaR (unhedged) | Rs 12.42 lakhs | Rs 51.38 lakhs |
| CVaR (hedged) | Rs 4.71 lakhs | Rs 45.47 lakhs |

EUR/INR carries approximately 4x the VaR of USD/INR on the same notional exposure.

### Insights and Recommendations

**Finding 1 - OLS dominates for USD/INR:** The constant minimum-variance hedge ratio of 0.789 produces the highest out-of-sample hedging effectiveness (62.3%) across all four models. This is attributed to India's RBI cash-settlement mechanism for currency futures, which structurally stabilizes the spot-futures basis and leaves limited time-variation for dynamic models to exploit.

**Finding 2 - VECM is actively harmful for EUR/INR:** The VECM hedge ratio of 0.939 applied to 2024 data produces -16.53% hedging effectiveness - worse than no hedge. The over-correction in the error correction term introduces noise rather than signal for the EUR/INR pair.

**Finding 3 - EUR/INR futures are an unreliable hedge instrument:** Across all four models, EUR/INR consistently underperforms USD/INR. The root cause is weak spot-futures correlation (0.484 vs 0.762 for USD/INR) and low trading volume in EUR/INR contracts on NSE. No model reliably hedges EUR/INR exposure through exchange-traded currency futures.

**Finding 4 - Instrument selection > model complexity:** The primary determinant of hedging effectiveness is the structural relationship between spot and futures markets, not the sophistication of the estimation method. Indian corporates should prioritize USD/INR futures for FX hedging and consider OTC instruments for EUR/INR exposure.

**Finding 5 - Two volatility regimes in USD/INR:** Markov regime switching (Hamilton 1989) confirms a calm regime (hedge ratio 0.723) and a crisis regime (hedge ratio 0.907). Optimal hedging intensity increases by 18.4 percentage points during market stress.

### Outputs Generated

- `ols_results.csv` - OLS hedge ratios and hedging effectiveness for both pairs
- `vecm_results.csv` - VECM hedge ratios, error correction coefficients, lambda values
- `dcc_results.csv` - GJR-DCC-GARCH parameters and hedging effectiveness
- `bekk_results.csv` - BEKK-GARCH parameters and hedging effectiveness
- `dcc_hedge_ratios_IS.csv` - Daily time-varying hedge ratios and correlations (2015-2023)
- `dcc_hedge_ratios_OOS.csv` - Daily time-varying hedge ratios and correlations (2024)
- `volatility_forecasts.csv` - 30-day GARCH volatility forecasts for all series
- `risk_metrics.csv` - VaR, CVaR, CFaR for default $1M exposure
- Streamlit dashboard (3 pages, fully interactive)
- Power BI report (2 pages, interactive slicers)
- 7 diagnostic and results PNG plots

### Challenges and Solutions

| Challenge | Solution |
|---|---|
| EUR/INR DCC optimizer consistently hit lower bounds (a+b = 0.13) | Switched from Nelder-Mead to L-BFGS-B with bounded search space and multiple starting points |
| BEKK-GARCH convergence taking 30+ minutes with full BEKK | Switched to diagonal BEKK specification (7 parameters vs 11) - retains time-varying covariance properties with substantially faster estimation |
| pandas .last() method deprecated in newer versions | Replaced with boolean index filter using pd.Timedelta cutoff |
| EGARCH OOS forecasting produced NaN due to alpha+beta > 1 | Switched to GJR-GARCH which stays in the positive-variance GARCH framework - no numerical instability |
| USD/INR GARCH showed IGARCH behavior (alpha+beta = 1.000) | Capped persistence at 0.9999 for forecast recursion - standard practice for near-integrated GARCH |
| Streamlit Cloud deployment failed on data path | Replaced hardcoded Windows paths with os.path.dirname(__file__) for cross-platform compatibility |

---

## Tech Stack

| Category | Technologies Used |
|---|---|
| Programming language | Python 3.11 |
| Data wrangling | pandas, numpy |
| Time series and econometrics | statsmodels (VAR, VECM, ARIMA, cointegration tests) |
| Volatility modeling | arch (GARCH, GJR-GARCH, EGARCH) |
| Statistical testing | scipy.stats (Jarque-Bera, ADF, normality tests) |
| Optimization | scipy.optimize (L-BFGS-B, Nelder-Mead for DCC and BEKK MLE) |
| Visualization (research) | matplotlib, seaborn |
| Dashboard frontend | Streamlit, Plotly |
| Analytics report | Microsoft Power BI Desktop |
| Deployment | Streamlit Community Cloud |
| Version control | Git, GitHub |

**How each was used:**
- `statsmodels` - ADF tests, Johansen cointegration, VAR lag selection, VECM estimation
- `arch` - Univariate GARCH/GJR-GARCH fitting, conditional volatility extraction, multi-step forecasting
- `scipy.optimize` - Maximum likelihood estimation for DCC correlation parameters and diagonal BEKK parameters
- `Streamlit + Plotly` - Three-page interactive dashboard with live sidebar controls
- `Power BI` - Two-page report with DAX measures, PairMetrics DATATABLE, and parameter-driven slicers

---

## Project Structure

```text
fx-hedge-ratio-analytics/
|
+-- Data/
|   +-- Original/                                  - Five raw CSV files downloaded from Investing.com and FRED
|   |   +-- USD_INR_Historical_Data.csv
|   |   +-- EUR_INR_Historical_Data.csv
|   |   +-- USD_INR_Futures_Historical_Data.csv
|   |   +-- EUR_INR_Futures_Historical_Data.csv
|   |   +-- VIXCLS.csv
|   |
|   +-- Results/                                   - Twelve cleaned and derived datasets produced by the analysis script
|       +-- prices_clean.csv
|       +-- returns_clean.csv
|       +-- train_returns.csv
|       +-- test_returns.csv
|       +-- ols_results.csv
|       +-- vecm_results.csv
|       +-- dcc_results.csv
|       +-- bekk_results.csv
|       +-- dcc_hedge_ratios_IS.csv
|       +-- dcc_hedge_ratios_OOS.csv
|       +-- volatility_forecasts.csv
|       +-- risk_metrics.csv
|
+-- Notebook/                                       - Complete Python analysis script covering all steps from preprocessing through BEKK-GARCH and risk metrics
|   +-- fx_hedge_analysis.py
|
+-- dashboard.py                                    - Streamlit dashboard (three pages, live advisory tool)
+-- requirements.txt                                - Python dependencies for Streamlit Cloud deployment
+-- README.md
+-- .gitignore
```

---

## Academic References

| Paper | Year | Journal | Contribution to Project |
|---|---|---|---|
| Ederington | 1979 | Journal of Finance | Minimum variance hedge ratio formula, hedging effectiveness metric |
| Bollerslev | 1986 | Journal of Econometrics | GARCH(1,1) model foundation |
| Johansen | 1991 | Econometrica | Cointegration test, VECM specification |
| Glosten, Jagannathan and Runkle | 1993 | Journal of Finance | GJR-GARCH asymmetric volatility |
| Kroner and Sultan | 1993 | JFQA | VECM-GARCH framework for FX futures hedging |
| Engle and Kroner | 1995 | Econometric Theory | BEKK-GARCH multivariate covariance model |
| Engle | 2002 | JBES | DCC - Dynamic Conditional Correlation |
| Jorion | 2006 | McGraw-Hill | VaR, CVaR, CFaR parametric methodology |
| Pandey | 2008 | SSRN/IIMA | Indian market precedent for same comparison |

---

## Learning Outcomes

### Concepts Applied

| Concept | Where Applied | Why It Mattered |
|---|---|---|
| Minimum variance hedging | All four models | Core framework for every hedge ratio estimate |
| Cointegration and error correction | Johansen test, VECM | Spot and futures share a long-run equilibrium - ignoring it misspecifies the model |
| Conditional heteroskedasticity | GARCH, GJR, DCC, BEKK | FX returns cluster in volatility - constant variance assumption is provably wrong |
| Maximum likelihood estimation | DCC and BEKK optimization | Only way to estimate correlation and covariance dynamics jointly |
| Train-test evaluation | All model comparisons | In-sample fit does not guarantee out-of-sample usefulness |
| Parametric VaR and CVaR | Risk metrics layer | Translates model volatility forecasts into actionable risk numbers |

### Key Takeaways

- Simple models can outperform complex ones when market structure limits the variation that dynamic models are designed to exploit - this is not a modeling failure, it is a finding about the market
- Every methodological choice in a quantitative finance project should trace to a named paper, not to intuition or convention
- An interactive dashboard that shows a corporate treasurer their specific VaR and hedge notional is more useful than a research paper showing abstract hedging effectiveness percentages - the last mile of any analytics project is making the output actionable
- Deployment exposes assumptions that local development hides - hardcoded paths, deprecated library methods, and missing requirements files are all real issues that only surface when someone else runs your code
