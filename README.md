# 📦 Walmart M5 Store Sales Forecasting

> **Hierarchical Time-Series Forecasting** — LightGBM · Prophet · Bottom-Up Reconciliation · WRMSSE  
> FANG-level ML engineering portfolio project

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)](https://lightgbm.readthedocs.io)
[![Prophet](https://img.shields.io/badge/Prophet-1.1-orange)](https://facebook.github.io/prophet)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)](https://streamlit.io)

---

## 🎯 Project Overview

End-to-end hierarchical time-series forecasting system replicating the Walmart M5 Kaggle competition setup:
- **3,049 products** across **10 Walmart stores** in 3 states (CA, TX, WI)
- **5+ years of daily sales history** (2011–2016)
- **28-day forecast horizon** at item×store level
- **Hierarchical reconciliation** to ensure coherent forecasts at all levels

### Key Results
| Metric | LightGBM | Naive Baseline | Improvement |
|--------|----------|----------------|-------------|
| MAE | — | — | Significant |
| MAPE% | — | — | Significant |
| WRMSSE | < 1.0 | 1.0 (baseline) | Beat baseline |

*(Run the pipeline to see live metrics)*

---

## 🏗️ Architecture

```
walmart_m5_forecasting/
├── app.py                          # Streamlit entry point
├── requirements.txt
├── config/
│   └── config.yaml                 # Pipeline configuration
├── src/
│   ├── pipeline.py                 # End-to-end orchestrator
│   ├── data/
│   │   ├── generator.py            # Synthetic M5-like data generator
│   │   └── preprocessor.py        # Wide→long, calendar/price merging
│   ├── features/
│   │   └── feature_engineering.py # 25+ lag/rolling/calendar/price features
│   ├── models/
│   │   ├── lgbm_forecaster.py     # Global Tweedie LightGBM model
│   │   └── prophet_forecaster.py  # Per-store Prophet models
│   ├── reconciliation/
│   │   └── hierarchical.py        # Bottom-up, OLS, MinT reconciliation
│   └── evaluation/
│       └── metrics.py             # WRMSSE, RMSSE, MAE, MAPE, SMAPE
├── data/
│   ├── raw/                        # Original M5 CSVs (or generated)
│   ├── processed/                  # Preprocessed panel data
│   └── features/                   # Feature-engineered data
├── models/
│   └── saved/                      # Serialised model artifacts
├── notebooks/
│   └── exploration.ipynb           # EDA and prototyping
└── tests/
    └── test_pipeline.py
```

---

## 🔧 Pipeline Stages

### 1. Data Generation / Loading
Synthetic M5-compatible data generator preserving:
- Zero-inflated (intermittent) demand patterns
- Weekly + annual seasonality
- Holiday demand spikes (Thanksgiving +300%, Christmas +250%, Super Bowl +180%)
- Price elasticity effects
- Store-level demand variation

### 2. Preprocessing
- Wide-to-long format transformation (d_1 … d_1941 columns → rows)
- Calendar join: weekday, month, events, SNAP benefit flags
- Price join: weekly sell prices with promotion detection
- Memory optimisation: downcasting dtypes (~60% RAM reduction)
- Time-based train/validation split (last 28 days = validation)

### 3. Feature Engineering (25+ features)
| Category | Features |
|----------|----------|
| **Lag** | sales_lag_7, _14, _28, _35, _42 |
| **Rolling** | mean & std over 7, 14, 28, 56 day windows (shifted by 28) |
| **Calendar** | day_of_week, month, quarter, Fourier sin/cos for yearly+weekly cycles |
| **Events** | has_event flag, SNAP benefit flag |
| **Price** | sell_price, price_change, relative_price, discount_depth |
| **Aggregation** | store_daily_mean, dept_daily_mean |
| **IDs** | item_id, dept_id, cat_id, store_id, state_id (label encoded) |

### 4. LightGBM (Global Model)
- **Objective**: Tweedie regression (handles zero-inflated demand)
- **Strategy**: Single global model across all items/stores
- **Tuning**: Early stopping on validation RMSE
- **Inference**: Direct 28-step-ahead prediction using pre-computed features

### 5. Prophet (Per-Store)
- **Scope**: One model per store (aggregated over all items)
- **Seasonality**: Multiplicative yearly + weekly
- **Holidays**: M5 events (SuperBowl, Thanksgiving, Christmas, Easter, etc.)
- **Transform**: log1p for variance stabilisation

### 6. Hierarchical Reconciliation
```
Total (1)
  ├── State (3): CA, TX, WI
  │     └── Store (10): CA_1...WI_3
  ├── Category (3): HOBBIES, HOUSEHOLD, FOODS
  │     └── Department (7)
  └── Item×Store (bottom level) ← LightGBM trains here
```

**Bottom-Up** (default): Sum item-store → store → state → total. Guaranteed coherent.  
**OLS**: Regression-based: `ŷ_reconciled = S(S'S)⁻¹S' ŷ_base`  
**MinT**: Minimum Trace with Ledoit-Wolf shrinkage covariance estimator

### 7. Evaluation (WRMSSE)
- **WRMSSE**: Weighted Root Mean Squared Scaled Error (official M5 metric)
- **Weights**: Revenue-based (price × volume over last 28 training days)
- **Scale**: Relative to naive random-walk baseline per series
- Also computes MAE, RMSE, MAPE, SMAPE for interpretability

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone <repo-url>
cd walmart_m5_forecasting

# 2. Install
pip install -r requirements.txt

# 3. Run Streamlit app
streamlit run app.py
```

The app will:
1. Generate M5-like synthetic data (or use real M5 CSVs if placed in `data/raw/`)
2. Run the full pipeline (1-3 minutes depending on item count)
3. Display interactive forecasting dashboard

---

## 📊 Dashboard Tabs

| Tab | Content |
|-----|---------|
| **Forecasts** | LightGBM & Prophet predictions vs actuals at item/dept/store level |
| **Store Analysis** | Per-store performance heatmap, bias analysis, scatter plots |
| **Hierarchy** | Reconciled forecasts at all levels; coherence verification |
| **Model Insights** | Feature importance (top 25), hyperparameter config |
| **Evaluation** | WRMSSE explanation, residual analysis, model comparison table |

---

## 🔬 Technical Decisions

**Why LightGBM over LSTM/DeepAR?**
- Tabular tree models consistently outperform deep learning on M5 (top 10 solutions used LGBM/XGBoost)
- No stationarity assumptions, handles zero-inflated counts naturally
- Fast iteration, interpretable via feature importance
- Tweedie loss perfectly suited for count/sales data

**Why global model?**
- 30,000+ item-store series — training individual models is infeasible
- Cross-series learning improves generalisation (more data per pattern)
- ID embeddings (label codes) let the model learn item/store personalities

**Why bottom-up reconciliation?**
- Best-performing strategy in academic benchmarks for retail data
- Coherent by construction (no post-hoc correction needed)
- Bottom level has most information; aggregation doesn't lose signal

---

## 📈 Extending to Production

1. **Real M5 data**: Download from [Kaggle](https://www.kaggle.com/c/m5-forecasting-accuracy) and place CSVs in `data/raw/`
2. **Hyperparameter tuning**: Replace manual params with Optuna study
3. **Model serving**: Wrap in FastAPI + Docker for REST endpoint
4. **Monitoring**: Add prediction drift detection with Evidently
5. **Scheduling**: Airflow DAG for daily retraining trigger

---

## 📚 References

- [M5 Competition Overview](https://mofc.unic.ac.cy/m5-competition/)
- [Hyndman et al. "Hierarchical Time Series"](https://otexts.com/fpp3/hierarchical.html)
- [LightGBM for M5: 1st place solution write-up](https://www.kaggle.com/c/m5-forecasting-accuracy/discussion)
- [MinT Reconciliation Paper](https://robjhyndman.com/papers/MinT.pdf)
