# Impact-of-Fed-Policy-Shock-on-AMD-Stock-Returns-Structural-Break-Analysis

## 📌 Overview

This project investigates whether the December 18, 2024, Federal Reserve policy announcement caused a structural change in AMD’s return-generating process. It uses a dummy-interacted AR(1) model and Augmented Dickey-Fuller (ADF) unit root tests to detect potential breaks in the mean and volatility of stock returns. Additionally, a deep learning model (multi-layer LSTM) is used to forecast volatility based on historical and macroeconomic data.

---

## 🎯 Objectives

- Analyze the impact of the Fed’s unexpected announcement on AMD’s stock returns.
- Test for structural breaks in return mean and volatility.
- Evaluate stationarity using the Augmented Dickey-Fuller test.
- Forecast future volatility using a multi-layer LSTM model with macroeconomic indicators.

---

## 📈 Dataset

- **Source**: Yahoo Finance
- **Ticker**: AMD (Advanced Micro Devices)
- **Date Range**: January 31, 2023 – January 31, 2025
- **Frequency**: Daily closing prices
- **Event**: December 18, 2024 — Fed delays expected interest rate cuts.

---

## Methodology

### 1. Data Preparation

- Extracted adjusted close prices.
- Computed daily log returns.
- Calculated rolling 20-day standard deviation as a volatility proxy.

### 2. Structural Break Regression

**Mean model**:
r_t = β₀ + β₁·r_{t−1} + γ₀·D_t + γ₁·(r_{t−1}·D_t) + ε_t

**Volatility model**:

r_t² = α₀ + α₁·r_{t−1}² + δ₀·D_t + δ₁·(r_{t−1}²·D_t) + η_t

Where `D_t` is a dummy variable for the post-Fed announcement period.

### 3. ADF Stationarity Tests

- Conducted on full, pre-event, and post-event windows.

### 4. LSTM Forecasting Model

- Inputs:
  - Lagged volatility (`rv5`)
  - VIX index
  - U.S. interest rate spreads
- Model architecture:
  - Multi-layer LSTM with dropout
  - StandardScaler normalization
  - MSE loss, Huber loss
- Tuned using Keras Tuner

## 🧰 Dependencies

Install required packages with:

```bash
pip install pandas numpy matplotlib yfinance tensorflow scikit-learn keras-tuner
```

## 🔍 Results Summary

- No significant structural break in mean returns post-event.
- Slight increase in volatility after December 18, 2024.
- ADF test confirms stationarity throughout.
- LSTM model improves forecasting accuracy when macroeconomic features are added.

## 🙋 Author

**Jinyan Yong**
 Master of Finance – McMaster University