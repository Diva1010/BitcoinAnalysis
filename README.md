# Bitcoin Analysis and Price Prediction

A machine learning research project exploring short-term Bitcoin price prediction using historical trading data. This project compares multiple regression and deep learning models, investigates the impact of technical indicators, and evaluates predictions across different time horizons.

## Overview

Bitcoin's high price volatility makes short-term price prediction a challenging time-series problem. This project investigates whether machine learning models can learn patterns from historical Bitcoin market data and predict future closing prices.

Four models were implemented and evaluated:

- Linear Regression
- Random Forest Regressor
- XGBoost
- Long Short-Term Memory (LSTM) neural network

After comparing their performance, the best-performing model was investigated further by incorporating technical indicators. The project also explored forecasting Bitcoin's closing price 10, 30, and 60 minutes ahead.

## Objectives

- Analyse historical Bitcoin trading data at one-minute intervals.
- Compare different machine learning approaches for price prediction.
- Evaluate model performance using Root Mean Squared Error (RMSE) and Mean Absolute Error (MAE).
- Investigate whether technical indicators improve prediction accuracy.
- Explore short-term forecasting at multiple future time horizons.
- Assess the limitations of machine learning for volatile cryptocurrency markets.

## Dataset

The research used historical Bitcoin trading data sourced from Kaggle, originally obtained from the Bitfinex exchange.

| Attribute | Details |
|---|---|
| Asset | Bitcoin (BTC) |
| Quote currency | US dollars (USD) |
| Data frequency | One-minute intervals |
| Historical period | 1 April 2013 – 1 September 2019 |
| Original dataset size | Approximately 2.4 million records |
| Primary prediction target | Closing price |

### Input features

The original dataset contained the following market features:

- **Open:** Opening price for the interval.
- **High:** Highest price recorded during the interval.
- **Low:** Lowest price recorded during the interval.
- **Close:** Closing price for the interval.
- **Volume:** Bitcoin trading volume.

Data preprocessing included handling missing and non-finite values, aligning the time series to one-minute intervals, and preparing features for model training.

## Technical Stack

- **Language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn, XGBoost
- **Deep learning:** TensorFlow, Keras
- **Evaluation:** RMSE, MAE, R² and adjusted R²
- **Environment:** Jupyter Notebook
- **Data source:** Kaggle / Bitfinex historical trading data

## Methodology

### 1. Data preprocessing

Historical trading data was prepared for model training and evaluation. The preprocessing workflow included handling missing timestamps, managing missing OHLCV values, and preparing numerical features.

For the technical-indicator experiments, the dataset was also filtered to a smaller historical window to reduce model training time.

### 2. Model development

Four approaches were implemented and compared.

| Model | Approach |
|---|---|
| Linear Regression | Establishes a simple regression baseline. |
| Random Forest | Uses an ensemble of decision trees to model nonlinear relationships. |
| XGBoost | Applies gradient boosting to learn patterns in the market features. |
| LSTM | Uses a recurrent neural network architecture designed for sequential data. |

The models were evaluated against actual closing prices using RMSE and MAE. Lower values indicate smaller prediction errors on the evaluated dataset.

### 3. Model evaluation

The initial model comparison identified Random Forest as the best-performing model among the four approaches tested.

The model was then investigated further to assess whether additional technical indicators could improve its performance.

### 4. Technical indicator analysis

The following indicators were calculated and evaluated:

- **Relative Strength Index (RSI):** A momentum indicator based on recent price movements.
- **Stochastic RSI:** Measures RSI relative to its recent range.
- **Moving Average Convergence Divergence (MACD):** Captures relationships between moving averages.
- **Bollinger Bands (%B and bandwidth):** Describe the closing price's position relative to the bands and the bands' relative width.
- **Williams %R:** Measures the closing price relative to its recent high-low range.

Feature importance, prediction errors, and regression statistics were examined to understand whether these additional features contributed meaningful predictive value.

### 5. Future price prediction

The final stage explored predictions at three horizons:

- 10 minutes ahead
- 30 minutes ahead
- 60 minutes ahead

The experiments used a shifted closing-price target to investigate future price prediction.

## Results

### Initial model comparison

The following results were reported in the original research for the initial model evaluation.

| Model | RMSE | MAE |
|---|---:|---:|
| **Random Forest** | **1.77** | **0.70** |
| Linear Regression | 2.50 | 1.21 |
| XGBoost | 4.93 | 3.18 |
| LSTM | 3.99 | 2.80 |

*Values are rounded from the report. Errors are expressed in the dataset's price units, USD.*

### Key findings

**Random Forest performed best in the initial comparison.**

It achieved the lowest RMSE and MAE among the four evaluated models, making it the strongest candidate for further investigation within this experiment.

**Adding technical indicators did not improve the reported results.**

| Experiment | RMSE | MAE |
|---|---:|---:|
| Base model | 2.32 | 0.96 |
| Model with technical indicators | 2.57 | 1.12 |

These figures come from the separate technical-indicator experiment and should not be directly compared with the initial model-comparison table, which used a different experimental setup.

The additional indicators did not improve the measured prediction accuracy in this experiment. The analysis also identified the original OHLC price features, particularly high and low, as important inputs to the Random Forest model.

**Prediction quality declined at longer forecast horizons.**

The 10-minute-ahead experiment produced more promising results than the longer-horizon forecasts. The 30-minute and 60-minute experiments highlighted the difficulty of extending short-term predictions further into the future.

## Limitations

This project is an academic research experiment, not a production forecasting system or a trading strategy.

Important limitations include:

- **Historical data:** The dataset covers April 2013 through September 2019 and does not represent current market conditions.
- **Market volatility:** Sudden price changes and external events may not be captured by historical price features.
- **Limited benefit from technical indicators:** The selected indicators did not improve the measured results in the reported experiment.
- **Forecast horizon:** Results became less useful as the prediction horizon increased.
- **Evaluation scope:** The reported metrics describe the specific experimental setup and should not be interpreted as guarantees of future performance.
- **Trading applicability:** The project does not establish profitability after fees, slippage, or other real-world trading constraints.

Predictions should not be treated as financial advice.

## Future Improvements

Potential extensions include:

- Experimenting with additional market and technical features.
- Investigating market sentiment from sources such as news and public social media data.
- Evaluating alternative time-series forecasting approaches.
- Using a more rigorous chronological validation strategy and testing across different market regimes.
- Comparing forecast errors across multiple horizons using consistent evaluation windows.
- Investigating the effects of transaction costs and baseline trading strategies, if extending the work toward trading research.

## Project Background

This project was developed as an individual research project at Middlesex University during the 2018–2019 academic year.

It explored the application of machine learning to financial time-series prediction, with an emphasis on model comparison, feature evaluation, and the challenges of forecasting volatile cryptocurrency prices.

## Disclaimer

This project is for educational and research purposes only. It does not provide financial advice, investment recommendations, or guarantees of future Bitcoin prices.

---

**Author:** Divya Kaushik  
**Repository:** [BitcoinAnalysis](https://github.com/Diva1010/BitcoinAnalysis)
