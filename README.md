# Store Sales Time Series Forecasting Ensemble

Machine learning solution for the Kaggle competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting), focused on forecasting product-family sales across Corporacion Favorita stores in Ecuador.

The project uses feature engineering, time-series lag features, promotion signals, store metadata, oil prices, holidays, transactions, and ensemble modeling to improve RMSLE performance.

## Competition

- Competition: [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- Task: Predict unit sales for each `store_nbr` and `family` combination
- Forecast horizon: 16 days
- Evaluation metric: RMSLE
- Lower score is better

## Current Kaggle Results

Recent public leaderboard submissions:

| Submission | Public Score |
|---|---:|
| time series 3 - Version 2 | 0.41979 |
| time series 2 - Version 3 | 0.43201 |
| notebook6162f977f5 - Version 36 | 0.47754 |

Current best score: **0.41979**

Target benchmark: **0.37687**

## Approach

The solution combines statistical forecasting and machine learning.

Main techniques:

- Calendar completion for missing store-family-date rows
- Log transformation using `log1p(sales)`
- Safe lag features to avoid future leakage
- Rolling sales statistics
- Promotion momentum features
- Oil price interpolation and oil shock features
- Holiday features using national, regional, and local events
- Store metadata features
- Transaction-based demand signals
- Weighted moving average baseline
- LightGBM model
- CatBoost model
- Ensemble blending and post-processing

## Data Files

The Kaggle dataset contains:

```text
train.csv
test.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
sample_submission.csv
