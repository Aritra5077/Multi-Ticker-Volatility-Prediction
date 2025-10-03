# Multi-Ticker Volatility Prediction
This project focuses on forecasting **financial volatility** across multiple tickers by combining traditional **machine learning models** with **advanced deep learning architectures**. The goal is to leverage both **structured market data** and **unstructured sentiment signals** to improve predictive performance in a multi-ticker **time-series** setting.
## Project Overview
Financial markets are inherently dynamic, with volatility influenced by a mix of price movements, macroeconomic conditions, and market sentiment.
In this project, we:
- Built a supervised time series forecasting pipeline using multiple tickers.
- Incorporated sentiment features (FinBERT and KeyBERT-based) along with price, technical, and derived indicators.
- Applied lagged and rolling features per ticker to capture temporal dependencies.
- Used **Granger Causality Analysis** to identify statistically significant predictive relationships between different features and tickers.
- Benchmarked several models ranging from tree-based methods to deep learning architectures.
- Evaluated models using MAE, RMSE and R² and rank them using a combined ranking strategy.
 ## Dataset
 - **Source**: Historical market data fetched via yfinance and news sentiment data collected from Kaggle and then processed through FinBERT and KeyBERT.
 - **Features**:
 - - Price-based & technical indicators
   - Sentiment scores and their lagged/rolling versions
   - Ticker identifiers (encoded + embedded for deep learning models)
 - **Target**: `Target_Volatility` (volatility values for each ticker over time)

The dataset spans 25 tickers over 8 years, sorted by date and ticker to maintain temporal order.
## Models Implemented
### Machine Learning Baselines
- XGBoost
- LightGBM
- CatBoost
### Deep Learning Architectures
- Simple GRU
- Stacked GRU
- Stacked LSTM
- Stacked GRU with Attention
- Stacked LSTM with Attention
- Multimodal Deep Learning with Attention (Ticker Embeddings + Sentiment + Features)
## Backtesting & Evaluation
All models are trained using a rolling time-based split (Train / Validation / Test) for each ticker to preserve temporal structure.
**Evaluation Metrics**:
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

Models are ranked based on their performance across all three metrics using an average rank score.
| Rank | Model                          |   MAE   |   RMSE  |    R²    | MAE Rank | RMSE Rank | R² Rank |
|:----:|--------------------------------|:-------:|:-------:|:--------:|:--------:|:---------:|:-------:|
|  1   | **Multimodal DL with Attention** | 0.0129 | 0.0212 | 0.9513 | 1 | 1 | 1 |
|  2   | Stacked LSTM with Attention     | 0.0157 | 0.0245 | 0.9346 | 2 | 2 | 2 |
|  3   | Stacked GRU with Attention      | 0.0171 | 0.0256 | 0.9287 | 5 | 3 | 3 |
|  4   | XGBoost                         | 0.0169 | 0.0258 | 0.9177 | 3 | 4 | 6 |
|  5   | Simple GRU                      | 0.0176 | 0.0270 | 0.9209 | 7 | 7 | 4 |
|  6   | LightGBM                        | 0.0169 | 0.0261 | 0.9156 | 4 | 5 | 7 |
|  7   | Stacked GRU                     | 0.0179 | 0.0271 | 0.9199 | 8 | 8 | 5 |
|  8   | CatBoost                        | 0.0171 | 0.0263 | 0.9143 | 6 | 6 | 8 |
|  9   | Stacked LSTM                    | 0.0200 | 0.0283 | 0.9128 | 9 | 9 | 9 |
## Key Insights
- Deep learning architectures with attention mechanisms outperform both classical ML and standard RNN models.
- Incorporating sentiment features provides significant performance gains, especially when combined with price sequences through ticker embeddings.
- Multimodal fusion allows the model to capture both temporal patterns and contextual signals, leading to superior forecasting accuracy.
## Conclusion
This project establishes a solid end-to-end pipeline for volatility forecasting using multimodal data. It demonstrates that combining structured features, sentiment signals and deep learning with attention significantly boosts predictive performance over traditional models.
## Tech Stack
- Python, NumPy, Pandas, Malplotlib, Seaborn, Scikit-learn
- TensorFlow, Keras (Deep Learning)
- XGBoost, LightGBM, CatBoost (ML)
- FinBERT for sentiment extraction
- Joblib for model saving
