# Multi-Ticker Volatility Prediction
This project focuses on forecasting financial volatility across multiple tickers by combining traditional machine learning models with advanced deep learning architectures. The goal is to leverage both structured market data and unstructured sentiment signals to improve predictive performance in a multi-ticker time series setting.
## Project Overview
Financial markets are inherently dynamic, with volatility influenced by a mix of price movements, macroeconomic conditions, and market sentiment.
In this project, we:
- Built a supervised time series forecasting pipeline using multiple tickers.
- Incorporated sentiment features (FinBERT and KeyBERT-based) along with price, technical, and derived indicators.
- Applied lagged and rolling features per ticker to capture temporal dependencies.
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
