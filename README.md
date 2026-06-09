# Initial Version Trade Signal Generator
This project is a multi-stage Machine Learning Trade Signal Generator intended to progress from stock price analysis to include expert sentiment analysis, quarterly reports, and market trend analysis. This is the initial version, intended to be expanded upon. Each following version will expand upon features added or changed.

Thus far, this project returns a buy, hold, or sell signal in a strictly 2D feature space, taking into account only stock price. This model takes into account Momentum/Returns and Volatility to relate to model trading decisions. This project trains 2 simple classifiers: Logistic Regression or Random Forest. Additionally, this project visualizes model decisions on a scatter plot.

## Reading Output
- X-axis: momentum (returns)
- Y-axis: volatility
- Color: red (sell), blue (hold), green (buy)
