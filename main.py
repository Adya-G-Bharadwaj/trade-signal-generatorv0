import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#Loading pricing data
def load_price_data(n):
    np.random.seed(42)
    price = np.cumsum(np.random.randn(n)) + 100
    return pd.DataFrame({'price': price})

#Extra features needed: momentum, rolling volatility
def create_features(df, window = 10):
    df['returns'] = df['price'].pct_change().fillna(0)
    df['volatility'] = df['returns'].rolling(window).std().fillna(0)
    return df

#Generate buy sell hold labels
def generate_labels(df):
    conds = [df['returns'] > 0.01, df['returns'] < 0.01]
    choices = [2,0] #0=sell, 1=hold, 2=buy
    df['signal'] = np.select(conds, choices, default = 1)
    return df

#Train model
MODEL_TYPE = 'LogisticRegression' #'RandomForest'
def train_model(X,y,model_type= MODEL_TYPE):
    if model_type == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000)
    elif model_type == 'RandomForest':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        raise ValueError(
            "must be RandForest or LogisticReg"
        )
    model.fit(X,y)
    return model

#Visualization
COLORS = {0:'red', 1:'blue', 2:'green'}
LABELS = {0:'Sell', 1:'Hold', 2:'Buy'}
def plot_signal_map(X,y_pred):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10,7))
    for label in np.unique(y_pred):
        idx = y_pred == label
        ax.scatter(X[idx,0], X[idx,1], c=COLORS[label], label=LABELS[label], alpha=0.6, s=60)
    ax.set_xlabel("Returns")
    ax.set_ylabel("Volatility")
    ax.set_title("Predicted Trading Signals")
    ax.legend()
    plt.show()

#Main method: Load and process data, standardize features, train/test split, train model, predict on all data, plot
if __name__ == '__main__':
    df = load_price_data(1000)
    df = create_features(df)
    df = generate_labels(df)
    X = df[["returns", "volatility"]].values
    y = df["signal"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #train test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    model = train_model(X_train, y_train, model_type=MODEL_TYPE)
    y_pred = model.predict(X_scaled)

    #plot and visualize
    plot_signal_map(X_scaled, y_pred)
