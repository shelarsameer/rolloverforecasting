import streamlit as st
import numpy as np
import pandas as pd
from arch import arch_model
import matplotlib.pyplot as plt

# Gathering Data
data = pd.read_csv('tatamotors.csv')
comp = data['Close Price'].values
DATE = data['Date'].values
Return = np.log(comp[1:] / comp[:-1])

# Function to fit model and make predictions
def fit_predict(returns, p, q, o=0, vol='GARCH', power=2.0):
    pred = np.zeros(len(returns) - 150)
    for i in range(len(pred)):
        if vol == 'TGARCH':
            model = arch_model(returns[i:i+151], vol='GARCH', p=p, q=q, o=o, power=1.0, mean='AR', dist='normal')
        else:
            model = arch_model(returns[i:i+151], vol=vol, p=p, q=q, o=o, power=power, mean='AR', dist='normal')
        res = model.fit(disp='off', show_warning=False)
        forecast = res.forecast(horizon=1)
        pred[i] = forecast.mean['h.1'].iloc[-1]
    return pred, model, res

# Function to predict next 10 days
def predict_next_10_days(returns, p, q, o=0, vol='GARCH', power=2.0):
    predictions = []
    current_returns = returns[-151:].copy()
    
    for _ in range(10):
        if vol == 'TGARCH':
            model = arch_model(current_returns, vol='GARCH', p=p, q=q, o=o, power=1.0, mean='AR', dist='normal')
        else:
            model = arch_model(current_returns, vol=vol, p=p, q=q, o=o, power=power, mean='AR', dist='normal')
        res = model.fit(disp='off', show_warning=False)
        forecast = res.forecast(horizon=1)
        next_return = forecast.mean.values[-1, -1]
        predictions.append(next_return)
        current_returns = np.append(current_returns[1:], next_return)
    
    return predictions

# Streamlit app
st.title('GARCH Models for Stock Price Prediction')

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.write(data.head())

    comp = data['Close Price'].values
    DATE = data['Date'].values
    Return = np.log(comp[1:] / comp[:-1])

    # Fit models and calculate RMSE
    garch_pred, garch_model, garch_res = fit_predict(Return, p=1, q=1)
    egarch_pred, egarch_model, egarch_res = fit_predict(Return, p=1, q=1, vol='EGARCH')
    tgarch_pred, tgarch_model, tgarch_res = fit_predict(Return, p=1, q=1, o=1, vol='TGARCH')

    actual = Return[150:]
    rmse_garch = np.sqrt(np.mean((actual - garch_pred)**2))
    rmse_egarch = np.sqrt(np.mean((actual - egarch_pred)**2))
    rmse_tgarch = np.sqrt(np.mean((actual - tgarch_pred)**2))

    st.write("RMSE Results:")
    st.write(f"GARCH(1,1) RMSE: {rmse_garch:.6f}")
    st.write(f"EGARCH(1,1) RMSE: {rmse_egarch:.6f}")
    st.write(f"TGARCH(1,1) RMSE: {rmse_tgarch:.6f}")

    # Determine best model
    best_rmse = min(rmse_garch, rmse_egarch, rmse_tgarch)
    if best_rmse == rmse_garch:
        best_model, best_res = garch_model, garch_res
        best_name = "GARCH(1,1)"
    elif best_rmse == rmse_egarch:
        best_model, best_res = egarch_model, egarch_res
        best_name = "EGARCH(1,1)"
    else:
        best_model, best_res = tgarch_model, tgarch_res
        best_name = "TGARCH(1,1)"

    st.write(f"\nBest model: {best_name}")

    # Predict next 10 days
    last_price = comp[-1]
    next_10_returns = predict_next_10_days(Return, p=1, q=1, vol=best_name.split('(')[0])
    next_10_prices = [last_price * np.exp(np.sum(next_10_returns[:i+1])) for i in range(10)]

    st.write(f"\nLast closing price: {last_price:.2f}")
    st.write("Predicted closing prices for next 10 days:")
    for i, price in enumerate(next_10_prices, 1):
        st.write(f"Day {i}: {price:.2f}")

    # Calculate percentage change
    percent_changes = [(price - last_price) / last_price * 100 for price in next_10_prices]

    # Plot predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(range(11), [last_price] + next_10_prices, marker='o')
    ax.set_title(f"10-Day Price Prediction using {best_name}")
    ax.set_xlabel("Day")
    ax.set_ylabel("Predicted Price")
    ax.set_xticks(range(11))
    ax.set_xticklabels(['Current'] + [f'Day {i}' for i in range(1, 11)])
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Display percentage changes
    st.write("\nPredicted percentage changes:")
    for i, change in enumerate(percent_changes, 1):
        st.write(f"Day {i}: {change:.2f}%")

else:
    st.write("Please upload a CSV file to begin analysis.")
