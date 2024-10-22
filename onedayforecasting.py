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

# Function to predict next day
def predict_next_day(returns, p, q, o=0, vol='GARCH', power=2.0):
    if vol == 'TGARCH':
        model = arch_model(returns[-151:], vol='GARCH', p=p, q=q, o=o, power=1.0, mean='AR', dist='normal')
    else:
        model = arch_model(returns[-151:], vol=vol, p=p, q=q, o=o, power=power, mean='AR', dist='normal')
    res = model.fit(disp='off', show_warning=False)
    forecast = res.forecast(horizon=1)
    next_return = forecast.mean.values[-1, -1]
    return next_return

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

    # Predict next day
    last_price = comp[-1]
    next_return = predict_next_day(Return, p=1, q=1, vol=best_name.split('(')[0])
    next_day_price = last_price * np.exp(next_return)

    st.write(f"\nLast closing price: {last_price:.2f}")
    st.write(f"Predicted closing price for next day: {next_day_price:.2f}")

    # Calculate percentage change
    percent_change = (next_day_price - last_price) / last_price * 100

    # Plot prediction
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['Current', 'Predicted'], [last_price, next_day_price], color=['blue', 'green'])
    ax.set_title(f"Next Day Price Prediction using {best_name}")
    ax.set_ylabel("Price")
    for i, price in enumerate([last_price, next_day_price]):
        ax.text(i, price, f'{price:.2f}', ha='center', va='bottom')
    plt.tight_layout()
    st.pyplot(fig)

    # Display percentage change
    st.write(f"Predicted percentage change: {percent_change:.2f}%")

else:
    st.write("Please upload a CSV file to begin analysis.")
