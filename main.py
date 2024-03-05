from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import StreamingResponse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, GRU

# Preprocessing data lib
import requests
from bs4 import BeautifulSoup
from io import StringIO
import datetime

app = FastAPI()

origins = [
    "http://localhost:3000",
    "https://exchangerateprediction.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Declare variable
scaler = MinMaxScaler(feature_range=(0, 1))

# PAGES=============================================
@app.get("/")
async def root():
    return { "message": "OK"}

@app.get("/data")
def handleProcessingData(currency: str):    
    df = getData(currency)
    # Plot chart
    plot_path = "./chart.png"
    plot_and_save(df["Close"], plot_path, currency)
    return df

def plot_and_save(data, plot_path, currency):
    plt.figure(figsize=(10, 5))
    plt.plot(data, label="Close")
    plt.title(f'Close price of {currency} from 2020-01-01')
    plt.legend()
    plt.savefig(plot_path)
    plt.close()

@app.get("/chart")
def get_plot_image():
    plot_path = "./chart.png"
    return StreamingResponse(open(plot_path, "rb"), media_type="image/png")

@app.get("/predict")
def make_predict(model: str, days: int, currency: str):
    data = getData(currency)

    if data is None:
        return "No data"
    
    if model == "Linear Regression":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        linear = LinearRegression()
        linear.fit(data_X_scaled, data_y_scaled)
        return predictRateInTheFuture("Linear Regression", linear, data, future_days=days)

    elif model == "Decision Tree Regression":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        decisionTree = DecisionTreeRegressor()
        decisionTree.fit(data_X_scaled, data_y_scaled)
        return predictRateInTheFuture("Decision Tree Regression", decisionTree, data, future_days=days)
    elif model == "Random Forest Regression":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        randomForest =RandomForestRegressor()
        randomForest.fit(data_X_scaled, data_y_scaled)
        return predictRateInTheFuture("Random Forest Regression", randomForest, data, future_days=days)
    elif model == "XGBoost":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        XGBoost = XGBRegressor()
        XGBoost.fit(data_X_scaled, data_y_scaled)
        return predictRateInTheFuture("XGBoost Regression", XGBoost, data, future_days=days)
    elif model == "LSTM":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        X_real, y_real = createSequentialData(data_y_scaled)
        model_name = "LSTM"
        lstm = Sequential()
        lstm.add(LSTM(50, return_sequences=True, input_shape=(X_real.shape[1], 1)))
        lstm.add(LSTM(50))
        lstm.add(Dense(1))
        lstm.compile(optimizer='adam', loss='mean_squared_error')
        lstm.fit(X_real, y_real, epochs=25, batch_size=64)
        return predictDLInTheFuture(model_name, lstm, data, future_days=days)
    elif model == "GRU":
        data_X_scaled, data_y_scaled = preprocess_data(data)
        X_real, y_real = createSequentialData(data_y_scaled)
        model_name = "GRU"
        gru = Sequential()
        gru.add(GRU(50, return_sequences=True, input_shape=(X_real.shape[1], 1)))
        gru.add(GRU(50))
        gru.add(Dense(1))
        gru.compile(optimizer='adam', loss='mean_squared_error')
        gru.fit(X_real, y_real, epochs=25, batch_size=64)
        return predictDLInTheFuture(model_name, gru, data, future_days=days)
    return "Model Not found"

# Function definition ===============================
# Get current date
def getCurrentDate():
  now = datetime.datetime.now()
  date_only = datetime.datetime(now.year, now.month, now.day)
  timestamp = int(date_only.timestamp())
  return timestamp

def predictRateInTheFuture(model_name, model,data, lookback = 30, future_days = 1):
    data_X_scaled, data_y_scaled = preprocess_data(data)
    last_input_data = data_X_scaled[-lookback:]
    now = datetime.datetime.now()
    print(data_X_scaled.shape)
    predicted_values = []
    forecasting_dates = []

    for day in range(future_days):
        next_date = now + datetime.timedelta(days=day+1)
        forecasting_dates.append(next_date.strftime('%Y-%m-%d'))

        prediction = model.predict(last_input_data)
        prediction = scaler.inverse_transform(prediction.reshape(-1,1))
        # print(prediction)
        predicted_values.append(round(float(prediction.flatten()[0]),2))
        last_input_data = np.roll(last_input_data, -1, axis=0)
    print(predicted_values)
    predictions = pd.DataFrame(list(zip(forecasting_dates, predicted_values)), columns=['Date','Predicted'])
    print("Predict exchange rate with",model_name, f'in {future_days} days')
    print(predictions)
    return predictions

def predictDLInTheFuture(model_name, model, data, future_days = 1):
    data_X_scaled, data_y_scaled = preprocess_data(data)
    X_real, y_real = createSequentialData(data_y_scaled)

    last_input_data = X_real[-1:]
    now = datetime.datetime.now()
    # Arrays to store predicted values and dates
    predicted_values = []
    forecasting_dates = []

    for day in range(future_days):
        next_date = now + datetime.timedelta(days=day+1)
        forecasting_dates.append(next_date.strftime('%Y-%m-%d'))

        predicted_price_scaled = model.predict(last_input_data)

        predicted_price = scaler.inverse_transform(predicted_price_scaled.reshape(-1, 1))
        predicted_values.append(round(float(predicted_price[0, 0]),2))
        last_input_data = np.roll(last_input_data, -1, axis=0)
        predicted_price_scaled = predicted_price_scaled[:, :, np.newaxis]
        last_input_data = np.concatenate([last_input_data[:, 1:, :], predicted_price_scaled], axis=1)

    predictions = pd.DataFrame(list(zip(forecasting_dates, predicted_values)), columns=['Date','Predict Close'])

    print("Predict exchange rate with",model_name, f'in {future_days} days')
    print(predictions)
    return predictions

def preprocess_data(data):
    features = ['Open', 'High', 'Low']
    target = ['Close']

    data_X_scaled = scaler.fit_transform(data[features])
    data_y_scaled = scaler.fit_transform(data[target])

    return data_X_scaled, data_y_scaled

def createSequentialData(data, window_size=14):
    X,y = [],[]
    for i in range(len(data)-window_size):
        X.append(data[i:(i+window_size)])
        y.append(data[(i+window_size)])
    return np.array(X),np.array(y)

def getData(currency):
    currency_rates = ['USDVND', 'EURVND','GBPVND','USDAUD','USDKRW','USDCNY']
    index = currency_rates.index(currency)
    period1 = 1577836800 #2020-01-01
    period2 = getCurrentDate() # now

    url = f'https://query1.finance.yahoo.com/v7/finance/download/{currency_rates[index]}=X?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    raw_data = ''
    # Send a GET request to the URL with headers
    response = requests.get(url, headers=headers)
    if response.status_code == 200:  # If the request was successful
        raw_data = response.text
    else:
        print("Failed to retrieve data. Status code:", response.status_code)
    # Đọc dữ liệu từ chuỗi text
    df = pd.read_csv(StringIO(raw_data))
    df = df.drop(['Volume', 'Adj Close'], axis=1)
    df.dropna(inplace=True)
    return df