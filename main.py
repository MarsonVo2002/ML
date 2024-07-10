import multiprocessing
import multiprocessing.process
from threading import Thread
from numpy import float64, int64
import pandas as pd
import asyncio
import websockets
import json
import time
import nest_asyncio
from binance.client import Client
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
nest_asyncio.apply()

API_KEY = '2SnAaj4wteV16uWiBaSkExvFWGU7M9xEml2ANA34lySiOtxAWzVu1ko4l9nAPR0H'
API_SECRET = 'YhCo1SBzGOYQC3ZvkgD72L2FPEEQ6OLBhQLnYjsfdkvztsFkj9puWJL2VdJDEent'
client = Client(API_KEY, API_SECRET)
app = dash.Dash()
global train, valid
train = pd.DataFrame()  
valid = pd.DataFrame() 
# Layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='Real-time BTC-USD Price Dashboard'),
    dcc.Graph(id='price-graph'),
    dcc.Interval(id='interval-component', interval=40000, n_intervals=0) 
])
# Callback to update the graph with predictions
@app.callback(Output('price-graph', 'figure'), [Input('interval-component', 'n_intervals')])
def update_graph(n):
    if(len(historical_data) == 1000):
        print("No update")
    else:
        print(f"Updated: {len(historical_data)}")
    train, valid = load_and_predict(historical_data)
    # Actual Closing Price Graph
    trace1 = go.Scatter(
        x=train.index,
        y=train['Close'],
        mode='lines',
        name='Train'
    )

    trace2 = go.Scatter(
        x=valid.index,
        y=valid['Close'],
        mode='lines',
        name='Actual'
    )

    trace3 = go.Scatter(
        x=valid.index,
        y=valid['Predictions'],
        mode='lines',
        name='Predictions'
    )

    return {
        'data': [trace1, trace2, trace3],
        'layout': go.Layout(
            title='BTC-USD Actual vs Predicted Prices',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Price'}
        )
    }
def load_and_predict(df):
    df["timestamp"] = pd.to_datetime(df.timestamp, format="%Y-%m-%d")
    df.index = df['timestamp']
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Timestamp', 'Close'])
    
    for i in range(0, len(data)):
        new_data["Timestamp"][i] = data['timestamp'][i]
        new_data["Close"][i] = data["close"][i]
    
    new_data.index = new_data.Timestamp
    new_data.drop("Timestamp", axis=1, inplace=True)
    dataset = new_data.values
    
    train = dataset[0:int(len(dataset)*0.8), :]
    valid = dataset[int(len(dataset)*0.8):, :]
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    x_train, y_train = [], []
    
    for i in range(60, len(train)):
        x_train.append(scaled_data[i-60:i, 0])
        y_train.append(scaled_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    
    lstm_model=Sequential()
    lstm_model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    lstm_model.add(LSTM(units=50))
    lstm_model.add(Dense(1))
    
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    lstm_model.compile(loss='mean_squared_error',optimizer='adam')
    lstm_model.fit(x_train,y_train,epochs=1,batch_size=1,verbose = 1)
    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = lstm_model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    
    train = new_data[:int(len(dataset)*0.8)]
    valid = new_data[int(len(dataset)*0.8):]
    valid['Predictions'] = closing_price
    
    return train, valid
# Function to fetch historical candle data
def fetch_historical_candles(symbol, interval, limit=1000):
    """
    Fetch historical candle data from Binance
    :param symbol: Symbol to fetch data for (e.g., 'BTCUSDT')
    :param interval: Candle interval (e.g., '1m', '5m', '1h', etc.)
    :param limit: Number of candles to fetch
    :return: DataFrame containing candle data
    """
    candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    # df.set_index('timestamp', inplace=True)
    return df

#Fetch the BTCUSDT 1-minute candles
symbol = 'BTCUSDT'
interval = '1m'
historical_data = fetch_historical_candles(symbol, interval)

# Function to handle incoming WebSocket messages
async def handle_message(websocket):
    async for message in websocket:
        data = json.loads(message)
        # print(f"Received message: {data}")
        kline = data['k']
        new_candle = {
            'timestamp': pd.to_datetime(kline['t'], unit='ms'),
            'open': float64(kline['o']),
            'high': float64(kline['h']),
            'low': float64(kline['l']),
            'close': float64(kline['c']),
            'volume': float64(kline['v']),
            'close_time': pd.to_datetime(kline['T'], unit='ms'),
            'quote_asset_volume': float64(kline['q']),
            'number_of_trades': int64(kline['n']),
            'taker_buy_base_asset_volume': float64(kline['V']),
            'taker_buy_quote_asset_volume': float64(kline['Q']),
            'ignore': float64(kline['B'])
        }
        global historical_data
        historical_data.loc[len(historical_data)] = new_candle 
        print("Add new candle")
        time.sleep(15)

        # train, valid = load_and_predict(historical_data)
        # graph(train_nse=train, valid_nse=valid)
        # Placeholder for prediction logic
        # Do the Prediction here
    
# Function to connect to Binance WebSocket
async def connect_to_websocket(symbol):
    websocket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
    async with websockets.connect(websocket_url) as websocket:
        await handle_message(websocket)
def fetch_data():
    asyncio.get_event_loop().run_until_complete(connect_to_websocket(symbol))
if __name__ == '__main__':
    
    th = Thread(target=fetch_data)
    th.start()
    app.run_server(debug=True)
    th.join()