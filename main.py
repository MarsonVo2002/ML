import os
import multiprocessing.process
import sys
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
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Conv1D, MaxPooling1D, Flatten, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np

nest_asyncio.apply()

API_KEY = '2SnAaj4wteV16uWiBaSkExvFWGU7M9xEml2ANA34lySiOtxAWzVu1ko4l9nAPR0H'
API_SECRET = 'YhCo1SBzGOYQC3ZvkgD72L2FPEEQ6OLBhQLnYjsfdkvztsFkj9puWJL2VdJDEent'
client = Client(API_KEY, API_SECRET)
app = dash.Dash()

# Layout of the dashboard
app.layout = html.Div(children=[
    html.H1(children='Real-time BTC-USD Price Dashboard', style={"textAlign": "center"}),
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'LSTM', 'value': 'lstm'},
            {'label': 'RNN', 'value': 'rnn'},
            {'label': 'GRU', 'value': 'gru'},
            {'label': 'CNN', 'value': 'cnn'}
        ],
        value='lstm',
        style={'width': '50%',"margin": "0 auto"}
    ),
    dcc.Graph(id='price-graph'),
    dcc.Graph(id='roc-graph'),
    dcc.Interval(id='interval-component', interval=70000, n_intervals=0),
    html.Button('Stop Program', id='stop-button', n_clicks=0)
])

# Callback to update the graph with predictions
@app.callback(
    Output('price-graph', 'figure'),
    Output('roc-graph', 'figure'),
    [Input('interval-component', 'n_intervals'), Input('model-dropdown', 'value')]
)
def update_graph(n, selected_model):
    if len(historical_data) == 1000:
        print("No update")
    else:
        print(f"Updated: {len(historical_data)}")
    
    train, valid = load_and_predict(historical_data, selected_model, True)
    train_roc, valid_roc = load_and_predict(historical_data, selected_model, False)
    data = []
    data1 = []
    trace1 = go.Scatter(
        x=train.index,
        y=train['Close'],
        mode='lines',
        name='Train'
    )
    data.append(trace1)

    trace2 = go.Scatter(
        x=valid.index,
        y=valid['Close'],
        mode='lines',
        name='Actual'
    )
    data.append(trace2)

   
    trace3 = go.Scatter(
            x=valid.index,
            y=valid['Predictions_Close'],
            mode='lines',
            name='Predictions Close'
        )
    data.append(trace3)
    trace4 = go.Scatter(
            x=train_roc.index,
            y=train_roc['roc'],
            mode='lines',
            name='Train'
        )
    data1.append(trace4)
    trace5 = go.Scatter(
            x=valid_roc.index,
            y=valid_roc['roc'],
            mode='lines',
            name='Actual'
        )
    data1.append(trace5)
    trace6 = go.Scatter(
            x=valid_roc.index,
            y=valid_roc['Predictions_Close'],
            mode='lines',
            name='Predictions Close'
        )
    data1.append(trace6)
    return [{
        'data': data,
        'layout': go.Layout(
            title='BTC-USD Actual vs Predicted Prices',
            xaxis={'title': 'Time'},
            yaxis={'title': 'Price'}
        )
    },
    {
        'data': data1,
        'layout': go.Layout(
            title='ROC',
            xaxis={'title': 'Time'},
            yaxis={'title': 'BTC-USD Actual vs Predicted ROC'}
        )
    }]

# Callback to stop the program
@app.callback(
    Output('stop-button', 'n_clicks'),
    [Input('stop-button', 'n_clicks')]
)
def stop_program(n_clicks):
    if n_clicks > 0:
        os._exit(1) # This will forcefully terminate the process
    return n_clicks

def load_and_predict(df, model_type, feature):
    # Loading and processing data
    print(len(df))
    df["timestamp"] = pd.to_datetime(df.timestamp, format="%Y-%m-%d")
    df.index = df['timestamp']
    data = df.sort_index(ascending=True, axis=0)
    new_data = pd.DataFrame(index=range(0, len(df)), columns=['Timestamp', 'Close'])
    
    for i in range(0, len(data)):
        new_data["Timestamp"][i] = data['timestamp'][i]
        new_data["Close"][i] = data["close"][i]
    
    new_data.index = new_data.Timestamp
    new_data.drop("Timestamp", axis=1, inplace=True)
    if(feature == False):
        new_data['roc'] = (new_data['Close'].astype(float).diff(3) / new_data['Close'].astype(float).shift(3)) * 100
        new_data['roc'] = new_data['roc'].fillna(0)
        new_data.drop("Close", axis=1, inplace=True)
        
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
    
    model = Sequential()
    if model_type == 'lstm':
        model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(LSTM(units=50))
        print("LSTM model")
    elif model_type == 'rnn':
        model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(SimpleRNN(units=50))
        print("RNN model")
    elif model_type == 'gru':
        model.add(GRU(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        model.add(GRU(units=50))
        print("GRU model")
    elif model_type == 'cnn':
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(x_train.shape[1], 1)))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(50, activation='relu'))
        print("CNN model")
       
    model.add(Dense(1))
    inputs = new_data[len(new_data) - len(valid) - 60:].values
    inputs = inputs.reshape(-1, 1)
    inputs = scaler.transform(inputs)
    
    model.compile(loss='mean_squared_error', optimizer='adamw')
    model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)
    
    X_test = []
    for i in range(60, inputs.shape[0]):
        X_test.append(inputs[i-60:i, 0])
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    closing_price = model.predict(X_test)
    closing_price = scaler.inverse_transform(closing_price)
    train = new_data[:int(len(dataset)*0.8)]
    valid = new_data[int(len(dataset)*0.8):]
    valid['Predictions_Close'] = closing_price
    return train, valid

def fetch_historical_candles(symbol, interval, limit=1000):
    candles = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(candles, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 
        'close_time', 'quote_asset_volume', 'number_of_trades', 
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# Fetch the BTCUSDT 1-minute candles
symbol = 'BTCUSDT'
interval = '1m'
historical_data = fetch_historical_candles(symbol, interval)

async def handle_message(websocket):
    async for message in websocket:
        data = json.loads(message)
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
        time.sleep(15)

async def connect_to_websocket(symbol):
    websocket_url = f"wss://stream.binance.com:9443/ws/{symbol.lower()}@kline_{interval}"
    async with websockets.connect(websocket_url) as websocket:
        await handle_message(websocket)

def fetch_data():
    asyncio.get_event_loop().run_until_complete(connect_to_websocket(symbol))

if __name__ == '__main__': 
    th = Thread(target=fetch_data)
    th.daemon = True
    th.start()
    app.run_server(debug=True)
    th.join()
    