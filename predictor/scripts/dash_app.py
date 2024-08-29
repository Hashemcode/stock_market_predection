import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import yfinance as yf
import requests
from flask import Flask
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
from sklearn.linear_model import LinearRegression

# Initialize the app with a custom dark theme
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.CYBORG])

# Custom CSS to ensure dropdowns, tables, and tabs are dark-themed
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dropdown styling */
            .Select-menu-outer {
                background-color: #2a2a2a !important;
                color: white !important;
            }
            .Select-control {
                background-color: #2a2a2a !important;
                color: white !important;
            }
            .Select--single .Select-control .Select-value {
                color: white !important;
            }
            /* Table styling */
            table {
                width: 100%;
                border-collapse: collapse;
                background-color: #2a2a2a;
                color: white;
            }
            th, td {
                padding: 8px 12px;
                border: 1px solid #444;
                text-align: left;
            }
            th {
                background-color: #1a1a1a;
            }
            /* Tab styling */
            .custom-tabs .nav-tabs {
                border-bottom: 2px solid #1a1a1a;
            }
            .custom-tabs .nav-tabs .nav-item.show .nav-link, 
            .custom-tabs .nav-tabs .nav-link.active {
                background-color: #000 !important;
                color: #fff !important;
                border-color: #1a1a1a #1a1a1a #000 !important;
            }
            .custom-tabs .nav-tabs .nav-link {
                background-color: #000 !important;
                color: #fff !important;
                border: 1px solid transparent;
                border-radius: 0 !important;
            }
            .custom-tabs .nav-tabs .nav-link:hover {
                color: #fff !important;
                background-color: #333 !important;
                border-color: #1a1a1a #1a1a1a #333 !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# List of stock symbols
stocks = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'TSLA', 'META', 'NFLX', 'NVDA']

# List of indices
indices = ['^GSPC', '^DJI', '^IXIC', '^RUT', '^FTSE', '^N225']

# List of cryptocurrencies
cryptos = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'XRP-USD', 'SOL-USD', 'DOGE-USD', 'DOT-USD']

# Function to get stock, index, and crypto data from Yahoo Finance
def get_yfinance_data(symbol):
    stock_data = yf.download(symbol, period='1y', interval='1d')
    stock_data.reset_index(inplace=True)
    return stock_data

# Function to calculate support, resistance, and trendline
def calculate_support_resistance_trendline(stock_data):
    stock_data['Support'] = stock_data['Close'].rolling(window=20, min_periods=1).min()
    stock_data['Resistance'] = stock_data['Close'].rolling(window=20, min_periods=1).max()
    X = np.arange(len(stock_data)).reshape(-1, 1)
    y = stock_data['Close'].values
    linear_regressor = LinearRegression()
    linear_regressor.fit(X, y)
    stock_data['Trendline'] = linear_regressor.predict(X)
    return stock_data

# Function to get predictions from Flask server
def get_predictions(ticker, period='6mo', is_crypto=False):
    url = 'http://127.0.0.1:5002/predict'
    asset_type = 'crypto' if is_crypto else 'stock'
    data = {'ticker': ticker, 'period': period, 'asset_type': asset_type}
    response = requests.post(url, json=data)
    
    if response.status_code != 200:
        raise ValueError(f"Error fetching predictions: {response.status_code}")
    
    prediction_data = response.json()
    return prediction_data

# Generate options for dropdown
def generate_options(symbols):
    return [{'label': symbol, 'value': symbol} for symbol in symbols]

# Layout of the app
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1('Stock and Crypto Price Prediction Dashboard', 
                        className='text-center mt-4 mb-4 text-white', 
                        style={'font-weight': 'bold', 'text-shadow': '2px 2px 5px rgba(0,0,0,0.5)'})
        , width=12)
    ]),
    
    dbc.Row([
        dbc.Col(dcc.Tabs(id='tabs', value='stocks', className='custom-tabs', children=[
            dcc.Tab(label='Stocks', value='stocks', children=[
                dbc.Card(dcc.Dropdown(id='stock-symbol', 
                                      options=generate_options(stocks), 
                                      placeholder="Select a stock symbol", 
                                      searchable=True,
                                      style={'background-color': '#2a2a2a', 'color': 'white'}),
                         className="mb-4")
            ]),
            dcc.Tab(label='Indices', value='indices', children=[
                dbc.Card(dcc.Dropdown(id='index-symbol', 
                                      options=generate_options(indices), 
                                      placeholder="Select an index symbol", 
                                      searchable=True,
                                      style={'background-color': '#2a2a2a', 'color': 'white'}),
                         className="mb-4")
            ]),
            dcc.Tab(label='Cryptocurrencies', value='cryptocurrencies', children=[
                dbc.Card(dcc.Dropdown(id='crypto-symbol', 
                                      options=generate_options(cryptos), 
                                      placeholder="Select a cryptocurrency symbol", 
                                      searchable=True,
                                      style={'background-color': '#2a2a2a', 'color': 'white'}),
                         className="mb-4")
            ])
        ]), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Live Price", className="card-title text-white"),
                    html.H2(id='live-price', className="card-text text-primary"),
                ])
            ], className="mt-4 mb-4 bg-dark shadow-lg")
        ], width=4),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Predicted Price", className="card-title text-white"),
                    html.H2(id='predicted-price', className="card-text text-success"),
                ])
            ], className="mt-4 mb-4 bg-dark shadow-lg")
        ], width=4),
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Model Performance Metrics", className="card-title text-white"),
                    html.P(id='mae', className="card-text text-warning"),
                    html.P(id='mse', className="card-text text-warning"),
                    html.P(id='r2', className="card-text text-warning"),
                ])
            ], className="mb-4 bg-dark shadow-lg")
        ], width=12)
    ]),

    dbc.Row([
        dbc.Col(dcc.Loading(
            id="loading",
            type="default",
            children=dcc.Graph(id='stock-price-chart', style={'height': '80vh'}),
        ), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Comparison Table", className="card-title text-white"),
                    dcc.Loading(
                        id="loading-table",
                        type="default",
                        children=html.Div(id='comparison-table')
                    )
                ])
            ], className="mb-4 bg-dark shadow-lg")
        ], width=12)
    ]),

    dcc.Interval(id='interval-component', interval=1*60*1000, n_intervals=0)
], fluid=True, style={'background-color': '#1a1a1a', 'color': 'white', 'min-height': '100vh'})

@app.callback(
    [Output('stock-price-chart', 'figure'),
     Output('live-price', 'children'),
     Output('predicted-price', 'children'),
     Output('mae', 'children'),
     Output('mse', 'children'),
     Output('r2', 'children'),
     Output('comparison-table', 'children')],
    [Input('tabs', 'value'),
     Input('stock-symbol', 'value'),
     Input('index-symbol', 'value'),
     Input('crypto-symbol', 'value'),
     Input('interval-component', 'n_intervals')]
)
def update_chart(tab, stock_symbol, index_symbol, crypto_symbol, n_intervals):
    symbol = None
    is_crypto = False

    if tab == 'stocks' and stock_symbol:
        symbol = stock_symbol
    elif tab == 'indices' and index_symbol:
        symbol = index_symbol
    elif tab == 'cryptocurrencies' and crypto_symbol:
        symbol = crypto_symbol
        is_crypto = True
    
    if symbol is None:
        return go.Figure(), 'Live Price: N/A', 'Predicted Price: N/A', 'MAE: N/A', 'MSE: N/A', 'R²: N/A', ''

    stock_data = get_yfinance_data(symbol)
    stock_data = calculate_support_resistance_trendline(stock_data)

    try:
        predictions = get_predictions(symbol, is_crypto=is_crypto)
    except ValueError as e:
        return go.Figure(), 'Live Price: N/A', 'Predicted Price: N/A', 'MAE: N/A', 'MSE: N/A', 'R²: N/A', ''

    actual = stock_data['Close'].iloc[-len(predictions['predictions']):]
    predicted = predictions['predictions']
    
    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=stock_data['Date'],
        open=stock_data['Open'],
        high=stock_data['High'],
        low=stock_data['Low'],
        close=stock_data['Close'],
        name='Actual Prices'
    ))

    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['Support'],
        mode='lines',
        name='Support Levels',
        line=dict(width=2, color='green')
    ))

    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['Resistance'],
        mode='lines',
        name='Resistance Levels',
        line=dict(width=2, color='red')
    ))

    fig.add_trace(go.Scatter(
        x=stock_data['Date'],
        y=stock_data['Trendline'],
        mode='lines',
        name='Trendline',
        line=dict(width=2, color='orange')
    ))

    fig.add_trace(go.Scatter(
        x=predictions['dates'],
        y=predictions['predictions'],
        mode='lines',
        name='Predicted Prices',
        line=dict(width=2, color='blue')
    ))

    fig.update_layout(
        title=f'{symbol} Price',
        yaxis_title='Price (USD)',
        xaxis_title='Date',
        height=800,
        showlegend=True,
        legend=dict(x=0.01, y=0.99),
        template='plotly_dark'
    )

    live_price = stock_data['Close'].iloc[-1]
    predicted_price = predictions['predictions'][-1]

    comparison_table = pd.DataFrame({
        'Date': predictions['dates'],
        'Actual': actual.values,
        'Predicted': predicted
    })

    comparison_table_html = html.Table([
        html.Thead(html.Tr([html.Th(col) for col in comparison_table.columns])),
        html.Tbody([
            html.Tr([
                html.Td(comparison_table.iloc[i][col]) for col in comparison_table.columns
            ]) for i in range(len(comparison_table))
        ])
    ])

    return fig, f'Live Price: {live_price}', f'Predicted Price: {predicted_price}', f'MAE: {mae}', f'MSE: {mse}', f'R²: {r2}', comparison_table_html

if __name__ == '__main__':
    app.run_server(debug=True)
