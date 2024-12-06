import dash
import pandas as pd
from dash import dcc, html, Input, Output, State
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from bs4 import BeautifulSoup
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA



# Helper Functions
def calculate_overall_sentiment(bullish, bearish):
    if len(bullish) > len(bearish):
        return "BUY"
    elif len(bearish) > len(bullish):
        return "SELL"
    else:
        return "HOLD"
    
def get_stock_data(ticker, start_date, end_date):
    try:
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty or 'Close' not in stock_data.columns:
            raise ValueError("Stock data is empty or missing necessary columns.")
        stock_data.reset_index(inplace=True)
        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()

    
def prepare_data_for_sentiment(data):
    """
    Prepare data with enhanced features for Sentiment Analysis predictions.
    """
    data['Volatility'] = data['Close'].rolling(window=10).std()
    data['Volume_MA'] = data['Volume'].rolling(window=10).mean()
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days
    return data

def smooth_transition(historical_prices, predicted_prices, steps=10):
    """
    Smoothens the transition between historical and predicted prices.
    """
    last_price = historical_prices[-1]
    first_predicted_price = predicted_prices[0]
    transition = np.linspace(last_price, first_predicted_price, steps)
    return list(transition) + predicted_prices[1:]



# News Fetching Functions
def fetch_google_news(ticker):
    base_url = "https://news.google.com/rss/search?q="
    query = f"{ticker}+stock+news"
    url = f"{base_url}{query}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "xml")
        items = soup.find_all("item")
        
        headlines = []
        for item in items[:5]:  # Limit to the top 5
            headline = item.title.text
            description = item.description.text
            headlines.append({"headline": headline, "description": description})
        return headlines
    except Exception as e:
        print(f"Error fetching Google News: {e}")
        return []

# Train and Predict Future Prices
def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


from pmdarima import auto_arima

def predict_future_prices_arima(data, future_days):
    """
    Predicts future stock prices using ARIMA with automatic parameter selection.
    """
    try:
        # Fit ARIMA model with dynamic order selection
        model = auto_arima(
            data['Close'], 
            seasonal=False, 
            stepwise=True, 
            suppress_warnings=True, 
            error_action="ignore"
        )
        forecast = model.predict(n_periods=future_days)
        return forecast.tolist()
    except Exception as e:
        print(f"ARIMA Error: {e}")
        return [data['Close'].iloc[-1]] * future_days  # Flat prediction on failure


# Utility Funcitons
def fetch_stock_data(ticker, date_range):
    """
    Fetches stock data for a given ticker and date range.

    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL').
        date_range (str): Date range for the data (e.g., '1mo', '3mo', '1y').
    Returns:
        pd.DataFrame: Stock data with columns like 'Close' and 'returns'.
    """
    import yfinance as yf
    try:
        # Fetch historical data from yfinance
        stock_data = yf.download(ticker, period=date_range)
        
        # Ensure there is data and calculate daily returns
        if not stock_data.empty:
            stock_data['returns'] = stock_data['Close'].pct_change()
            stock_data = stock_data.reset_index()  # Reset index to include the 'Date' column
        else:
            return pd.DataFrame()  # Return empty DataFrame if no data

        return stock_data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return pd.DataFrame()  # Return empty DataFrame on error


# Sentiment classification keywords with themes
SENTIMENT_KEYWORDS = {
    "Bullish": {
        "growth": "This reflects growth opportunities that could positively impact the stock price.",
        "record profits": "The mention of record profits highlights strong financial performance.",
        "positive earnings": "Positive earnings signal robust financial health, boosting investor confidence.",
        "partnerships": "New partnerships drive innovation and market expansion, a bullish indicator.",
        "investments": "Significant investments indicate confidence in future growth, contributing to bullish sentiment.",
    },
    "Bearish": {
        "regulatory scrutiny": "Regulatory scrutiny could hinder operations or long-term growth.",
        "decline": "Declining performance or sentiment is a bearish signal for the stock.",
        "layoffs": "Layoffs may reflect underlying challenges, signaling a bearish outlook.",
        "missed targets": "Missed targets indicate underperformance, raising concerns about financial health.",
        "competition pressure": "Rising competition pressures could challenge profitability and market share.",
    }
}


def extract_sentiment_analysis(ticker, data):
    """
    Extracts insights from sentiment keywords and provides a recommendation.
    """
    # Example headlines for simplicity (Replace with real scraped headlines)
    headlines = [
        "Positive earnings boost investor confidence for {}".format(ticker),
        "Record profits achieved by {}".format(ticker),
        "Regulatory scrutiny could challenge {}'s growth".format(ticker),
    ]

    # Analyze sentiment in headlines
    sentiment_summary = {"Bullish": [], "Bearish": []}
    for headline in headlines:
        for sentiment, keywords in SENTIMENT_KEYWORDS.items():
            for keyword, description in keywords.items():
                if keyword in headline.lower():
                    sentiment_summary[sentiment].append(description)

    # Create a recommendation
    bullish_count = len(sentiment_summary["Bullish"])
    bearish_count = len(sentiment_summary["Bearish"])
    if bullish_count > bearish_count:
        recommendation = f"**{ticker}: Buy** - The sentiment leans strongly bullish."
    elif bearish_count > bullish_count:
        recommendation = f"**{ticker}: Hold or Sell** - The sentiment leans bearish."
    else:
        recommendation = f"**{ticker}: Hold** - Sentiment is mixed, further analysis recommended."

    # Generate formatted text
    insights = f"**Key Sentiment Analysis Insights for {ticker}:**\n"
    insights += "\n".join([f"**Bullish:** {desc}" for desc in sentiment_summary["Bullish"]])
    insights += "\n" + "\n".join([f"**Bearish:** {desc}" for desc in sentiment_summary["Bearish"]])
    insights += f"\n\n**Recommendation:** {recommendation}"

    return insights




# Function Definitions
def fetch_yfinance_news(ticker):
    url = f"https://finance.yahoo.com/quote/{ticker}?p={ticker}&.tsrc=fin-srch"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise ValueError("Yahoo Finance request failed.")

        soup = BeautifulSoup(response.text, "html.parser")
        headlines = []
        
        for item in soup.find_all("h3", class_="Mb(5px)"):
            headline = item.get_text(strip=True)
            link = item.a["href"] if item.a else ""
            headlines.append({"headline": headline, "description": link})
        return headlines[:5]  # Limit to top 5
    except Exception as e:
        print(f"Error fetching Yahoo Finance news for {ticker}: {e}")
        return []


def fetch_combined_news(ticker):
    # Fetch news from Google News and Yahoo Finance
    google_news = fetch_google_news(ticker)
    yfinance_news = fetch_yfinance_news(ticker)
    
    # Combine results
    combined_news = google_news + yfinance_news
    return combined_news


# Classify sentiment using nltk
analyzer = SentimentIntensityAnalyzer()

def classify_sentiment(news):
    bullish = []
    bearish = []
    neutral = []

    for article in news:
        combined_text = article["headline"] + " " + article["description"]
        sentiment = analyzer.polarity_scores(combined_text)
        category = "General"
        
        # Classify sentiment based on compound score
        if sentiment["compound"] > 0.05:
            # Check for keywords to personalize the reason
            reason = next(
                (desc for keyword, desc in SENTIMENT_KEYWORDS["Bullish"].items() if keyword in combined_text.lower()), 
                "The article mentions positive developments or signals strong performance."
            )
            category = "Positive Outlook"
            bullish.append({
                "headline": article["headline"],
                "description": article["description"],
                "sentiment": "Bullish",
                "score": sentiment["compound"],
                "reason": reason,
                "category": category
            })
        elif sentiment["compound"] < -0.05:
            reason = next(
                (desc for keyword, desc in SENTIMENT_KEYWORDS["Bearish"].items() if keyword in combined_text.lower()), 
                "The article highlights concerns, challenges, or negative trends."
            )
            category = "Risk/Concern"
            bearish.append({
                "headline": article["headline"],
                "description": article["description"],
                "sentiment": "Bearish",
                "score": sentiment["compound"],
                "reason": reason,
                "category": category
            })
        else:
            neutral.append({
                "headline": article["headline"],
                "description": article["description"],
                "sentiment": "Neutral",
                "score": sentiment["compound"],
                "reason": "The article is balanced with no strong positive or negative sentiment.",
                "category": "Neutral"
            })
    
    return bullish, bearish, neutral




def perform_sentiment_analysis(ticker, data):
    # Placeholder logic for sentiment analysis
    # Replace this with your actual sentiment analysis implementation
    future_dates = pd.date_range(start=data.index[-1], periods=30, freq='D')
    predicted_prices = [data['Close'].iloc[-1] + (i * 0.5) for i in range(30)]
    return {"dates": future_dates, "predicted_prices": predicted_prices}



# Initialize app with Bootstrap for better layout
app = dash.Dash(__name__, external_stylesheets=["https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css"])
app.title = "Stock Analysis Dashboard"



# App Layout
app.layout = html.Div(style={
    "fontFamily": "Arial, sans-serif",  # Global font applied to the entire app
    "backgroundColor": "#f9f9f9",  # Light background for the app
    "padding": "10px",  # Add overall padding for spacing
}, children=[
    # Header Section
    html.Div([
        html.H1("Welcome to the Stock Analysis Dashboard", style={
            "textAlign": "center",
            "marginBottom": "10px",
            "color": "#333",
            "fontFamily": "Arial, sans-serif",  # Apply global font here
            "fontSize": "36px"  # Slightly larger font for better emphasis
        }),
        html.P("Analyze stock performance, returns, and sentiment with ease.", style={
            "textAlign": "center",
            "color": "#666",
            "fontFamily": "Arial, sans-serif",  # Consistent font
            "fontSize": "18px"  # Slightly larger font for readability
        }),
    ], style={
        "backgroundColor": "#f1f1f1",  # Light background for distinction
        "padding": "20px",  # Add padding for spacing
        "borderBottom": "2px solid #ddd",  # Light border for separation
        "borderRadius": "5px"  # Slight rounding for a modern look
    }),

    

    # Tabs Section (Existing Tabs)
    dcc.Tabs([
        # Tab 1: Price & Returns Analysis
        dcc.Tab(label="Price & Returns Analysis", children=[
            html.Div([
                dcc.Input(
                    id="stock_ticker_1",
                    type="text",
                    placeholder="Enter ticker 1 (e.g., AAPL)",
                    value="AAPL"
                ),
                dcc.Input(
                    id="stock_ticker_2",
                    type="text",
                    placeholder="Enter ticker 2 (e.g., MSFT)",
                    value="MSFT"
                ),
                dcc.Dropdown(
                    id="date_range",
                    options=[
                        {"label": "Last Month", "value": "1mo"},
                        {"label": "Last Quarter", "value": "3mo"},
                        {"label": "Last Year", "value": "1y"},
                        {"label": "All Time", "value": "max"}
                    ],
                    placeholder="Select date range",
                    style={"width": "250px"}
                ),
                html.Button(
                    "Submit",
                    id="submit_button",
                    n_clicks=0,
                    style={"marginTop": "10px", "marginBottom": "20px"}
                ),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "20px"}),

            # Metrics Section
            html.Div(id="metrics_table", style={
                "marginBottom": "20px",
                "padding": "10px",
                "border": "1px solid #ddd",
                "backgroundColor": "#f9f9f9",
            }),

            # Graphs for Price and Returns
            dcc.Graph(id="price_comparison_chart"),
            dcc.Graph(id="returns_comparison_chart"),
        ]),

        # Tab 2: Detailed Visuals
        dcc.Tab(label="Detailed Visuals", children=[
            html.Div([
                dcc.Input(
                    id="detailed_ticker1",
                    type="text",
                    placeholder="Enter ticker 1 (e.g., AAPL)",
                    value="AAPL"
                ),
                dcc.Input(
                    id="detailed_ticker2",
                    type="text",
                    placeholder="Enter ticker 2 (e.g., MSFT)",
                    value="MSFT"
                ),
                dcc.Dropdown(
                    id="detailed_date_range",
                    options=[
                        {"label": "Last Month", "value": "1mo"},
                        {"label": "Last Quarter", "value": "3mo"},
                        {"label": "Last Year", "value": "1y"},
                        {"label": "All Time", "value": "max"}
                    ],
                    placeholder="Select date range",
                    style={"width": "250px"}
                ),
                html.Button(
                    "Submit",
                    id="detailed_submit",
                    n_clicks=0,
                    style={"marginTop": "10px", "marginBottom": "20px"}
                ),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "20px"}),

            # Graphs for Detailed Visuals
            dcc.Graph(id="candlestick_chart"),
            dcc.Graph(id="volume_chart"),
            dcc.Graph(id="volatility_chart"),
        ]),

        # Tab 3: Sentiment Analysis
        dcc.Tab(label="Sentiment Analysis", children=[
            html.Div([
                dcc.Input(
                    id="sentiment_ticker_1",
                    type="text",
                    placeholder="Enter ticker 1 (e.g., AAPL)",
                    value="AAPL"
                ),
                dcc.Input(
                    id="sentiment_ticker_2",
                    type="text",
                    placeholder="Enter ticker 2 (e.g., MSFT)",
                    value="MSFT"
                ),
                html.Button(
                    "Analyze Sentiment",
                    id="sentiment_submit",
                    n_clicks=0,
                    style={"marginTop": "10px", "marginBottom": "20px"}
                ),
            ], style={"display": "flex", "gap": "10px", "marginBottom": "20px"}),

            # Graph and Sentiment Text
            dcc.Graph(id="sentiment_chart"),
            html.Div(id="sentiment_text", style={
                "padding": "20px",
                "backgroundColor": "#f9f9f9",
                "border": "1px solid #ddd",
                "fontSize": "16px",
                "lineHeight": "1.6"
            }),
        ])
    ]),

    # Footer Section
    html.Div([
        html.P("Created by Caleb Tong. Data powered by Yahoo Finance and Google News.", style={
            "textAlign": "center",
            "fontSize": "12px",
            "color": "#888"
        }),
    ], style={
        "backgroundColor": "#f1f1f1",
        "padding": "10px",
        "borderTop": "2px solid #ddd",
        "marginTop": "30px"
    })
])




# Callback for Price & Returns Analysis
@app.callback(
    [
        Output("price_comparison_chart", "figure"),
        Output("returns_comparison_chart", "figure"),
        Output("metrics_table", "children"),
    ],
    [Input("submit_button", "n_clicks")],
    [
        State("stock_ticker_1", "value"),
        State("stock_ticker_2", "value"),
        State("date_range", "value"),
    ],
)
def update_price_analysis(n_clicks, ticker1, ticker2, date_range):
    try:
        if not ticker1 or not ticker2:
            return go.Figure(), go.Figure(), "Error: Please provide valid tickers."

        # Fetch stock data
        data_ticker1 = fetch_stock_data(ticker1, date_range)
        data_ticker2 = fetch_stock_data(ticker2, date_range)

        if data_ticker1.empty or data_ticker2.empty:
            return go.Figure(),go.Figure(), "Error: Data not found for one or both tickers."

        # Fetch metrics
        ticker1_data = yf.Ticker(ticker1).info
        ticker2_data = yf.Ticker(ticker2).info

        ticker1_metrics = {
            "P/E Ratio": ticker1_data.get("trailingPE", "N/A"),
            "PEG Ratio": ticker1_data.get("pegRatio", "N/A"),
            "EPS": ticker1_data.get("trailingEps", "N/A"),
            "ROE": ticker1_data.get("returnOnEquity", "N/A"),
            "Debt/Equity": ticker1_data.get("debtToEquity", "N/A"),
        }

        ticker2_metrics = {
            "P/E Ratio": ticker2_data.get("trailingPE", "N/A"),
            "PEG Ratio": ticker2_data.get("pegRatio", "N/A"),
            "EPS": ticker2_data.get("trailingEps", "N/A"),
            "ROE": ticker2_data.get("returnOnEquity", "N/A"),
            "Debt/Equity": ticker2_data.get("debtToEquity", "N/A"),
        }

        # Create metrics table
        metrics_table = html.Table([
            html.Thead(html.Tr([html.Th("Metric"), html.Th(ticker1), html.Th(ticker2)])),
            html.Tbody([
                html.Tr([html.Td("P/E Ratio"), html.Td(ticker1_metrics["P/E Ratio"]), html.Td(ticker2_metrics["P/E Ratio"])]),
                html.Tr([html.Td("PEG Ratio"), html.Td(ticker1_metrics["PEG Ratio"]), html.Td(ticker2_metrics["PEG Ratio"])]),
                html.Tr([html.Td("EPS"), html.Td(ticker1_metrics["EPS"]), html.Td(ticker2_metrics["EPS"])]),
                html.Tr([html.Td("ROE"), html.Td(ticker1_metrics["ROE"]), html.Td(ticker2_metrics["ROE"])]),
                html.Tr([html.Td("Debt/Equity"), html.Td(ticker1_metrics["Debt/Equity"]), html.Td(ticker2_metrics["Debt/Equity"])]),
            ])
        ], style={"width": "100%", "borderCollapse": "collapse", "marginBottom": "20px"})

        # Create price comparison chart
        price_chart = go.Figure()
        price_chart.add_trace(go.Scatter(x=data_ticker1["Date"], y=data_ticker1["Close"], name=f"{ticker1} Price"))
        price_chart.add_trace(go.Scatter(x=data_ticker2["Date"], y=data_ticker2["Close"], name=f"{ticker2} Price"))
        price_chart.update_layout(title="Price Comparison", xaxis_title="Date", yaxis_title="Price")

        # Create returns comparison chart
        returns_chart = go.Figure()
        returns_chart.add_trace(go.Scatter(x=data_ticker1["Date"], y=data_ticker1["returns"], name=f"{ticker1} Returns"))
        returns_chart.add_trace(go.Scatter(x=data_ticker2["Date"], y=data_ticker2["returns"], name=f"{ticker2} Returns"))
        returns_chart.update_layout(title="Returns Comparison", xaxis_title="Date", yaxis_title="Returns")

        return price_chart, returns_chart, metrics_table

    except Exception as e:
        return go.Figure(), go.Figure(), f"An error occurred: {e}"

# Callback for Detailed Visuals Tab
@app.callback(
    [
        Output("candlestick_chart", "figure"),
        Output("volume_chart", "figure"),
        Output("volatility_chart", "figure"),
    ],
    [Input("detailed_submit", "n_clicks")],
    [
        State("detailed_ticker1", "value"),
        State("detailed_ticker2", "value"),
        State("detailed_date_range", "value"),
    ],
)
def update_detailed_visuals(n_clicks, ticker1, ticker2, date_range):
    try:
        if not ticker1 or not ticker2:
            return go.Figure(), go.Figure(), go.Figure()

        # Fetch stock data for both tickers
        data_ticker1 = yf.download(ticker1, period=date_range)
        data_ticker2 = yf.download(ticker2, period=date_range)

        if data_ticker1.empty or data_ticker2.empty:
            return go.Figure(), go.Figure(), go.Figure()

        # Create candlestick chart
        candlestick_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"{ticker1} Candlestick Chart with Moving Averages",
                f"{ticker2} Candlestick Chart with Moving Averages"
            ]
        )

        for data, ticker, col in zip([data_ticker1, data_ticker2], [ticker1, ticker2], [1, 2]):
            candlestick_fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name=f"{ticker} Candlestick"
                ),
                row=1, col=col
            )

        candlestick_fig.update_layout(
            height=600,
            title="Candlestick Charts with Moving Averages"
        )

        # Create volume charts
        volume_fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f"{ticker1} Volume", f"{ticker2} Volume"
            ]
        )

        for data, ticker, col in zip([data_ticker1, data_ticker2], [ticker1, ticker2], [1, 2]):
            volume_fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data["Volume"],
                    name=f"{ticker} Volume"
                ),
                row=1, col=col
            )

        volume_fig.update_layout(
            height=400,
            title="Volume Charts", barmode="overlay"
        )

        # Create combined volatility chart
        volatility_fig = go.Figure()

        for data, ticker, color in zip([data_ticker1, data_ticker2], [ticker1, ticker2], ["blue", "red"]):
            volatility_fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data["Close"].pct_change().rolling(window=20).std(),
                    mode="lines",
                    name=f"{ticker} Volatility",
                    line=dict(color=color)
                )
            )

        volatility_fig.update_layout(
            title="Volatility Comparison",
            xaxis_title="Date",
            yaxis_title="Volatility"
        )

        return candlestick_fig, volume_fig, volatility_fig

    except Exception as e:
        return go.Figure(), go.Figure(), go.Figure()
    
    
# Callback for Sentiment Analysis
@app.callback(
    [Output("sentiment_chart", "figure"), Output("sentiment_text", "children")],
    [Input("sentiment_submit", "n_clicks")],
    [State("sentiment_ticker_1", "value"), State("sentiment_ticker_2", "value")],
)
def update_sentiment_analysis(n_clicks, ticker1, ticker2):
    try:
        if not ticker1 or not ticker2:
            raise ValueError("Please provide valid tickers.")

        # Fetch historical data
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        data_ticker1 = get_stock_data(ticker1, start_date, end_date)
        data_ticker2 = get_stock_data(ticker2, start_date, end_date)

        if data_ticker1.empty or data_ticker2.empty:
            raise ValueError("Error fetching stock data. Please ensure the tickers are correct.")

        # Predict future prices using ARIMA
        future_days = 100
        future_prices1 = predict_future_prices_arima(data_ticker1, future_days)
        future_prices2 = predict_future_prices_arima(data_ticker2, future_days)

        # Smooth price transition
        transition_steps = 20
        smoothed_prices1 = smooth_transition(
            data_ticker1['Close'].values[-transition_steps:],
            future_prices1,
            steps=transition_steps,
        )
        smoothed_prices1 = list(data_ticker1['Close']) + smoothed_prices1
        smoothed_prices2 = smooth_transition(
            data_ticker2['Close'].values[-transition_steps:],
            future_prices2,
            steps=transition_steps,
        )
        smoothed_prices2 = list(data_ticker2['Close']) + smoothed_prices2

        # Generate future dates for plotting
        future_dates1 = pd.date_range(start=data_ticker1["Date"].iloc[-1], periods=future_days + 1, freq="B")
        future_dates2 = pd.date_range(start=data_ticker2["Date"].iloc[-1], periods=future_days + 1, freq="B")

        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data_ticker1["Date"],
            y=data_ticker1["Close"],
            name=f"{ticker1} Historical",
            line=dict(color="blue", dash="solid")
        ))
        fig.add_trace(go.Scatter(
            x=list(data_ticker1["Date"]) + list(future_dates1),
            y=smoothed_prices1,
            name=f"{ticker1} Predicted",
            line=dict(color="blue", dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=data_ticker2["Date"],
            y=data_ticker2["Close"],
            name=f"{ticker2} Historical",
            line=dict(color="green", dash="solid")
        ))
        fig.add_trace(go.Scatter(
            x=list(data_ticker2["Date"]) + list(future_dates2),
            y=smoothed_prices2,
            name=f"{ticker2} Predicted",
            line=dict(color="green", dash="dot")
        ))
        fig.update_layout(
            title="Stock Price Prediction with Sentiment Analysis and ARIMA",
            xaxis_title="Date",
            yaxis_title="Price",
        )

        # Sentiment Analysis
        news1 = fetch_combined_news(ticker1)
        news2 = fetch_combined_news(ticker2)

        bullish1, bearish1, neutral1 = classify_sentiment(news1)
        bullish2, bearish2, neutral2 = classify_sentiment(news2)

        overall1 = calculate_overall_sentiment(bullish1, bearish1)
        overall2 = calculate_overall_sentiment(bullish2, bearish2)

        # Build sentiment text
        sentiment_text = html.Div([
            html.H3(f"{ticker1}: Enhanced Sentiment Analysis"),
            html.Ul([
                html.Li(
                    f"Headline: {article['headline']} - "
                    f"(Sentiment: {article['sentiment']} | Score: {article['score']} | Category: {article['category']}) "
                    f"Reason: {article['reason']}"
                ) for article in bullish1 + bearish1 + neutral1
            ]),
            html.P(f"Overall Recommendation: {overall1}"),

            html.H3(f"{ticker2}: Enhanced Sentiment Analysis"),
            html.Ul([
                html.Li(
                    f"Headline: {article['headline']} - "
                    f"(Sentiment: {article['sentiment']} | Score: {article['score']} | Category: {article['category']}) "
                    f"Reason: {article['reason']}"
                ) for article in bullish2 + bearish2 + neutral2
            ]),
            html.P(f"Overall Recommendation: {overall2}"),

            html.H3("Comparison Summary"),
            html.P(
                f"Based on sentiment and growth adjustments, "
                f"{ticker1 if overall1 == 'BUY' else ticker2} is a better investment option."
            )
        ])

        return fig, sentiment_text

    except Exception as e:
        print(f"Error: {e}")
        return go.Figure(), f"An error occurred: {e}"




if __name__ == "__main__":
    app.run_server(debug=True)
