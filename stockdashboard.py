import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.subplots as sp
from textblob import TextBlob
from newsapi import NewsApiClient
import ta
from typing import Dict, List, Tuple
import plotly.express as px

st.set_page_config(
        page_title="stockdashboard",
        page_icon="📊",
        layout="wide",  # Set default to wide mode
        initial_sidebar_state="collapsed"  # Hide sidebar by default
    )

st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
        
        /* Base theme */
        .main * {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.9rem !important;
        }
        
        /* Headers */
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.2rem !important; }
        h4 { font-size: 1.1rem !important; }
        
        /* Data displays */
        .data-label {
            color: #666;
            font-size: 0.85rem !important;
        }
        .data-value {
            color: #FFF;
            font-size: 0.95rem !important;
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* Ticker cells */
        .ticker-cell {
            padding: 0.5rem;
            margin: 0.3rem;
            background-color: #1A1A1A;
            border: 1px solid #30363D;
            border-radius: 4px;
        }
        .ticker-cell .price {
            font-size: 1.2rem !important;
            font-weight: 600;
        }
        
        /* Cards and containers */
        .stats-container {
            padding: 0.5rem;
            margin: 0.3rem 0;
            background-color: #1A1A1A;
            border: 1px solid #30363D;
            border-radius: 4px;
        }
        
        /* News cards */
        .news-card {
            padding: 0.6rem;
            margin: 0.4rem 0;
            font-size: 0.85rem !important;
            background-color: #1A1A1A;
            border: 1px solid #30363D;
            border-radius: 4px;
        }
        .news-card h4 {
            font-size: 0.95rem !important;
            margin: 0.3rem 0;
        }
        .news-card p {
            font-size: 0.85rem !important;
            margin: 0.3rem 0;
        }
        
        /* Links */
        .news-link {
            color: #61dafb !important;
            text-decoration: none;
        }
        .news-link:hover {
            text-decoration: underline;
        }
    </style>
""", unsafe_allow_html=True)

# Updated Constants
MAJOR_INDICES = {
    "S&P 500": "^GSPC",
    "Dow Jones": "^DJI",
    "Nasdaq": "^IXIC",
    "Russell 2000": "^RUT"
}

POPULAR_ETFS = {
    "SPY (S&P 500)": "SPY",
    "QQQ (Nasdaq)": "QQQ",
    "DIA (Dow)": "DIA",
    "IWM (Russell)": "IWM",
    "VTI (Total Market)": "VTI",
    "VOO (S&P 500)": "VOO",
    "VEA (Developed)": "VEA",
    "VWO (Emerging)": "VWO",
    "AGG (Bonds)": "AGG",
    "GLD (Gold)": "GLD"
}

MARKET_SECTORS = {
    "Technology": "XLK",
    "Financials": "XLF",
    "Healthcare": "XLV",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Materials": "XLB",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communications": "XLC"
}

class StockAnalyzer:
    @staticmethod
    def get_market_data(symbol: str) -> pd.DataFrame:
        """Fetch and process stock/ETF data"""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period='1d', interval='1m')
            if df.empty:
                return pd.DataFrame()
            
            info = ticker.info
            return df, info
            
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame(), {}

    @staticmethod
    def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate stock-specific indicators"""
        try:
            df = df.copy()
            close_series = pd.Series(df['Close'].values, index=df.index)
            
            # Basic indicators
            df['SMA_20'] = ta.trend.sma_indicator(close_series, 20)
            df['EMA_20'] = ta.trend.ema_indicator(close_series, 20)
            df['RSI'] = ta.momentum.rsi(close_series)
            df['MACD'] = ta.trend.macd_diff(close_series)
            df['BB_upper'] = ta.volatility.bollinger_hband(close_series)
            df['BB_lower'] = ta.volatility.bollinger_lband(close_series)
            
            # Volume analysis
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Change'] = df['Volume'].pct_change()
            
            # Additional stock-specific metrics
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return df

    @staticmethod
    def get_fundamentals(info: Dict) -> Dict:
        """Extract and format key fundamental data"""
        fundamentals = {}
        try:
            # Handle market cap
            if 'marketCap' in info:
                fundamentals['marketCap'] = info['marketCap']
            elif 'totalAssets' in info:  # For ETFs
                fundamentals['marketCap'] = info['totalAssets']
            
            # Handle P/E ratios
            if 'trailingPE' in info:
                fundamentals['peRatio'] = info['trailingPE']
            elif 'forwardPE' in info:
                fundamentals['peRatio'] = info['forwardPE']
            
            # Forward P/E
            if 'forwardPE' in info:
                fundamentals['forwardPE'] = info['forwardPE']
            
            # Dividend Yield
            if 'dividendYield' in info:
                fundamentals['dividendYield'] = info['dividendYield']
            elif 'trailingAnnualDividendYield' in info:
                fundamentals['dividendYield'] = info['trailingAnnualDividendYield']
            
            # Beta
            if 'beta' in info:
                fundamentals['beta'] = info['beta']
            
            # 52 Week Change
            if 'fiftyTwoWeekChange' in info:
                fundamentals['52WeekChange'] = info['fiftyTwoWeekChange']
            elif 'regularMarketPrice' in info and 'fiftyTwoWeekLow' in info:
                change = (info['regularMarketPrice'] - info['fiftyTwoWeekLow']) / info['fiftyTwoWeekLow']
                fundamentals['52WeekChange'] = change
            
            return fundamentals
            
        except Exception as e:
            st.error(f"Error processing fundamentals: {str(e)}")
            return fundamentals

class MarketNewsAggregator:
    def __init__(self, api_key: str):
        self.newsapi = NewsApiClient(api_key=api_key)
    
    def _analyze_article(self, article: Dict) -> Dict:
        """Analyze article sentiment and format data"""
        try:
            text = f"{article['title']} {article['description']}"
            analysis = TextBlob(text)
            
            return {
                'title': article['title'],
                'description': article['description'],
                'source': article['source']['name'],
                'url': article['url'],
                'published': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
                'sentiment': analysis.sentiment.polarity,
                'subjectivity': analysis.sentiment.subjectivity
            }
        except Exception as e:
            st.error(f"Error analyzing article: {str(e)}")
            return None

    def get_stock_news(self, symbol: str = None) -> Dict:
        """Get stock-specific or general market news"""
        try:
            if symbol:
                query = f"{symbol} stock OR {symbol} market OR {symbol} trading"
            else:
                query = ("stock market OR wall street OR investing OR trading OR "
                        "market analysis OR financial markets")
            
            news = self.newsapi.get_everything(
                q=query,
                language='en',
                sort_by='publishedAt',
                page_size=30
            )
            
            # Process articles with error handling
            articles = []
            for article in news['articles']:
                if article['title'] and article['description']:
                    processed = self._analyze_article(article)
                    if processed:
                        articles.append(processed)
            
            # Split into market news and analysis
            analysis_articles = [a for a in articles if abs(a['sentiment']) > 0.2][:5]
            market_articles = [a for a in articles if a not in analysis_articles][:10]
            
            return {
                'market': market_articles,
                'analysis': analysis_articles
            }
            
        except Exception as e:
            st.error(f"Error fetching news: {str(e)}")
            return {'market': [], 'analysis': []}
        
class MarketTerminalUI:
    def __init__(self, news_aggregator: MarketNewsAggregator):
        self.news = news_aggregator
        self.analyzer = StockAnalyzer()
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _create_etf_chart(self, data: pd.DataFrame, title: str) -> go.Figure:
        """Create advanced chart for ETF analysis"""
        # Calculate moving averages for price chart
        data = self.analyzer.calculate_indicators(data)
        
        # Create subplots
        fig = sp.make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25],
            subplot_titles=(f'{title}', 'Volume', 'RSI')
        )

        # Add price candlesticks
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price',
                increasing_line_color='#26A69A',  # Green for up
                decreasing_line_color='#EF5350'   # Red for down
            ),
            row=1, col=1
        )

        # Add moving averages
        ema_colors = {'EMA_20': '#FFD700', 'EMA_50': '#FF8C00', 'EMA_200': '#FF4500'}
        for ma, color in ema_colors.items():
            if ma in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[ma],
                        name=ma,
                        line=dict(color=color, width=1)
                    ),
                    row=1, col=1
                )

        # Add Bollinger Bands
        if 'BB_upper' in data.columns and 'BB_lower' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_upper'],
                    name='BB Upper',
                    line=dict(color='rgba(200, 200, 200, 0.5)', dash='dash'),
                    showlegend=True
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['BB_lower'],
                    name='BB Lower',
                    line=dict(color='rgba(200, 200, 200, 0.5)', dash='dash'),
                    fill='tonexty',
                    fillcolor='rgba(200, 200, 200, 0.1)',
                    showlegend=True
                ),
                row=1, col=1
            )

        # Add volume bars
        colors = ['red' if row['Close'] < row['Open'] else 'green' 
                for _, row in data.iterrows()]
        
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )

        # Add RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name='RSI',
                    line=dict(color='#9C27B0', width=1)
                ),
                row=3, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", 
                        row=3, col=1, opacity=0.5)
            fig.add_hline(y=30, line_dash="dash", line_color="green", 
                        row=3, col=1, opacity=0.5)
            fig.add_hline(y=50, line_dash="dash", line_color="white", 
                        row=3, col=1, opacity=0.3)

        # Update layout
        fig.update_layout(
            height=700,
            template='plotly_dark',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)'
            ),
            margin=dict(t=30, l=10, r=10, b=10)
        )

        # Update axes
        fig.update_xaxes(
            rangeslider_visible=False,
            gridcolor='rgba(255,255,255,0.1)'
        )
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.1)',
            zerolinecolor='rgba(255,255,255,0.1)'
        )

        # Update Y-axis labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)

        return fig

    def render_header(self):
        """Render terminal header with proper spacing"""
        # Add empty space to account for streamlit's top bar
        st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
        
        # Header container
        st.markdown("""
            <div class="header-container">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 class="header-text">📊 STOCK MARKET ANALYTICS</h3>
                        <div class="timestamp">Last updated: {}</div>
                    </div>
                    <div>
                        <span style="color: #666;">Live Market Data</span>
                    </div>
                </div>
            </div>
        """.format(self.current_time), unsafe_allow_html=True)
        
        # Add small space after header
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    def _render_etf_metrics(self, fundamentals: Dict):
        """Render ETF fundamental metrics with proper formatting"""
        st.markdown("### Fund Metrics")
        
        # Get current price for additional calculations if needed
        current_price = fundamentals.get('regularMarketPrice', 0)
        
        metrics = [
            ('Market Cap', fundamentals.get('marketCap', 0), 
            lambda x: f"${x/1e9:.2f}B" if x > 1e6 else "$N/A"),
            
            ('P/E Ratio', fundamentals.get('peRatio', 0), 
            lambda x: f"{x:.2f}" if x > 0 else "N/A"),
            
            ('Forward P/E', fundamentals.get('forwardPE', 0), 
            lambda x: f"{x:.2f}" if x > 0 else "N/A"),
            
            ('Dividend Yield', fundamentals.get('dividendYield', 0), 
            lambda x: f"{x*100:.2f}%" if x > 0 else "N/A"),
            
            ('Beta', fundamentals.get('beta', 0), 
            lambda x: f"{x:.2f}" if x != 0 else "N/A"),
            
            ('52W Change', fundamentals.get('52WeekChange', 0), 
            lambda x: f"{x*100:+.2f}%" if x != 0 else "N/A")
        ]
        
        for label, value, formatter in metrics:
            formatted_value = formatter(value)
            value_color = ("green" if "+" in str(formatted_value) else 
                        "red" if "-" in str(formatted_value) else "white")
            
            st.markdown(f"""
                <div class="stats-container">
                    <div class="data-label">{label}</div>
                    <div class="data-value" style="color: {value_color}">
                        {formatted_value}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    def _render_news_card(self, article: Dict):
        """Render news card with clickable link"""
        sentiment_class = (
            "sentiment-positive" if article['sentiment'] > 0.1
            else "sentiment-negative" if article['sentiment'] < -0.1
            else "sentiment-neutral"
        )
        
        time_ago = self._get_time_ago(article['published'])
        
        st.markdown(f"""
            <div class="news-card">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                    <small>{article['source']} • {time_ago}</small>
                    <small class="{sentiment_class}">
                        {'▲' if article['sentiment'] > 0.1 else '▼' if article['sentiment'] < -0.1 else '■'}
                    </small>
                </div>
                <h4>
                    <a href="{article['url']}" target="_blank" class="news-link">
                        {article['title']}
                    </a>
                </h4>
                <p>{article['description'][:150]}...</p>
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def _get_time_ago(published_date: datetime) -> str:
        """Convert datetime to relative time string"""
        now = datetime.now()
        diff = now - published_date
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        hours = diff.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"
        minutes = (diff.seconds % 3600) // 60
        return f"{minutes}m ago"
    
    def render_market_overview(self):
        """Render market overview section"""
        st.markdown("### Market Overview")
        
        # Create columns for major indices
        cols = st.columns(len(MAJOR_INDICES))
        
        # Get and display data for each major index
        for idx, (name, symbol) in enumerate(MAJOR_INDICES.items()):
            data, info = self.analyzer.get_market_data(symbol)
            
            if not data.empty:
                try:
                    # Calculate current price and change
                    current = float(data['Close'].iloc[-1])
                    prev = float(data['Close'].iloc[-2])
                    change = ((current - prev) / prev) * 100
                    
                    # Create color-coded display
                    color = "green" if change >= 0 else "red"
                    arrow = "▲" if change >= 0 else "▼"
                    
                    with cols[idx]:
                        st.markdown(f"""
                            <div class="ticker-cell">
                                <div class="data-label">{name}</div>
                                <div style="color: {color}; font-size: 1.1rem;">
                                    {current:,.2f}
                                </div>
                                <div style="color: {color}; font-size: 0.9rem;">
                                    {arrow} {change:+.2f}%
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Add mini stats below main price
                        st.markdown(f"""
                            <div class="stats-container" style="margin-top: 0.5rem;">
                                <small class="data-label">Volume: {data['Volume'].iloc[-1]:,.0f}</small>
                            </div>
                        """, unsafe_allow_html=True)
                
                except Exception as e:
                    with cols[idx]:
                        st.error(f"Error displaying {name}")
        
        # Display market breadth indicators
        st.markdown("### Market Breadth")
        breadth_cols = st.columns(3)
        
        with breadth_cols[0]:
            try:
                # Fix RSI calculation
                sp500_data, _ = self.analyzer.get_market_data("^GSPC")
                if not sp500_data.empty and len(sp500_data) > 14:  # Ensure enough data points
                    close_series = pd.Series(sp500_data['Close'].astype(float))
                    rsi = ta.momentum.RSIIndicator(close_series, window=14).rsi()
                    current_rsi = float(rsi.iloc[-1])
                    
                    st.markdown(f"""
                        <div class="stats-container">
                            <div class="data-label">S&P 500 RSI</div>
                            <div class="data-value" style="color: {'red' if current_rsi > 70 else 'green' if current_rsi < 30 else 'white'}">
                                {current_rsi:.1f}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown("""
                    <div class="stats-container">
                        <div class="data-label">S&P 500 RSI</div>
                        <div class="data-value">N/A</div>
                    </div>
                """, unsafe_allow_html=True)
        
        with breadth_cols[1]:
            try:
                # Fix VIX calculation
                vix_data = yf.download("^VIX", period="1d", interval="1m")
                if not vix_data.empty:
                    current_vix = float(vix_data['Close'].iloc[-1])
                    color = "red" if current_vix > 30 else "green" if current_vix < 15 else "white"
                    
                    st.markdown(f"""
                        <div class="stats-container">
                            <div class="data-label">VIX</div>
                            <div class="data-value" style="color: {color}">
                                {current_vix:.2f}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown("""
                    <div class="stats-container">
                        <div class="data-label">VIX</div>
                        <div class="data-value">N/A</div>
                    </div>
                """, unsafe_allow_html=True)
        
        with breadth_cols[2]:
            try:
                # Treasury Yield
                tnx_data = yf.download("^TNX", period="1d", interval="1m")
                if not tnx_data.empty:
                    treasury_yield = float(tnx_data['Close'].iloc[-1])
                    st.markdown(f"""
                        <div class="stats-container">
                            <div class="data-label">10Y Treasury Yield</div>
                            <div class="data-value">{treasury_yield:.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown("""
                    <div class="stats-container">
                        <div class="data-label">10Y Treasury Yield</div>
                        <div class="data-value">N/A</div>
                    </div>
                """, unsafe_allow_html=True)
    
    def render_sector_performance(self):
        """Render sector performance heatmap"""
        st.markdown("### Sector Performance")
        
        # Get sector data
        sector_data = []
        for name, symbol in MARKET_SECTORS.items():
            try:
                data, _ = self.analyzer.get_market_data(symbol)
                if not data.empty:
                    daily_return = ((data['Close'].iloc[-1] - data['Close'].iloc[0]) / 
                                data['Close'].iloc[0] * 100)
                    
                    sector_data.append({
                        'Sector': name,
                        'Return': daily_return,
                        'Close': data['Close'].iloc[-1],
                        'Volume': data['Volume'].iloc[-1]
                    })
            except Exception as e:
                continue
        
        if sector_data:
            df = pd.DataFrame(sector_data)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Create treemap
                fig = px.treemap(
                    df,
                    path=['Sector'],
                    values=abs(df['Return']),
                    color='Return',
                    color_continuous_scale='RdYlGn',
                    title='Sector Performance (%)'
                )
                
                fig.update_layout(
                    template='plotly_dark',
                    height=400,
                    margin=dict(t=30, l=10, r=10, b=10)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Display sector metrics using a custom styled table
                df_display = df.sort_values('Return', ascending=False)
                
                st.markdown("""
                    <style>
                        .sector-table {
                            width: 100%;
                            border-collapse: collapse;
                        }
                        .sector-table th, .sector-table td {
                            padding: 8px;
                            text-align: left;
                            border-bottom: 1px solid #30363D;
                        }
                        .sector-table th {
                            background-color: #1A1A1A;
                            color: #666;
                            font-size: 0.8rem;
                        }
                        .return-value {
                            float: right;
                            font-family: monospace;
                        }
                    </style>
                    <div class="stats-container">
                        <div class="data-label">Sector Rankings</div>
                        <table class="sector-table">
                            <tr>
                                <th>Sector</th>
                                <th style="text-align: right">Return</th>
                            </tr>
                """, unsafe_allow_html=True)
                
                for _, row in df_display.iterrows():
                    color = '#00FF00' if row['Return'] > 0 else '#FF4B4B'
                    st.markdown(f"""
                        <tr>
                            <td style="color: #FFF;">{row['Sector']}</td>
                            <td>
                                <span class="return-value" style="color: {color};">
                                    {row['Return']:+.2f}%
                                </span>
                            </td>
                        </tr>
                    """, unsafe_allow_html=True)
                
                st.markdown("</table></div>", unsafe_allow_html=True)

    def render_etf_analysis(self):
        """Render ETF analysis section"""
        st.markdown("### ETF Analysis")
        
        # ETF selector
        selected_etf = st.selectbox("Select ETF", list(POPULAR_ETFS.keys()))
        
        # Get ETF data
        data, info = self.analyzer.get_market_data(POPULAR_ETFS[selected_etf])
        
        if not data.empty:
            # Calculate indicators
            data = self.analyzer.calculate_indicators(data)
            
            # Split view
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Price chart
                fig = self._create_etf_chart(data, selected_etf)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Metrics and analysis
                fundamentals = self.analyzer.get_fundamentals(info)
                self._render_etf_metrics(fundamentals)
                
                # Current statistics
                latest = data.iloc[-1]
                st.markdown("### Technical Indicators")
                
                indicators = {
                    'RSI': f"{latest.get('RSI', 0):.1f}",
                    'Volume': f"{latest.get('Volume', 0):,.0f}",
                    'Volatility': f"{latest.get('Volatility', 0)*100:.2f}%"
                }
                
                for label, value in indicators.items():
                    st.markdown(f"""
                        <div class="stats-container">
                            <div class="data-label">{label}</div>
                            <div class="data-value">{value}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # ETF-specific news
                news_data = self.news.get_stock_news(POPULAR_ETFS[selected_etf])
                if news_data['market']:
                    st.markdown("### Recent News")
                    for article in news_data['market'][:3]:
                        self._render_news_card(article)
        else:
            st.error(f"Unable to fetch data for {selected_etf}")

    def render_market_news(self):
        """Render market news section"""
        st.markdown("### Market News & Analysis")
        
        try:
            news_data = self.news.get_stock_news()
            
            if news_data['market'] or news_data['analysis']:
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("#### Latest Market News")
                    if news_data['market']:
                        for article in news_data['market']:
                            self._render_news_card(article)
                    else:
                        st.info("No market news available")
                
                with col2:
                    st.markdown("#### Market Analysis")
                    if news_data['analysis']:
                        for article in news_data['analysis']:
                            self._render_news_card(article)
                    else:
                        st.info("No market analysis available")
            else:
                st.warning("No news articles available at the moment")
            
        except Exception as e:
            st.error(f"Error rendering news section: {str(e)}")

    def _analyze_article(self, article: Dict) -> Dict:
        """Analyze article sentiment and format data"""
        text = f"{article['title']} {article['description']}"
        analysis = TextBlob(text)
        
        return {
            'title': article['title'],
            'description': article['description'],
            'source': article['source']['name'],
            'url': article['url'],
            'published': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
            'sentiment': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }

def main():
    # Initialize components
    news_aggregator = MarketNewsAggregator(api_key='82471e592e8c4d3d95ec33e3add393be')
    terminal = MarketTerminalUI(news_aggregator)
    
    # Render layout
    terminal.render_header()
    terminal.render_market_overview()
    terminal.render_sector_performance()
    terminal.render_etf_analysis()
    terminal.render_market_news()

if __name__ == "__main__":
    main()