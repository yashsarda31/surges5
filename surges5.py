import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json

# Page configuration
st.set_page_config(
    page_title="Stock Intelligence Hub",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI - FIXED TEXT COLORS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        padding: 0rem 1rem;
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        border-radius: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    .metric-card {
        background: white;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
        color: #1e293b !important;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
    }
    
    .metric-card h4 {
        color: #1e293b !important;
    }
    
    .metric-card p {
        color: #64748b !important;
    }
    
    .score-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #1e293b !important;
    }
    
    .glass-card h2, .glass-card h3 {
        color: #1e293b !important;
    }
    
    .glass-card p {
        color: #64748b !important;
    }
    
    /* Only apply gradient to the main header */
    .gradient-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .info-pill {
        display: inline-block;
        padding: 6px 12px;
        background: #f3f4f6;
        border-radius: 20px;
        font-size: 12px;
        margin: 2px;
        color: #1e293b !important;
    }
    
    /* Additional metrics cards - Fixed text colors */
    .additional-metric-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .additional-metric-card .metric-icon {
        font-size: 20px;
        margin-bottom: 5px;
    }
    
    .additional-metric-card .metric-label {
        color: #64748b !important;
        font-size: 12px;
        margin: 5px 0;
    }
    
    .additional-metric-card .metric-value {
        font-size: 18px;
        font-weight: bold;
        color: #1e293b !important;
    }
    
    /* Ensure all text in white backgrounds is dark */
    .white-bg-card {
        background: white;
        color: #1e293b !important;
    }
    
    .white-bg-card * {
        color: #1e293b !important;
    }
    </style>
    """, unsafe_allow_html=True)

class TechnicalAnalyzer:
    """Technical analysis calculations"""
    
    @staticmethod
    def calculate_ema(data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        middle_band = data.rolling(window=period).mean()
        std = data.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def calculate_vwap(high, low, close, volume):
        """Calculate Volume Weighted Average Price"""
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).cumsum() / volume.cumsum()
        return vwap
    
    @staticmethod
    def calculate_rsi(data, period=14):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

class EnhancedStockAnalyzer:
    def __init__(self, ticker_symbol):
        """Initialize with stock ticker symbol"""
        self.ticker_symbol = ticker_symbol.upper()
        self.ticker = yf.Ticker(self.ticker_symbol)
        self.info = {}
        self.financials = None
        self.balance_sheet = None
        self.cashflow = None
        self.earnings = None
        self._load_all_data()
        
    def _load_all_data(self):
        """Load all available data from yfinance"""
        try:
            self.info = self.ticker.info or {}
            self.financials = self.ticker.financials
            self.balance_sheet = self.ticker.balance_sheet
            self.cashflow = self.ticker.cashflow
            self.earnings = self.ticker.earnings
            
            # Get quarterly financials for more recent data
            self.quarterly_financials = self.ticker.quarterly_financials
            self.quarterly_cashflow = self.ticker.quarterly_cashflow
        except:
            pass
    
    def get_price_data(self, period="6mo"):
        """Get historical price data for technical analysis"""
        try:
            data = self.ticker.history(period=period)
            return data
        except:
            return pd.DataFrame()
    
    def calculate_beta(self):
        """Calculate or fetch Beta"""
        beta = self.info.get('beta')
        
        if beta is None:
            # Calculate beta manually using 1-year price data
            try:
                stock_data = self.ticker.history(period="1y")
                market_data = yf.Ticker("^GSPC").history(period="1y")  # S&P 500
                
                if not stock_data.empty and not market_data.empty:
                    stock_returns = stock_data['Close'].pct_change().dropna()
                    market_returns = market_data['Close'].pct_change().dropna()
                    
                    # Align the dates
                    aligned_data = pd.DataFrame({
                        'stock': stock_returns,
                        'market': market_returns
                    }).dropna()
                    
                    if len(aligned_data) > 30:  # Need sufficient data points
                        covariance = aligned_data.cov().iloc[0, 1]
                        market_variance = aligned_data['market'].var()
                        beta = covariance / market_variance if market_variance != 0 else None
            except:
                pass
        
        return beta
    
    def calculate_peg_ratio(self):
        """Calculate PEG ratio from available data"""
        peg = self.info.get('pegRatio')
        
        if peg is None:
            # Try to calculate PEG manually
            pe_ratio = self.info.get('trailingPE') or self.info.get('forwardPE')
            
            if pe_ratio:
                # Try different growth rate sources
                growth_rate = None
                
                # Option 1: Use earnings growth
                earnings_growth = self.info.get('earningsGrowth')
                if earnings_growth:
                    growth_rate = earnings_growth * 100
                
                # Option 2: Use revenue growth
                if growth_rate is None:
                    revenue_growth = self.info.get('revenueGrowth')
                    if revenue_growth:
                        growth_rate = revenue_growth * 100
                
                # Option 3: Calculate from historical earnings
                if growth_rate is None and self.earnings is not None:
                    try:
                        if len(self.earnings) >= 2:
                            recent_earnings = self.earnings['Earnings'].iloc[-2:]
                            if recent_earnings.iloc[0] > 0:
                                growth_rate = ((recent_earnings.iloc[1] - recent_earnings.iloc[0]) / recent_earnings.iloc[0]) * 100
                    except:
                        pass
                
                # Calculate PEG if we have growth rate
                if growth_rate and growth_rate > 0:
                    peg = pe_ratio / growth_rate
        
        return peg
    
    def calculate_ev_to_ebitda(self):
        """Calculate EV/EBITDA from available data"""
        ev_ebitda = self.info.get('enterpriseToEbitda')
        
        if ev_ebitda is None:
            # Calculate manually
            enterprise_value = self.info.get('enterpriseValue')
            ebitda = self.info.get('ebitda')
            
            if enterprise_value and ebitda and ebitda > 0:
                ev_ebitda = enterprise_value / ebitda
            else:
                # Try to calculate EBITDA from financials
                if self.financials is not None and not self.financials.empty:
                    try:
                        # Get most recent data
                        if 'Total Revenue' in self.financials.index and 'Operating Income' in self.financials.index:
                            revenue = self.financials.loc['Total Revenue'].iloc[0]
                            operating_income = self.financials.loc['Operating Income'].iloc[0]
                            
                            # Rough EBITDA calculation
                            if revenue and operating_income:
                                ebitda_estimate = operating_income * 1.3  # Rough estimate adding back D&A
                                
                                if enterprise_value and ebitda_estimate > 0:
                                    ev_ebitda = enterprise_value / ebitda_estimate
                    except:
                        pass
        
        return ev_ebitda
    
    def calculate_operating_cashflow(self):
        """Calculate or fetch operating cash flow"""
        ocf = None
        
        # Try different sources for operating cash flow
        if self.cashflow is not None and not self.cashflow.empty:
            try:
                if 'Total Cash From Operating Activities' in self.cashflow.index:
                    ocf = self.cashflow.loc['Total Cash From Operating Activities'].iloc[0]
                elif 'Operating Cash Flow' in self.cashflow.index:
                    ocf = self.cashflow.loc['Operating Cash Flow'].iloc[0]
            except:
                pass
        
        # Try quarterly data if annual is not available
        if ocf is None and self.quarterly_cashflow is not None and not self.quarterly_cashflow.empty:
            try:
                if 'Total Cash From Operating Activities' in self.quarterly_cashflow.index:
                    # Sum last 4 quarters
                    ocf = self.quarterly_cashflow.loc['Total Cash From Operating Activities'].iloc[:4].sum()
                elif 'Operating Cash Flow' in self.quarterly_cashflow.index:
                    ocf = self.quarterly_cashflow.loc['Operating Cash Flow'].iloc[:4].sum()
            except:
                pass
        
        # Fallback: Use free cash flow if available
        if ocf is None:
            fcf = self.info.get('freeCashflow')
            if fcf:
                # Operating cash flow is typically higher than free cash flow
                ocf = fcf * 1.3  # Rough estimate
        
        return ocf
    
    def calculate_additional_metrics(self):
        """Calculate additional helpful metrics"""
        metrics = {}
        
        # P/E Ratio
        metrics['pe_ratio'] = self.info.get('trailingPE') or self.info.get('forwardPE')
        
        # Price to Book
        metrics['price_to_book'] = self.info.get('priceToBook')
        
        # Profit Margin
        metrics['profit_margin'] = self.info.get('profitMargins')
        
        # ROE
        metrics['roe'] = self.info.get('returnOnEquity')
        
        # Debt to Equity
        metrics['debt_to_equity'] = self.info.get('debtToEquity')
        
        # Current Ratio
        metrics['current_ratio'] = self.info.get('currentRatio')
        
        # Revenue Growth
        metrics['revenue_growth'] = self.info.get('revenueGrowth')
        
        return metrics

class MetricScorer:
    @staticmethod
    def score_beta(beta):
        """Score Beta (0-25 points)"""
        if beta is None:
            return 12, "Unable to calculate - Neutral score assigned", "#94a3b8"
        
        if 0.8 <= beta <= 1.2:
            return 25, "Excellent - Optimal volatility balance", "#10b981"
        elif 0.5 <= beta < 0.8:
            return 20, "Good - Lower volatility, stable", "#22c55e"
        elif 1.2 < beta <= 1.5:
            return 15, "Moderate - Higher volatility", "#eab308"
        elif 0 <= beta < 0.5:
            return 10, "Low volatility - Limited growth potential", "#f59e0b"
        elif beta > 1.5:
            return 5, "High risk - Excessive volatility", "#ef4444"
        else:
            return 3, "Inverse market correlation", "#dc2626"
    
    @staticmethod
    def score_peg(peg):
        """Score PEG Ratio (0-25 points)"""
        if peg is None:
            return 12, "Unable to calculate - Neutral score assigned", "#94a3b8"
        
        if peg <= 0:
            return 5, "Negative or no growth", "#dc2626"
        elif 0 < peg <= 1:
            return 25, "Excellent - Potentially undervalued", "#10b981"
        elif 1 < peg <= 1.5:
            return 20, "Good - Reasonably valued", "#22c55e"
        elif 1.5 < peg <= 2:
            return 15, "Fair - Slightly overvalued", "#eab308"
        elif 2 < peg <= 3:
            return 10, "Poor - Overvalued", "#f59e0b"
        else:
            return 5, "Very overvalued", "#ef4444"
    
    @staticmethod
    def score_ev_ebitda(ev_ebitda):
        """Score EV/EBITDA (0-25 points)"""
        if ev_ebitda is None or ev_ebitda < 0:
            return 12, "Unable to calculate - Neutral score assigned", "#94a3b8"
        
        if 0 < ev_ebitda <= 8:
            return 25, "Excellent - Attractive valuation", "#10b981"
        elif 8 < ev_ebitda <= 12:
            return 20, "Good - Fair valuation", "#22c55e"
        elif 12 < ev_ebitda <= 15:
            return 15, "Moderate - Market valuation", "#eab308"
        elif 15 < ev_ebitda <= 20:
            return 10, "Poor - Expensive valuation", "#f59e0b"
        else:
            return 5, "Very expensive", "#ef4444"
    
    @staticmethod
    def score_cashflow(ocf):
        """Score Operating Cash Flow (0-25 points)"""
        if ocf is None:
            return 12, "Unable to calculate - Neutral score assigned", "#94a3b8"
        
        ocf_billions = ocf / 1e9
        
        if ocf_billions > 20:
            return 25, f"Excellent - Strong cash generation (${ocf_billions:.1f}B)", "#10b981"
        elif 5 < ocf_billions <= 20:
            return 20, f"Good - Healthy cash flow (${ocf_billions:.1f}B)", "#22c55e"
        elif 1 < ocf_billions <= 5:
            return 15, f"Moderate cash flow (${ocf_billions:.1f}B)", "#eab308"
        elif 0 < ocf_billions <= 1:
            return 10, f"Low but positive (${ocf_billions:.2f}B)", "#f59e0b"
        else:
            return 5, f"Negative cash flow (${ocf_billions:.1f}B)", "#ef4444"

def create_technical_chart(price_data, ticker_symbol):
    """Create technical analysis chart with indicators"""
    if price_data.empty:
        return None
    
    # Calculate technical indicators
    tech_analyzer = TechnicalAnalyzer()
    
    # Calculate indicators
    price_data['EMA_50'] = tech_analyzer.calculate_ema(price_data['Close'], 50)
    price_data['VWAP'] = tech_analyzer.calculate_vwap(
        price_data['High'], 
        price_data['Low'], 
        price_data['Close'], 
        price_data['Volume']
    )
    
    upper_bb, middle_bb, lower_bb = tech_analyzer.calculate_bollinger_bands(price_data['Close'])
    price_data['Upper_BB'] = upper_bb
    price_data['Middle_BB'] = middle_bb
    price_data['Lower_BB'] = lower_bb
    
    price_data['RSI'] = tech_analyzer.calculate_rsi(price_data['Close'])
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f'{ticker_symbol} - Price & Technical Indicators', 'Volume', 'RSI')
    )
    
    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=price_data.index,
            open=price_data['Open'],
            high=price_data['High'],
            low=price_data['Low'],
            close=price_data['Close'],
            name='Price',
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add 50 EMA
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['EMA_50'],
            mode='lines',
            name='EMA 50',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Add VWAP
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['VWAP'],
            mode='lines',
            name='VWAP',
            line=dict(color='purple', width=2, dash='dash')
        ),
        row=1, col=1
    )
    
    # Add Bollinger Bands
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['Upper_BB'],
            mode='lines',
            name='Upper BB',
            line=dict(color='gray', width=1),
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['Lower_BB'],
            mode='lines',
            name='Lower BB',
            line=dict(color='gray', width=1),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.1)',
            showlegend=True
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['Middle_BB'],
            mode='lines',
            name='Middle BB',
            line=dict(color='orange', width=1, dash='dot'),
            showlegend=True
        ),
        row=1, col=1
    )
    
    # Add volume chart
    colors = ['red' if close < open else 'green' 
              for close, open in zip(price_data['Close'], price_data['Open'])]
    
    fig.add_trace(
        go.Bar(
            x=price_data.index,
            y=price_data['Volume'],
            name='Volume',
            marker_color=colors,
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add RSI
    fig.add_trace(
        go.Scatter(
            x=price_data.index,
            y=price_data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='purple', width=2),
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Add RSI reference lines
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=True,
        xaxis_rangeslider_visible=False,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    # Update x-axis
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig

def create_speedometer(value, max_value=100, title="Score"):
    """Create a modern speedometer chart"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 24}},
        delta = {'reference': 50, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, max_value], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25], 'color': '#fee2e2'},
                {'range': [25, 50], 'color': '#fed7aa'},
                {'range': [50, 75], 'color': '#fef3c7'},
                {'range': [75, 100], 'color': '#d1fae5'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def create_radar_chart(metrics_dict):
    """Create a radar chart for metrics visualization"""
    categories = list(metrics_dict.keys())
    values = [metrics_dict[cat]['score'] for cat in categories]
    max_values = [metrics_dict[cat]['max_score'] for cat in categories]
    
    fig = go.Figure()
    
    # Add trace for actual scores
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Actual Score',
        fillcolor='rgba(99, 102, 241, 0.3)',
        line=dict(color='rgb(99, 102, 241)', width=2)
    ))
    
    # Add trace for max scores
    fig.add_trace(go.Scatterpolar(
        r=max_values,
        theta=categories,
        name='Max Score',
        line=dict(color='rgba(200, 200, 200, 0.5)', width=1, dash='dash')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 25]
            )),
        showlegend=True,
        height=400,
        title="Metrics Radar Analysis"
    )
    
    return fig

def main():
    # Header with gradient - white text on gradient background
    st.markdown("""
        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 20px; margin-bottom: 2rem;'>
            <h1 style='text-align: center; color: white !important; margin: 0; font-weight: 700;'>🎯 Stock Intelligence Hub</h1>
            <p style='text-align: center; color: rgba(255,255,255,0.9) !important; margin-top: 10px;'>Advanced Fundamental Analysis & Technical Charts</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("### 📊 Analysis Panel")
        
        # Ticker input with modern styling
        ticker_input = st.text_input(
            "Stock Ticker",
            placeholder="AAPL",
            help="Enter a valid US stock ticker"
        ).strip().upper()
        
        # Quick select buttons
        st.markdown("**Quick Select:**")
        quick_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        cols = st.columns(2)
        for i, stock in enumerate(quick_stocks):
            with cols[i % 2]:
                if st.button(stock, key=f"quick_{stock}", use_container_width=True):
                    ticker_input = stock
        
        st.markdown("---")
        
        # Analyze button
        analyze_button = st.button(
            "🔍 Analyze Stock",
            type="primary",
            use_container_width=True
        )
        
        # Additional options
        with st.expander("⚙️ Settings"):
            show_additional = st.checkbox("Show additional metrics", value=True)
            show_radar = st.checkbox("Show radar chart", value=True)
            show_technical = st.checkbox("Show technical analysis", value=True)
            chart_period = st.selectbox(
                "Chart Period",
                options=["1mo", "3mo", "6mo", "1y", "2y"],
                index=2
            )
            export_enabled = st.checkbox("Enable export", value=True)
    
    with col2:
        if analyze_button and ticker_input:
            try:
                # Create progress bar
                progress_bar = st.progress(0, text="Initializing analysis...")
                
                # Initialize analyzer
                progress_bar.progress(20, text="Fetching stock data...")
                analyzer = EnhancedStockAnalyzer(ticker_input)
                scorer = MetricScorer()
                
                # Get company info
                progress_bar.progress(30, text="Processing company information...")
                company_name = analyzer.info.get('longName', ticker_input)
                sector = analyzer.info.get('sector', 'N/A')
                industry = analyzer.info.get('industry', 'N/A')
                market_cap = analyzer.info.get('marketCap', 0)
                current_price = analyzer.info.get('currentPrice') or analyzer.info.get('regularMarketPrice', 0)
                
                # Company header card - ensure dark text on white background
                st.markdown(f"""
                    <div class='glass-card'>
                        <h2 style='margin: 0; color: #1e293b !important;'>{company_name}</h2>
                        <div style='margin-top: 10px;'>
                            <span class='info-pill'>📍 {sector}</span>
                            <span class='info-pill'>🏢 {industry}</span>
                            <span class='info-pill'>💰 ${current_price:.2f}</span>
                            <span class='info-pill'>📊 MCap: ${market_cap/1e9:.1f}B</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Technical Analysis Section (NEW)
                if show_technical:
                    progress_bar.progress(40, text="Loading technical data...")
                    st.markdown("### 📈 Technical Analysis")
                    
                    # Get price data
                    price_data = analyzer.get_price_data(period=chart_period)
                    
                    if not price_data.empty:
                        # Create technical chart
                        tech_chart = create_technical_chart(price_data, ticker_input)
                        if tech_chart:
                            st.plotly_chart(tech_chart, use_container_width=True)
                        
                        # Display current technical indicators
                        tech_analyzer = TechnicalAnalyzer()
                        current_rsi = tech_analyzer.calculate_rsi(price_data['Close']).iloc[-1]
                        current_ema50 = tech_analyzer.calculate_ema(price_data['Close'], 50).iloc[-1]
                        current_close = price_data['Close'].iloc[-1]
                        
                        # Technical indicators summary
                        tech_cols = st.columns(4)
                        
                        with tech_cols[0]:
                            rsi_color = "#22c55e" if 30 <= current_rsi <= 70 else "#ef4444"
                            st.markdown(f"""
                                <div class='additional-metric-card'>
                                    <div class='metric-icon'>📊</div>
                                    <div class='metric-label'>RSI</div>
                                    <div class='metric-value' style='color: {rsi_color} !important;'>{current_rsi:.1f}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with tech_cols[1]:
                            ema_signal = "Above" if current_close > current_ema50 else "Below"
                            ema_color = "#22c55e" if ema_signal == "Above" else "#ef4444"
                            st.markdown(f"""
                                <div class='additional-metric-card'>
                                    <div class='metric-icon'>📈</div>
                                    <div class='metric-label'>EMA 50</div>
                                    <div class='metric-value'>${current_ema50:.2f}</div>
                                    <div style='color: {ema_color} !important; font-size: 12px;'>{ema_signal}</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with tech_cols[2]:
                            volume_avg = price_data['Volume'].rolling(20).mean().iloc[-1]
                            current_volume = price_data['Volume'].iloc[-1]
                            volume_ratio = current_volume / volume_avg
                            st.markdown(f"""
                                <div class='additional-metric-card'>
                                    <div class='metric-icon'>📊</div>
                                    <div class='metric-label'>Volume Ratio</div>
                                    <div class='metric-value'>{volume_ratio:.2f}x</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with tech_cols[3]:
                            price_change = price_data['Close'].pct_change().iloc[-1] * 100
                            change_color = "#22c55e" if price_change > 0 else "#ef4444"
                            st.markdown(f"""
                                <div class='additional-metric-card'>
                                    <div class='metric-icon'>💹</div>
                                    <div class='metric-label'>Daily Change</div>
                                    <div class='metric-value' style='color: {change_color} !important;'>{price_change:+.2f}%</div>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Unable to load technical data for this ticker")
                
                # Calculate fundamental metrics
                progress_bar.progress(60, text="Calculating fundamental metrics...")
                beta = analyzer.calculate_beta()
                peg = analyzer.calculate_peg_ratio()
                ev_ebitda = analyzer.calculate_ev_to_ebitda()
                ocf = analyzer.calculate_operating_cashflow()
                
                # Score metrics
                progress_bar.progress(80, text="Scoring metrics...")
                beta_score, beta_msg, beta_color = scorer.score_beta(beta)
                peg_score, peg_msg, peg_color = scorer.score_peg(peg)
                ev_score, ev_msg, ev_color = scorer.score_ev_ebitda(ev_ebitda)
                ocf_score, ocf_msg, ocf_color = scorer.score_cashflow(ocf)
                
                total_score = beta_score + peg_score + ev_score + ocf_score
                
                progress_bar.progress(100, text="Analysis complete!")
                progress_bar.empty()
                
                # Display main score
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 🎯 Fundamental Analysis Score")
                
                # Score display with speedometer
                col_score1, col_score2 = st.columns([1, 1])
                
                with col_score1:
                    speedometer = create_speedometer(total_score, title="Fundamental Score")
                    st.plotly_chart(speedometer, use_container_width=True)
                
                with col_score2:
                    # Interpretation card - ensure dark text
                    if total_score >= 80:
                        grade = "A"
                        grade_color = "#10b981"
                        recommendation = "STRONG BUY"
                        description = "Exceptional fundamentals across all metrics"
                    elif total_score >= 70:
                        grade = "B+"
                        grade_color = "#22c55e"
                        recommendation = "BUY"
                        description = "Strong fundamentals with good potential"
                    elif total_score >= 60:
                        grade = "B"
                        grade_color = "#84cc16"
                        recommendation = "MODERATE BUY"
                        description = "Solid fundamentals worth considering"
                    elif total_score >= 50:
                        grade = "C+"
                        grade_color = "#eab308"
                        recommendation = "HOLD"
                        description = "Mixed signals, suitable for holding"
                    elif total_score >= 40:
                        grade = "C"
                        grade_color = "#f59e0b"
                        recommendation = "WEAK HOLD"
                        description = "Below average fundamentals"
                    else:
                        grade = "D"
                        grade_color = "#ef4444"
                        recommendation = "SELL/AVOID"
                        description = "Poor fundamentals, consider alternatives"
                    
                    st.markdown(f"""
                        <div style='text-align: center; padding: 40px; background: white; border-radius: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);'>
                            <div style='font-size: 72px; font-weight: bold; color: {grade_color}; margin: 0;'>{grade}</div>
                            <div style='font-size: 24px; color: {grade_color}; margin: 10px 0; font-weight: 600;'>{recommendation}</div>
                            <div style='color: #64748b !important; margin-top: 10px;'>{description}</div>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Metrics breakdown
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 📊 Fundamental Metrics Breakdown")
                
                metrics_data = {
                    'Beta': {
                        'value': beta,
                        'score': beta_score,
                        'max_score': 25,
                        'message': beta_msg,
                        'color': beta_color,
                        'icon': '📊'
                    },
                    'PEG Ratio': {
                        'value': peg,
                        'score': peg_score,
                        'max_score': 25,
                        'message': peg_msg,
                        'color': peg_color,
                        'icon': '📈'
                    },
                    'EV/EBITDA': {
                        'value': ev_ebitda,
                        'score': ev_score,
                        'max_score': 25,
                        'message': ev_msg,
                        'color': ev_color,
                        'icon': '💰'
                    },
                    'Operating Cash Flow': {
                        'value': ocf,
                        'score': ocf_score,
                        'max_score': 25,
                        'message': ocf_msg,
                        'color': ocf_color,
                        'icon': '💵'
                    }
                }
                
                # Display metrics in a 2x2 grid
                col1, col2 = st.columns(2)
                
                for idx, (metric_name, data) in enumerate(metrics_data.items()):
                    with col1 if idx % 2 == 0 else col2:
                        value_display = "N/A"
                        if data['value'] is not None:
                            if metric_name == 'Operating Cash Flow':
                                value_display = f"${data['value']/1e9:.1f}B"
                            else:
                                value_display = f"{data['value']:.2f}"
                        
                        score_pct = (data['score'] / data['max_score']) * 100
                        
                        st.markdown(f"""
                            <div class='metric-card'>
                                <div style='display: flex; justify-content: space-between; align-items: center;'>
                                    <h4 style='margin: 0; color: #1e293b !important;'>{data['icon']} {metric_name}</h4>
                                    <span class='score-badge' style='background: {data['color']}22; color: {data['color']};'>
                                        {data['score']}/{data['max_score']}
                                    </span>
                                </div>
                                <div style='font-size: 32px; font-weight: bold; color: #1e293b !important; margin: 15px 0;'>
                                    {value_display}
                                </div>
                                <div style='background: #e5e7eb; border-radius: 10px; height: 8px; margin: 10px 0;'>
                                    <div style='background: {data['color']}; width: {score_pct}%; height: 100%; border-radius: 10px; transition: width 0.5s ease;'></div>
                                </div>
                                <p style='color: #64748b !important; font-size: 13px; margin: 10px 0 0 0;'>{data['message']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Radar chart
                if show_radar:
                    st.markdown("<br>", unsafe_allow_html=True)
                    radar_chart = create_radar_chart(metrics_data)
                    st.plotly_chart(radar_chart, use_container_width=True)
                
                # Additional metrics - FIXED
                if show_additional:
                    st.markdown("### 🔍 Additional Financial Metrics")
                    additional = analyzer.calculate_additional_metrics()
                    
                    cols = st.columns(4)
                    metric_configs = [
                        ('P/E Ratio', additional.get('pe_ratio'), '📊'),
                        ('Price/Book', additional.get('price_to_book'), '📖'),
                        ('ROE', additional.get('roe'), '💹'),
                        ('Debt/Equity', additional.get('debt_to_equity'), '⚖️')
                    ]
                    
                    for idx, (name, value, icon) in enumerate(metric_configs):
                        with cols[idx]:
                            if value:
                                if 'ROE' in name or 'Margin' in name or 'Growth' in name:
                                    display_value = f"{value*100:.1f}%"
                                else:
                                    display_value = f"{value:.2f}"
                            else:
                                display_value = "N/A"
                            
                            st.markdown(f"""
                                <div class='additional-metric-card'>
                                    <div class='metric-icon'>{icon}</div>
                                    <div class='metric-label'>{name}</div>
                                    <div class='metric-value'>{display_value}</div>
                                </div>
                            """, unsafe_allow_html=True)
                
                # Export functionality
                if export_enabled:
                    st.markdown("### 💾 Export Report")
                    
                    # Prepare export data
                    export_data = {
                        'Ticker': ticker_input,
                        'Company': company_name,
                        'Sector': sector,
                        'Total Score': total_score,
                        'Grade': grade,
                        'Recommendation': recommendation
                    }
                    
                    for metric, data in metrics_data.items():
                        export_data[f"{metric} Value"] = data['value']
                        export_data[f"{metric} Score"] = data['score']
                    
                    df_export = pd.DataFrame([export_data])
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        csv = df_export.to_csv(index=False)
                        st.download_button(
                            label="📄 Download CSV Report",
                            data=csv,
                            file_name=f"{ticker_input}_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        json_str = json.dumps(export_data, indent=2, default=str)
                        st.download_button(
                            label="📋 Download JSON Report",
                            data=json_str,
                            file_name=f"{ticker_input}_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                
            except Exception as e:
                st.error(f"❌ Error analyzing {ticker_input}: {str(e)}")
                st.info("Please verify the ticker symbol and try again. Some metrics may not be available for all stocks.")
        
        elif not ticker_input and analyze_button:
            st.warning("⚠️ Please enter a stock ticker symbol")
        
        else:
            # Welcome screen - ensure dark text on white background
            st.markdown("""
                <div class='glass-card' style='text-align: center;'>
                    <h2 style='color: #1e293b !important;'>Welcome to Stock Intelligence Hub</h2>
                    <p style='color: #64748b !important; font-size: 18px; margin: 20px 0;'>
                        Get instant fundamental analysis scores and technical charts for any US stock
                    </p>
                    <div style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px; margin-top: 30px;'>
                        <div style='padding: 20px; background: #f8fafc; border-radius: 10px;'>
                            <h3 style='color: #6366f1 !important; margin: 0;'>🎯 Smart Scoring</h3>
                            <p style='color: #64748b !important; margin: 10px 0 0 0;'>Advanced algorithms analyze key fundamentals</p>
                        </div>
                        <div style='padding: 20px; background: #f8fafc; border-radius: 10px;'>
                            <h3 style='color: #6366f1 !important; margin: 0;'>📊 Technical Charts</h3>
                            <p style='color: #64748b !important; margin: 10px 0 0 0;'>Interactive charts with key indicators</p>
                        </div>
                        <div style='padding: 20px; background: #f8fafc; border-radius: 10px;'>
                            <h3 style='color: #6366f1 !important; margin: 0;'>🚀 Real-time Data</h3>
                            <p style='color: #64748b !important; margin: 10px 0 0 0;'>Latest financial data from reliable sources</p>
                        </div>
                        <div style='padding: 20px; background: #f8fafc; border-radius: 10px;'>
                            <h3 style='color: #6366f1 !important; margin: 0;'>💡 Smart Insights</h3>
                            <p style='color: #64748b !important; margin: 10px 0 0 0;'>Clear recommendations based on analysis</p>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
