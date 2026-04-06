"""
AI Sales Forecaster + GenAI Insights Dashboard
Main Streamlit Application
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.io as pio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import custom utilities
from utils import (
    download_superstore_data,
    preprocess_data,
    create_monthly_trend_chart,
    create_quarterly_comparison_chart,
    create_category_breakdown,
    create_regional_heatmap,
    create_top_products_chart,
    create_seasonality_chart,
    ProphetForecaster,
    LSTMForecaster,
    compare_forecasts,
    get_prophet_feature_importance,
    create_feature_importance_chart,
    initialize_llm,
    generate_business_insights,
    answer_business_question,
    generate_pdf_report,
    get_forecast_metrics_table,
    format_number
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="AI Sales Forecaster + GenAI Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# THEME & CUSTOM CSS
# =============================================================================

def load_css():
    """Load custom CSS for professional appearance."""
    st.markdown("""
    <style>
    /* Main app styling */
    .main {
        background-color: var(--background-color);
    }
    
    /* Title styling */
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    
    /* Custom tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 4px 4px 0 0;
        padding: 10px 20px;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 0 4px 4px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 1rem;
        border-radius: 0 4px 4px 0;
        margin: 1rem 0;
        color: #333333;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 0 4px 4px 0;
        margin: 1rem 0;
    }
    
    /* Chart containers */
    .chart-container {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: #1f77b4 !important;
    }
    
    /* Chat styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .chat-user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    
    .chat-assistant {
        background-color: #f5f5f5;
        border-left: 4px solid #757575;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================

def render_sidebar():
    """Render sidebar with configuration options."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/artificial-intelligence.png", width=80)
        st.title("⚙️ Configuration")
        
        st.markdown("---")
        
        # Forecast Horizon Selection
        st.subheader("📅 Forecast Settings")
        forecast_horizon = st.selectbox(
            "Select Forecast Horizon:",
            options=[30, 60, 90, 180],
            format_func=lambda x: f"{x} Days ({'1 Month' if x == 30 else '2 Months' if x == 60 else '3 Months' if x == 90 else '6 Months'})",
            index=2
        )
        
        # Model Settings
        st.subheader("🤖 Model Settings")
        
        prophet_enabled = st.toggle("Enable Prophet Model", value=True)
        if prophet_enabled:
            changepoint_scale = st.slider(
                "Changepoint Sensitivity",
                min_value=0.001,
                max_value=0.5,
                value=0.05,
                help="Higher values = more flexible trend fitting"
            )
        
        lstm_enabled = st.toggle("Enable LSTM Model", value=True)
        if lstm_enabled:
            lstm_epochs = st.slider(
                "LSTM Training Epochs",
                min_value=10,
                max_value=200,
                value=50,
                step=10,
                help="More epochs = better training (but slower)"
            )
        
        st.markdown("---")
        
        # API Key Input
        st.subheader("🔑 API Configuration")
        groq_api_key = st.text_input(
            "Groq API Key:",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Get free key from https://console.groq.com"
        )
        
        if groq_api_key:
            st.success("✅ API Key configured")
        else:
            st.warning("⚠️ API Key required for GenAI chat")
        
        st.markdown("---")
        
        # About
        st.subheader("ℹ️ About")
        st.markdown("""
        **AI Sales Forecaster** v1.0
        
        Built with:
        - 🐍 Python 3.10+
        - 📊 Streamlit
        - 🔮 Prophet
        - 🧠 TensorFlow/Keras
        - 🤖 LangChain + Groq
        
        [View on GitHub](https://github.com)
        """)
        
        return {
            'forecast_horizon': forecast_horizon,
            'prophet_enabled': prophet_enabled,
            'prophet_changepoint': changepoint_scale if prophet_enabled else 0.05,
            'lstm_enabled': lstm_enabled,
            'lstm_epochs': lstm_epochs if lstm_enabled else 50,
            'groq_api_key': groq_api_key
        }

# =============================================================================
# DATA LOADING & CACHING
# =============================================================================

@st.cache_data(ttl=3600)
def load_data():
    """Load and preprocess data with caching."""
    df = download_superstore_data()
    df = preprocess_data(df)
    return df

@st.cache_resource
def get_llm(api_key: str):
    """Initialize LLM with caching."""
    if api_key and api_key != "your_groq_api_key_here":
        return initialize_llm(api_key)
    return None

# =============================================================================
# MAIN DASHBOARD SECTIONS
# =============================================================================

def render_kpi_metrics(df: pd.DataFrame):
    """Render KPI metric cards."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Sales</div>
            <div class="metric-value">{format_number(df['Sales'].sum())}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Profit</div>
            <div class="metric-value">{format_number(df['Profit'].sum())}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        profit_margin = (df['Profit'].sum() / df['Sales'].sum()) * 100
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Profit Margin</div>
            <div class="metric-value">{profit_margin:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Orders</div>
            <div class="metric-value">{df['Order ID'].nunique():,}</div>
        </div>
        """, unsafe_allow_html=True)

def render_overview_tab(df: pd.DataFrame):
    """Render Overview & EDA tab."""
    st.header("📈 Overview & Exploratory Data Analysis")
    
    # KPI Metrics
    render_kpi_metrics(df)
    
    st.markdown("---")
    
    # EDA Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Trends")
        monthly_chart = create_monthly_trend_chart(df)
        st.plotly_chart(monthly_chart, use_container_width=True)
    
    with col2:
        st.subheader("Quarterly Performance")
        quarterly_chart = create_quarterly_comparison_chart(df)
        st.plotly_chart(quarterly_chart, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Category Analysis")
        category_chart = create_category_breakdown(df)
        st.plotly_chart(category_chart, use_container_width=True)
    
    with col4:
        st.subheader("Seasonality Patterns")
        seasonality_chart = create_seasonality_chart(df)
        st.plotly_chart(seasonality_chart, use_container_width=True)
    
    # Additional EDA
    st.subheader("Regional Heatmap")
    regional_chart = create_regional_heatmap(df)
    st.plotly_chart(regional_chart, use_container_width=True)
    
    st.subheader("Top Products")
    top_products_chart = create_top_products_chart(df, top_n=15)
    st.plotly_chart(top_products_chart, use_container_width=True)
    
    # Data preview
    with st.expander("📋 View Raw Data Preview"):
        st.dataframe(df.head(100), use_container_width=True)

def render_forecast_tab(df: pd.DataFrame, config: dict):
    """Render Sales Forecast tab."""
    st.header("🔮 Sales Forecast")
    
    horizon = config['forecast_horizon']
    
    st.markdown(f"""
    <div class="info-box">
        Generating {horizon}-day forecast using Prophet and LSTM models.
        Training on historical data from {df['Order Date'].min().strftime('%Y-%m-%d')} to {df['Order Date'].max().strftime('%Y-%m-%d')}.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize models
    prophet_forecast = None
    lstm_forecast = None
    prophet_eval = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Prophet Model
    if config['prophet_enabled']:
        status_text.text("🔄 Training Prophet model...")
        
        prophet = ProphetForecaster()
        
        # Try to load existing model
        if not prophet.load():
            prophet.build_model(changepoint_prior_scale=config['prophet_changepoint'])
            prophet.fit(df)
            prophet.save()
        
        prophet_forecast = prophet.predict(periods=horizon)
        prophet_eval = prophet.evaluate(df)
        
        progress_bar.progress(50)
        
        # Prophet forecast chart
        st.subheader("Prophet Forecast")
        prophet_chart = prophet.get_forecast_chart(df)
        st.plotly_chart(prophet_chart, use_container_width=True)
        
        # Prophet components
        with st.expander("🔍 View Prophet Components"):
            components_chart = prophet.get_forecast_components()
            st.plotly_chart(components_chart, use_container_width=True)
    
    # LSTM Model
    if config['lstm_enabled']:
        status_text.text("🧠 Training LSTM model...")
        
        lstm = LSTMForecaster(sequence_length=30)
        
        # Try to load existing model
        if not lstm.load():
            with st.spinner("Training LSTM... This may take a few minutes."):
                lstm.fit(df, epochs=config['lstm_epochs'])
                lstm.save()
        
        lstm_forecast = lstm.predict(df, periods=horizon)
        
        progress_bar.progress(100)
        status_text.text("✅ Forecasting complete!")
        
        # LSTM forecast chart
        st.subheader("LSTM Forecast")
        lstm_chart = lstm.get_forecast_chart(df, lstm_forecast)
        st.plotly_chart(lstm_chart, use_container_width=True)
    
    # Model Comparison
    if prophet_forecast is not None and lstm_forecast is not None:
        st.subheader("📊 Model Comparison")
        comparison_chart = compare_forecasts(prophet_forecast, lstm_forecast, df)
        st.plotly_chart(comparison_chart, use_container_width=True)
    
    # Store forecasts in session state for other tabs
    st.session_state['prophet_forecast'] = prophet_forecast
    st.session_state['lstm_forecast'] = lstm_forecast
    st.session_state['prophet_eval'] = prophet_eval
    
    # Download buttons
    st.subheader("💾 Download Forecasts")
    col1, col2 = st.columns(2)
    
    with col1:
        if prophet_forecast is not None:
            csv = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False)
            st.download_button(
                label="📥 Download Prophet Forecast (CSV)",
                data=csv,
                file_name=f"prophet_forecast_{horizon}days.csv",
                mime="text/csv"
            )
    
    with col2:
        if lstm_forecast is not None:
            csv = lstm_forecast.to_csv(index=False)
            st.download_button(
                label="📥 Download LSTM Forecast (CSV)",
                data=csv,
                file_name=f"lstm_forecast_{horizon}days.csv",
                mime="text/csv"
            )

def render_genai_tab(df: pd.DataFrame, config: dict):
    """Render Ask Anything (GenAI) tab."""
    st.header("🤖 Ask Anything - GenAI Business Intelligence")
    
    if not config['groq_api_key']:
        st.markdown("""
        <div class="warning-box">
            <strong>API Key Required</strong><br>
            Please enter your Groq API key in the sidebar to use the GenAI assistant.
            Get a free API key at: <a href="https://console.groq.com" target="_blank">https://console.groq.com</a>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Initialize LLM
    llm = get_llm(config['groq_api_key'])
    
    if llm is None:
        st.error("❌ Failed to initialize LLM. Please check your API key.")
        return
    
    # Example questions
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "Why did sales drop in Q3 2022?",
        "What is the forecast for next month?",
        "Which category is growing fastest?",
        "What drives profit in the West region?",
        "How does seasonality affect our business?",
        "What are the top 5 products by profit margin?"
    ]
    
    cols = st.columns(3)
    for i, question in enumerate(example_questions):
        with cols[i % 3]:
            if st.button(f"🔍 {question[:40]}...", key=f"example_{i}"):
                st.session_state['current_question'] = question
    
    # Chat interface
    st.markdown("---")
    st.markdown("### 💬 Chat with Your Data")
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Display chat history
    for message in st.session_state['chat_history']:
        role = message['role']
        content = message['content']
        css_class = 'chat-user' if role == 'user' else 'chat-assistant'
        st.markdown(f'<div class="chat-message {css_class}"><strong>{role.upper()}:</strong> {content}</div>', 
                    unsafe_allow_html=True)
    
    # Input area
    current_q = st.session_state.get('current_question', '')
    question = st.text_input("Ask a question about your sales data:", 
                             value=current_q,
                             placeholder="e.g., What are the top selling products in the East region?")
    
    col1, col2 = st.columns([1, 6])
    
    with col1:
        ask_clicked = st.button("🚀 Ask", type="primary")
    
    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state['chat_history'] = []
            st.session_state['current_question'] = ''
            st.rerun()
    
    if ask_clicked and question:
        # Add user message to history
        st.session_state['chat_history'].append({'role': 'user', 'content': question})
        
        # Generate response
        with st.spinner("🤖 Analyzing your data..."):
            prophet_forecast = st.session_state.get('prophet_forecast')
            response = answer_business_question(
                question, 
                df, 
                prophet_forecast=prophet_forecast,
                llm=llm
            )
        
        # Add assistant response to history
        st.session_state['chat_history'].append({'role': 'assistant', 'content': response})
        
        # Clear current question and rerun to display
        st.session_state['current_question'] = ''
        st.rerun()

def render_comparison_tab(df: pd.DataFrame, config: dict):
    """Render Model Comparison & Recommendations tab."""
    st.header("⚖️ Model Comparison & Business Recommendations")
    
    prophet_eval = st.session_state.get('prophet_eval', {})
    prophet_forecast = st.session_state.get('prophet_forecast')
    
    # Model Performance Metrics
    st.subheader("📊 Model Performance Metrics")
    
    if prophet_eval:
        metrics_df = get_forecast_metrics_table(prophet_eval)
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        # Visual metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("MAE", f"${prophet_eval.get('MAE', 0):,.2f}")
        with col2:
            st.metric("RMSE", f"${prophet_eval.get('RMSE', 0):,.2f}")
        with col3:
            st.metric("MAPE", f"{prophet_eval.get('MAPE', 0):.2f}%")
        with col4:
            st.metric("R² Score", f"{prophet_eval.get('R2', 0):.4f}")
    else:
        st.info("ℹ️ Please run the Sales Forecast tab first to generate model metrics.")
    
    st.markdown("---")
    
    # Feature Importance
    if prophet_forecast is not None:
        st.subheader("🔍 Feature Importance (Prophet Components)")
        importance_df = get_prophet_feature_importance(prophet_forecast)
        importance_chart = create_feature_importance_chart(importance_df)
        st.plotly_chart(importance_chart, use_container_width=True)
    
    st.markdown("---")
    
    # Business Insights
    st.subheader("💡 Automated Business Insights")
    
    insights = generate_business_insights(df, prophet_eval)
    st.markdown(f"""
    <div class="success-box">
        {insights.replace(chr(10), '<br>')}
    </div>
    """, unsafe_allow_html=True)
    
    # Strategic Recommendations
    st.subheader("🎯 Strategic Recommendations")
    
    recommendations = """
    1. **INVENTORY OPTIMIZATION**
       - Use Prophet forecasts to anticipate demand spikes
       - Maintain higher stock levels during November-December (holiday season)
       - Reduce inventory during historically low-performing months

    2. **CATEGORY MANAGEMENT**
       - Focus marketing spend on top-performing categories
       - Evaluate and optimize underperforming product lines
       - Consider bundling strategies for complementary products

    3. **REGIONAL STRATEGY**
       - Allocate resources based on regional performance
       - Develop targeted campaigns for growth regions
       - Investigate and replicate success factors from top regions

    4. **PRICING & DISCOUNTS**
       - Optimize discount strategies to improve profit margins
       - Implement dynamic pricing based on seasonal patterns
       - Monitor and control high-discount transactions

    5. **FORECAST RELIABILITY**
       - Prophet model with <15% MAPE is suitable for business planning
       - Retrain models monthly with new data
       - Use ensemble approaches combining Prophet and LSTM for robust forecasts
    """
    
    st.markdown(recommendations)
    
    # PDF Report Generation
    st.markdown("---")
    st.subheader("📄 Generate Full Report")
    
    if st.button("📥 Generate & Download PDF Report", type="primary"):
        with st.spinner("Generating comprehensive PDF report..."):
            pdf_bytes = generate_pdf_report(
                df, 
                prophet_eval, 
                config['forecast_horizon'],
                insights
            )
            
            st.download_button(
                label="📥 Download Full Report (PDF)",
                data=pdf_bytes,
                file_name=f"sales_forecast_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf"
            )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    load_css()
    
    # Title
    st.markdown('<div class="main-title">📊 AI Sales Forecaster + GenAI Insights</div>', 
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Intelligent Forecasting with Natural Language Business Intelligence</div>', 
                unsafe_allow_html=True)
    
    # Sidebar configuration
    config = render_sidebar()
    
    # Load data
    with st.spinner("📥 Loading sales data..."):
        try:
            df = load_data()
            st.success(f"✅ Loaded {len(df):,} sales records from {df['Order Date'].min().strftime('%Y-%m-%d')} to {df['Order Date'].max().strftime('%Y-%m-%d')}")
        except Exception as e:
            st.error(f"❌ Error loading data: {e}")
            return
    
    # Main tabs
    tabs = st.tabs([
        "📈 Overview & EDA",
        "🔮 Sales Forecast",
        "🤖 Ask Anything",
        "⚖️ Comparison & Recommendations"
    ])
    
    with tabs[0]:
        render_overview_tab(df)
    
    with tabs[1]:
        render_forecast_tab(df, config)
    
    with tabs[2]:
        render_genai_tab(df, config)
    
    with tabs[3]:
        render_comparison_tab(df, config)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>AI Sales Forecaster v1.0 | Built with Streamlit, Prophet, TensorFlow, LangChain & Groq</p>
        <p>© 2024 - Production-Ready Data Science Portfolio Project</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
