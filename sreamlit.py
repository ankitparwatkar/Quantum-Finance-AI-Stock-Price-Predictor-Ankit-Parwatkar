# Quantum Stock Predictor
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_extras.let_it_rain import rain
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Must be the FIRST Streamlit command
st.set_page_config(
    page_title="Quantum Stock Predictor",
    layout="wide",
    page_icon="🌌"
)

# Custom CSS with animations
st.markdown("""
<style>
    @keyframes particle-move {
        0% { transform: translateY(0) translateX(0); opacity: 1; }
        100% { transform: translateY(-100vh) translateX(100vw); opacity: 0; }
    }

    .particle {
        position: fixed;
        pointer-events: none;
        animation: particle-move 1.5s linear infinite;
        background: radial-gradient(circle, #4FD3C4 20%, transparent 80%);
        width: 5px;
        height: 5px;
        border-radius: 50%;
    }

    .cyber-card {
        background: rgba(16, 18, 27, 0.8);
        border-radius: 20px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        padding: 2rem;
        position: relative;
        overflow: hidden;
    }

    .stSlider .thumb {
        background: #4FD3C4 !important;
        box-shadow: 0 0 15px #4FD3C466 !important;
    }

    .metric-card {
        animation: glow 1s infinite;
        transition: transform 0.2s;
    }

    @keyframes glow {
        0% { box-shadow: 0 0 5px #4FD3C4; }
        50% { box-shadow: 0 0 15px #4FD3C4; }
        100% { box-shadow: 0 0 5px #4FD3C4; }
    }
</style>
""", unsafe_allow_html=True)

# Add animated particles
st.markdown("""
<div class="particle" style="top: 20%; left: 10%"></div>
<div class="particle" style="top: 40%; left: 30%"></div>
<div class="particle" style="top: 60%; left: 50%"></div>
<div class="particle" style="top: 80%; left: 70%"></div>
""", unsafe_allow_html=True)

# Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv("ADANIPORTS.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df

# Model Training
@st.cache_resource
def train_model():
    df = load_data()
    x = df.iloc[:, 3:8].values
    y = df.iloc[:, 8].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
    model = SVR(kernel='poly', degree=2, C=2)
    model.fit(x_train, y_train)
    return model, x_test, y_test

model, x_test, y_test = train_model()
df = load_data()

# Main Header
st.markdown("""
<div style="text-align: center; padding: 4rem 0;">
    <h1 style="font-size: 4rem; color: #4FD3C4; text-shadow: 0 0 30px #4FD3C4;">
        QUANTUM FINANCE AI
    </h1>
    <p style="color: #fff; font-size: 1.2rem; letter-spacing: 2px;">
        Instant Stock Price Predictions with Quantum AI
    </p>
</div>
""", unsafe_allow_html=True)

# Control Panel
with st.sidebar:
    with st.expander("⚙️ CONTROL PANEL", expanded=True):
        features = ['Prev Close', 'Open', 'High', 'Low', 'Last']
        
        # Initialize session state
        for i in range(5):
            if f"slider_{i}" not in st.session_state:
                st.session_state[f"slider_{i}"] = float(df.iloc[-1, 3+i])
            if f"num_{i}" not in st.session_state:
                st.session_state[f"num_{i}"] = float(df.iloc[-1, 3+i])
        
        inputs = {}
        for i, feat in enumerate(features):
            col1, col2 = st.columns([3,1])
            with col1:
                # Slider with bidirectional sync
                slider_val = st.slider(
                    feat,
                    min_value=0.0,
                    max_value=3000.0,
                    value=st.session_state[f"slider_{i}"],
                    key=f"slider_{i}",
                    step=10.0,
                    on_change=lambda i=i: st.session_state.update({
                        f"num_{i}": st.session_state[f"slider_{i}"]
                    })
                )
                inputs[feat] = slider_val
            with col2:
                # Number input with bidirectional sync
                num_val = st.number_input(
                    "Value",
                    value=st.session_state[f"num_{i}"],
                    key=f"num_{i}",
                    on_change=lambda i=i: st.session_state.update({
                        f"slider_{i}": st.session_state[f"num_{i}"]
                    }),
                    step=10.0
                )

        if st.button("🚀 PREDICT NOW", use_container_width=True):
            rain(emoji="✨", font_size=20, falling_speed=7, animation_length=0.5)
            st.session_state.predict = True

# Main Content
if 'predict' in st.session_state:
    input_values = [inputs[feat] for feat in features]
    result = model.predict([input_values])[0]
    last_close = df.iloc[-1]['Close']
    change = ((result - last_close) / last_close) * 100

    # Prediction Display
    st.markdown(f"""
    <div class="cyber-card" style="margin: 2rem 0; text-align: center;">
        <h2 style="color: #4FD3C4;">INSTANT PREDICTION RESULT</h2>
        <h1 style="font-size: 4rem; color: {"#4FD3C4" if result >= last_close else "#ff4b4b"};">
            ${result:,.2f} {"📈" if result >= last_close else "📉"}
        </h1>
        <div style="color: {"#4FD3C4" if change >=0 else "#ff4b4b"}; font-size: 1.5rem;">
            {change:+.2f}% vs Last Close (${last_close:,.2f})
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Historical Data Visualization
    with st.expander("📈 LIVE MARKET TRENDS", expanded=True):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Close'],
            mode='lines',
            line=dict(color='#4FD3C4', width=2),
            fill='tozeroy',
            fillcolor='rgba(79,211,196,0.1)'
        ))
        fig.update_layout(
            template='plotly_dark',
            margin=dict(l=20, r=20, t=40, b=20),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # Performance Metrics
    st.markdown("## 📊 REAL-TIME MODEL ANALYTICS")
    y_pred = model.predict(x_test)
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R² Score': model.score(x_test, y_test)
    }

    cols = st.columns(4)
    for col, (name, value) in zip(cols, metrics.items()):
        with col:
            st.markdown(f"""
            <div class="metric-card" style="
                background: rgba(16, 18, 27, 0.8);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1rem;
                text-align: center;">
                <div style="font-size: 1.2rem; color: #4FD3C4;">
                    {name}
                </div>
                <div style="font-size: 2rem; color: white; margin: 0.5rem 0;">
                    {"{:.2f}".format(value) if name != 'R² Score' else "{:.2%}".format(value)}
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="padding: 3rem 0; text-align: center;">
    <div style="display: flex; justify-content: center; gap: 2rem; margin-bottom: 1.5rem;">
        <a href="https://linkedin.com" target="_blank" style="color: #4FD3C4; text-decoration: none; padding: 0.5rem 1rem; border: 1px solid #4FD3C4; border-radius: 8px; transition: all 0.2s ease;" 
           onmouseover="this.style.transform='scale(1.05)'; this.style.background='rgba(79,211,196,0.1)'" 
           onmouseout="this.style.transform='scale(1)'; this.style.background='transparent'">
            👔 LinkedIn
        </a>
        <a href="https://github.com" target="_blank" style="color: #4FD3C4; text-decoration: none; padding: 0.5rem 1rem; border: 1px solid #4FD3C4; border-radius: 8px; transition: all 0.2s ease;" 
           onmouseover="this.style.transform='scale(1.05)'; this.style.background='rgba(79,211,196,0.1)'" 
           onmouseout="this.style.transform='scale(1)'; this.style.background='transparent'">
            💻 GitHub
        </a>
    </div>
    <p style="color: rgba(255,255,255,0.5);">
        © 2023 Quantum Finance AI • Ultra-Fast Predictions • v3.1
    </p>
</div>
""", unsafe_allow_html=True)