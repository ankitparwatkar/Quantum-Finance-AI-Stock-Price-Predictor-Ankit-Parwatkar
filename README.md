# Quantum Finance AI Stock Price Predictor
```markdown
# 🌌 Quantum Stock Predictor

A futuristic web application built with Streamlit that predicts stock prices using machine learning and showcases real-time analytics with an immersive UI.

[![Streamlit App](https://quantum-finance-ai-stock-price-predictor-ankit-parwatkar.streamlit.app/)]

## 🚀 Features

- **Quantum-Themed UI**: Animated particles, cyber-card design, and glowing metrics.
- **Interactive Controls**: Synced sliders and number inputs for feature adjustments.
- **Instant Predictions**: Get stock price predictions using a Linear Regression model.
- **Live Market Trends**: Interactive historical stock price visualization with Plotly.
- **Performance Metrics**: MAE, MSE, RMSE, and R² scores to evaluate model accuracy.
- **Responsive Design**: Optimized for wide-screen layouts and mobile devices.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/ankitparwatkar/Quantum-Finance-AI-Stock-Price-Predictor-Ankit-Parwatkar.git
   cd quantum-stock-predictor
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   **`requirements.txt`**:
   ```
   streamlit
   numpy
   pandas
   plotly
   scikit-learn
   streamlit-extras
   ```

3. **Add stock data**:  
   Place your `ADANIPORTS.csv` file in the project root (or modify the code to use your dataset).

## 🖥️ Usage

1. **Run the app**:
   ```bash
   streamlit run sreamlit.py
   ```

2. **Adjust input values** using sliders or number inputs in the **Control Panel** (sidebar).

3. Click **🚀 PREDICT NOW** to see the forecasted stock price and performance metrics.

4. Explore:
   - **📈 LIVE MARKET TRENDS**: Historical closing price chart.
   - **📊 REAL-TIME MODEL ANALYTICS**: Model evaluation metrics.

## 🛠️ Technologies Used

- **Frontend**: Streamlit, Plotly, Custom CSS animations
- **Backend**: scikit-learn (LR model), pandas, numpy
- **Styling**: Cyber-themed UI with gradient effects and particles

## 📝 Notes

- The "Quantum" branding is thematic; the model uses classical ML (not quantum computing).
- Ensure your dataset matches the expected format (columns: `Date, Open, High, Low, Last, Close` etc.).
- Model trained on `ADANIPORTS.csv` (replace with your data for other stocks).

## 👥 Contributing

Contributions are welcome!  
1. Fork the repository.  
2. Create a feature branch (`git checkout -b feature/amazing-feature`).  
3. Commit changes (`git commit -m 'Add amazing feature'`).  
4. Push to the branch (`git push origin feature/amazing-feature`).  
5. Open a Pull Request.

---

🔗 **Connect with Me**  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ankit_Parwatkar-blue?style=flat&logo=linkedin)](https://linkedin.com/in/ankitparwatkar)  
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/ankitparwatkar)

*"The stock market is a device for transferring money from the impatient to the patient." - Warren Buffett*
```
