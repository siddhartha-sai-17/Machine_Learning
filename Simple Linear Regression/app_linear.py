import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Linear Regression",
    layout="centered"
)

# ---------------- LOAD CSS ----------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# ---------------- TITLE ----------------
st.markdown("""
<div class="card">
    <h1>Linear Regression</h1>
    <p>Predict <b>Tip Amount</b> from <b>Total Bill</b> using Linear Regression.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREPARE DATA ----------------
X = df[["total_bill"]]
y = df["tip"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------- TRAIN MODEL ----------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# ---------------- METRICS ----------------
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

n = len(y_test)
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# ---------------- VISUALIZATION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6)

x_line = np.linspace(df["total_bill"].min(), df["total_bill"].max(), 100).reshape(-1, 1)
x_line_scaled = scaler.transform(x_line)
y_line = model.predict(x_line_scaled)

ax.plot(x_line, y_line, color="red")
ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip ($)")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PERFORMANCE METRICS ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R²", f"{r2:.3f}")
c4.metric("Adj R²", f"{adj_r2:.3f}")

st.markdown('</div>', unsafe_allow_html=True)

# ---------------- MODEL DETAILS ----------------
st.markdown(f"""
<div class="card">
    <h3>Model Parameters</h3>
    <p>
        <b>Coefficient:</b> {model.coef_[0]:.3f}<br>
        <b>Intercept:</b> {model.intercept_:.3f}
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip")

bill = st.slider(
    "Select Total Bill ($)",
    float(df.total_bill.min()),
    float(df.total_bill.max()),
    30.0
)

bill_scaled = scaler.transform([[bill]])
tip = model.predict(bill_scaled)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${tip:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
