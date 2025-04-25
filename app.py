# app.py
"""
Streamlit Time-Series Forecaster
--------------------------------
• Upload **any** CSV that contains **one date column** and **one or more numeric columns**.  
• Pick the series, pick the horizon, click **Run forecast**.  
• The app will train Facebook Prophet (or fall back to auto-ARIMA if Prophet is absent),  
  plot the results, and let you download a CSV of the forecast.
"""

import io
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ────────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

@st.cache_resource(show_spinner=False)
def train_prophet(df: pd.DataFrame, date_col: str, value_col: str):
    """Fit a Prophet model and return it."""
    from prophet import Prophet

    tmp = df[[date_col, value_col]].copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
    tmp = tmp.dropna(subset=[date_col, value_col])
    tmp = tmp.rename(columns={date_col: "ds", value_col: "y"})

    model = Prophet()
    model.fit(tmp)
    return model

@st.cache_resource(show_spinner=False)
def train_auto_arima(ts):
    """Fit an auto-ARIMA model and return it (fallback when Prophet unavailable)."""
    from pmdarima import auto_arima

    model = auto_arima(
        ts,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
        seasonal=False,
    )
    return model


def forecast_prophet(model, periods: int, freq: str) -> pd.DataFrame:
    future = model.make_future_dataframe(periods=periods, freq=freq)
    fc = model.predict(future)
    return fc[["ds", "yhat", "yhat_lower", "yhat_upper"]]


def forecast_auto_arima(model, periods: int, last_date: pd.Timestamp, freq: str):
    fc, conf = model.predict(periods, return_conf_int=True)
    idx = pd.date_range(last_date + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq)
    out = (
        pd.DataFrame(
            {
                "ds": idx,
                "yhat": fc,
                "yhat_lower": conf[:, 0],
                "yhat_upper": conf[:, 1],
            }
        )
        .reset_index(drop=True)
    )
    return out


def plot_forecast(history_df, date_col, value_col, fc_df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history_df[date_col], y=history_df[value_col],
                             name="Actual", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat"],
                             name="Forecast", mode="lines"))
    fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat_upper"],
                             name="Upper CI", mode="lines", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=fc_df["ds"], y=fc_df["yhat_lower"],
                             name="Lower CI", mode="lines", line=dict(dash="dash")))
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


# ────────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Time-Series Forecaster", layout="wide")
st.title("🕒 Universal Time-Series Forecaster")

uploaded = st.file_uploader("📂 Upload a CSV file", type="csv")

if uploaded:
    df = load_csv(uploaded)
    st.subheader("Preview")
    st.dataframe(df.head())

    # Identify candidate columns
    date_candidates = df.select_dtypes(include=["datetime64[ns]", "object"]).columns.tolist()
    numeric_candidates = df.select_dtypes(include=["number"]).columns.tolist()

    if not date_candidates:
        st.error("❌ No date-like columns detected. Ensure your CSV has a date column.")
        st.stop()
    if not numeric_candidates:
        st.error("❌ No numeric columns detected. Ensure your CSV has at least one number column.")
        st.stop()

    date_col = st.selectbox("Date column", date_candidates)
    value_col = st.selectbox("Numeric series to forecast", numeric_candidates)
    horizon = st.number_input("Periods to forecast (steps into the future)",
                              min_value=1, max_value=365 * 5, value=30)
    freq = st.selectbox("Data frequency", options=["D", "W", "M", "H"],
                        help="D=Daily, W=Weekly, M=Monthly, H=Hourly")

    if st.button("🚀 Run forecast"):
        with st.spinner("Training model… this might take a moment ⏳"):
            try:
                model = train_prophet(df, date_col, value_col)
                fc_df = forecast_prophet(model, periods=horizon, freq=freq)
                engine = "Prophet"
            except ModuleNotFoundError:
                st.warning("Prophet not installed — falling back to auto-ARIMA. "
                           "Add `prophet` to requirements.txt to use Prophet.")
                ts = pd.Series(df[value_col].values,
                               index=pd.to_datetime(df[date_col], errors="coerce")).dropna()
                model = train_auto_arima(ts)
                fc_df = forecast_auto_arima(model, periods=horizon,
                                            last_date=ts.index[-1], freq=freq)
                engine = "auto-ARIMA"
            except Exception as e:
                st.exception(e)
                st.stop()

        st.success(f"Forecast complete using **{engine}**!")

        # Plot
        st.plotly_chart(plot_forecast(df, date_col, value_col, fc_df),
                        use_container_width=True)

        # Download button
        csv_bytes = fc_df.to_csv(index=False).encode()
        st.download_button("⬇️ Download forecast CSV",
                           data=csv_bytes,
                           file_name="forecast.csv",
                           mime="text/csv")
