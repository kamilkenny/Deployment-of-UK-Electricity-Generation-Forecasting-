# -*- coding: utf-8 -*-
"""GB Electricity Generation Forecasting Dashboard.

Streamlit deployment for the univariate LSTM model using:
- uni_lstm_model.h5
- uni_scaler.pkl
- load_hindcast_data.csv
- kamil_profile.png

The application supports historical hindcasting and recursive future forecasting.
"""

from pathlib import Path
import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model


st.set_page_config(
    page_title="GB Electricity Generation Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "uni_lstm_model.h5"
SCALER_PATH = BASE_DIR / "uni_scaler.pkl"
DATA_PATH = BASE_DIR / "load_hindcast_data.csv"
PROFILE_PATH = BASE_DIR / "kamil_profile.png"

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(21, 118, 110, 0.08), transparent 30rem),
                linear-gradient(180deg, #f7faf9 0%, #f1f6f5 100%);
            color: #183630;
        }
        .block-container {
            max-width: 1220px;
            padding-top: 1.15rem;
            padding-bottom: 2rem;
        }
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header[data-testid="stHeader"] {
            background: rgba(247, 250, 249, 0.90);
        }
        .hero {
            background: linear-gradient(135deg, #0d3935 0%, #155f57 52%, #126b85 100%);
            border-radius: 20px;
            padding: 1.75rem 1.95rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 34px rgba(12, 59, 53, 0.13);
        }
        .hero-kicker {
            margin: 0 0 0.42rem 0;
            color: #cbeae5;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.09em;
            text-transform: uppercase;
        }
        .hero h1 {
            margin: 0;
            color: #ffffff;
            font-size: clamp(1.8rem, 3vw, 2.7rem);
            line-height: 1.1;
            letter-spacing: -0.025em;
        }
        .hero p {
            max-width: 940px;
            margin: 0.72rem 0 0 0;
            color: #e8f4f1;
            font-size: 0.98rem;
            line-height: 1.6;
        }
        .version {
            color: #78908a;
            font-size: 0.76rem;
            margin: -0.35rem 0 0.9rem 0.1rem;
        }
        .info-card {
            min-height: 112px;
            background: rgba(255, 255, 255, 0.96);
            border: 1px solid #dbe7e3;
            border-radius: 16px;
            padding: 1rem 1.05rem;
            box-shadow: 0 6px 20px rgba(18, 63, 57, 0.055);
        }
        .info-label {
            color: #6a7f7a;
            font-size: 0.71rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }
        .info-value {
            color: #123e38;
            font-size: 1.16rem;
            font-weight: 800;
            line-height: 1.2;
        }
        .info-note {
            color: #70817d;
            font-size: 0.81rem;
            line-height: 1.38;
            margin-top: 0.34rem;
        }
        .section-title {
            color: #173f39;
            font-size: 1.34rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            margin-top: 0.72rem;
            margin-bottom: 0.14rem;
        }
        .section-copy {
            color: #677d78;
            font-size: 0.91rem;
            line-height: 1.55;
            margin-top: 0;
            margin-bottom: 0.88rem;
        }
        .pillar {
            min-height: 100px;
            background: #ffffff;
            border: 1px solid #dde8e5;
            border-radius: 14px;
            padding: 0.92rem 1rem;
        }
        .pillar-title {
            color: #164f48;
            font-size: 0.93rem;
            font-weight: 800;
            margin-bottom: 0.27rem;
        }
        .pillar-copy {
            color: #6a7c78;
            font-size: 0.81rem;
            line-height: 1.44;
        }
        div.stButton > button {
            width: 100%;
            border: 0;
            border-radius: 12px;
            padding: 0.72rem 1rem;
            background: linear-gradient(90deg, #155f57, #126b85);
            color: #ffffff;
            font-weight: 800;
            box-shadow: 0 6px 18px rgba(18, 107, 133, 0.17);
        }
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dce8e4;
            border-radius: 15px;
            padding: 0.92rem 1rem;
            box-shadow: 0 4px 16px rgba(18, 63, 57, 0.05);
        }
        div[data-testid="stMetricLabel"] { color: #627873; }
        div[data-testid="stMetricValue"] { color: #143f39; }
        div[data-testid="stAlert"] { border-radius: 12px; }
        div.stDownloadButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid #155f57;
            background: #ffffff;
            color: #155f57;
            font-weight: 750;
        }
        .app-footer {
            margin-top: 1.7rem;
            padding-top: 1rem;
            border-top: 1px solid #dce7e4;
            text-align: center;
            color: #73847f;
            font-size: 0.77rem;
            line-height: 1.5;
        }
        @media (max-width: 768px) {
            .block-container {
                padding-top: 0.8rem;
                padding-left: 1rem;
                padding-right: 1rem;
            }
            .hero {
                padding: 1.3rem 1.2rem;
                border-radius: 16px;
            }
            .info-card, .pillar { min-height: auto; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def require_file(path: Path, label: str):
    if not path.exists():
        st.error(f"{label} was not found in the deployment repository: {path.name}")
        st.stop()


require_file(MODEL_PATH, "LSTM model")
require_file(SCALER_PATH, "Scaler")
require_file(DATA_PATH, "Hindcast dataset")


@st.cache_resource(show_spinner=False)
def load_resources():
    lstm_model = load_model(str(MODEL_PATH), compile=False)
    fitted_scaler = joblib.load(SCALER_PATH)
    return lstm_model, fitted_scaler


@st.cache_data(show_spinner=False)
def load_hindcast_data():
    return pd.read_csv(DATA_PATH)


model, scaler = load_resources()
raw_df = load_hindcast_data()


def normalise_name(name):
    return (
        str(name).strip().lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("[", "")
        .replace("]", "")
        .replace("(", "")
        .replace(")", "")
    )


def detect_datetime_column(df):
    preferred = {
        "datetime", "date_time", "timestamp", "date",
        "settlement_datetime", "settlement_date",
    }
    for col in df.columns:
        if normalise_name(col) in preferred:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                return col
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        if parsed.notna().mean() >= 0.9:
            return col
    return None


def detect_generation_column(df, datetime_col):
    preferred = {
        "generation", "generation_mw", "total_generation",
        "total_generation_mw", "load", "load_mw",
        "actual", "actual_mw", "value",
    }
    for col in df.columns:
        if col == datetime_col:
            continue
        if normalise_name(col) in preferred:
            numeric = pd.to_numeric(df[col], errors="coerce")
            if numeric.notna().mean() >= 0.8:
                return col
    numeric_candidates = []
    for col in df.columns:
        if col == datetime_col:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.95:
            numeric_candidates.append(col)
    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    return None


datetime_col = detect_datetime_column(raw_df)
generation_col = detect_generation_column(raw_df, datetime_col)

if datetime_col is None:
    st.error(
        "The app could not identify the datetime column in load_hindcast_data.csv. "
        f"Available columns: {list(raw_df.columns)}"
    )
    st.stop()

if generation_col is None:
    st.error(
        "The app could not identify the generation column in load_hindcast_data.csv. "
        f"Available columns: {list(raw_df.columns)}"
    )
    st.stop()

data = raw_df[[datetime_col, generation_col]].copy()
data.columns = ["Datetime", "Actual_MW"]
data["Datetime"] = pd.to_datetime(data["Datetime"], errors="coerce")
data["Actual_MW"] = pd.to_numeric(data["Actual_MW"], errors="coerce")
data = (
    data.dropna()
    .sort_values("Datetime")
    .drop_duplicates(subset=["Datetime"], keep="last")
    .reset_index(drop=True)
)

if len(data) < 3:
    st.error("The hindcast dataset does not contain enough valid observations.")
    st.stop()


def infer_lookback(lstm_model):
    shape = lstm_model.input_shape
    if isinstance(shape, list):
        shape = shape[0]
    if shape is not None and len(shape) >= 3 and shape[1] is not None:
        return int(shape[1])
    return 24


LOOKBACK = infer_lookback(model)

if len(data) <= LOOKBACK:
    st.error(
        f"The dataset contains {len(data)} observations, but the model requires "
        f"a lookback window of {LOOKBACK} observations."
    )
    st.stop()

time_steps = data["Datetime"].diff().dropna()
median_step = time_steps.median()
if pd.isna(median_step) or median_step <= pd.Timedelta(0):
    median_step = pd.Timedelta(minutes=30)

data_end = data["Datetime"].max()


def scale_series(values):
    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    return scaler.transform(arr).reshape(-1)


def inverse_series(values):
    arr = np.asarray(values, dtype=float).reshape(-1, 1)
    return scaler.inverse_transform(arr).reshape(-1)


def model_predict_scaled(windows):
    arr = np.asarray(windows, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    prediction = model.predict(arr, verbose=0)
    return np.asarray(prediction).reshape(-1)


def build_hindcast(start_dt, end_dt):
    mask = (data["Datetime"] >= start_dt) & (data["Datetime"] <= end_dt)
    target_indices = data.index[mask].to_numpy()
    target_indices = target_indices[target_indices >= LOOKBACK]
    if len(target_indices) == 0:
        return pd.DataFrame()

    scaled_actual = scale_series(data["Actual_MW"].values)
    windows = np.stack(
        [scaled_actual[i - LOOKBACK:i] for i in target_indices]
    )
    predicted_scaled = model_predict_scaled(windows)
    predicted_mw = inverse_series(predicted_scaled)

    result = pd.DataFrame(
        {
            "Datetime": data.loc[target_indices, "Datetime"].values,
            "Actual_MW": data.loc[target_indices, "Actual_MW"].values,
            "Predicted_MW": predicted_mw,
        }
    )
    result["Residual_MW"] = result["Predicted_MW"] - result["Actual_MW"]
    return result


def build_future_forecast(start_dt, end_dt):
    first_future_time = data_end + median_step
    if end_dt < first_future_time:
        return pd.DataFrame()

    start_dt = max(start_dt, first_future_time)
    full_index = pd.date_range(
        start=first_future_time,
        end=end_dt,
        freq=median_step,
    )
    if len(full_index) == 0:
        return pd.DataFrame()

    seed_actual = data["Actual_MW"].iloc[-LOOKBACK:].values
    window = list(scale_series(seed_actual))
    predictions_scaled = []

    for _ in full_index:
        x = np.asarray(window[-LOOKBACK:], dtype=np.float32).reshape(
            1, LOOKBACK, 1
        )
        next_scaled = float(model_predict_scaled(x)[0])
        predictions_scaled.append(next_scaled)
        window.append(next_scaled)

    predictions_mw = inverse_series(predictions_scaled)
    result = pd.DataFrame(
        {"Datetime": full_index, "Forecast_MW": predictions_mw}
    )
    return result[
        (result["Datetime"] >= start_dt)
        & (result["Datetime"] <= end_dt)
    ].reset_index(drop=True)


def regression_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = predicted - actual
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    bias = np.mean(errors)
    denominator = np.sum((actual - np.mean(actual)) ** 2)
    r2 = np.nan if denominator == 0 else 1 - (
        np.sum((actual - predicted) ** 2) / denominator
    )
    return mae, rmse, bias, r2


def energy_gwh(series_mw):
    hours = median_step.total_seconds() / 3600
    return float(np.sum(series_mw) * hours / 1000)


st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Great Britain · Electricity System Intelligence</div>
        <h1>GB Electricity Generation Forecasting</h1>
        <p>
            A deployed univariate LSTM forecasting application for exploring
            Great Britain's electricity generation profile using historical
            NESO generation data. The interface supports historical hindcasting
            and recursive future forecasting from the trained model.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="version">Interface version: GB LSTM Forecast Dashboard v3.0</div>',
    unsafe_allow_html=True,
)

profile_col, card_1, card_2, card_3 = st.columns([1.05, 1, 1, 1])

with profile_col:
    if PROFILE_PATH.exists():
        try:
            profile_img = Image.open(PROFILE_PATH)
            st.image(profile_img, width=220, caption="Kamil Kehinde · Lead Modeller")
        except Exception:
            st.info("**Kamil Ridwan Kehinde**\n\nLead Modeller")
    else:
        st.info("**Kamil Ridwan Kehinde**\n\nLead Modeller")

with card_1:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Forecast Model</div>
            <div class="info-value">Univariate LSTM</div>
            <div class="info-note">
                Sequence model using a {LOOKBACK}-observation historical lookback.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card_2:
    resolution_minutes = int(median_step.total_seconds() / 60)
    resolution_text = (
        f"{resolution_minutes} Minutes"
        if resolution_minutes < 60
        else f"{resolution_minutes / 60:g} Hours"
    )
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-label">Data Resolution</div>
            <div class="info-value">{resolution_text}</div>
            <div class="info-note">
                Forecast cadence inferred directly from the deployed historical dataset.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card_3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Operational Modes</div>
            <div class="info-value">Forecast + Hindcast</div>
            <div class="info-note">
                Compare model estimates with history or generate future trajectories.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="section-title">Why Generation Forecasting Matters</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="section-copy">
        Short-term generation forecasting supports understanding of system
        behaviour as Great Britain integrates variable renewable generation,
        manages changing demand patterns and maintains secure system operation.
    </div>
    """,
    unsafe_allow_html=True,
)

p1, p2, p3 = st.columns(3)

with p1:
    st.markdown(
        """
        <div class="pillar">
            <div class="pillar-title">⚡ Grid Balancing</div>
            <div class="pillar-copy">
                Anticipating generation conditions supports the balancing of
                electricity supply and demand.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p2:
    st.markdown(
        """
        <div class="pillar">
            <div class="pillar-title">🌱 Renewable Integration</div>
            <div class="pillar-copy">
                Forecasting provides context for managing greater volumes of
                variable wind and solar generation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        """
        <div class="pillar">
            <div class="pillar-title">📊 System Planning</div>
            <div class="pillar-copy">
                Forecast profiles provide analytical evidence for planning,
                operational readiness and energy-system studies.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="section-title">Model Workspace</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="section-copy">
        Use Hindcast to compare the model with historical observations, or
        Future Forecast to recursively extend the generation series beyond the
        latest observation available in the deployed dataset.
    </div>
    """,
    unsafe_allow_html=True,
)

hindcast_tab, forecast_tab = st.tabs(
    ["📚 Historical Hindcast", "🔮 Future Forecast"]
)

with hindcast_tab:
    earliest_hindcast = data["Datetime"].iloc[LOOKBACK].date()
    latest_hindcast = data_end.date()
    default_hindcast_end = latest_hindcast
    default_hindcast_start = max(
        earliest_hindcast,
        default_hindcast_end - datetime.timedelta(days=3),
    )

    selected_hindcast = st.date_input(
        "Select historical start and end dates",
        value=(default_hindcast_start, default_hindcast_end),
        min_value=earliest_hindcast,
        max_value=latest_hindcast,
        key="hindcast_range",
    )

    run_hindcast = st.button("📈 Run Hindcast Analysis", key="run_hindcast")

    if run_hindcast:
        if not (isinstance(selected_hindcast, tuple) and len(selected_hindcast) == 2):
            st.error("Please select both a start and an end date.")
        else:
            start_d, end_d = selected_hindcast
            start_dt = pd.Timestamp(start_d)
            end_dt = pd.Timestamp(end_d) + pd.Timedelta(days=1) - median_step

            with st.spinner("Running historical model inference..."):
                hindcast_df = build_hindcast(start_dt, end_dt)

            if hindcast_df.empty:
                st.warning("No valid hindcast observations are available for the selected period.")
            else:
                mae, rmse, bias, r2 = regression_metrics(
                    hindcast_df["Actual_MW"],
                    hindcast_df["Predicted_MW"],
                )

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("MAE", f"{mae:,.0f} MW")
                k2.metric("RMSE", f"{rmse:,.0f} MW")
                k3.metric("Forecast Bias", f"{bias:,.0f} MW")
                k4.metric("R²", "N/A" if np.isnan(r2) else f"{r2:.4f}")

                st.markdown(
                    '<div class="section-title">Actual vs Modelled Generation</div>',
                    unsafe_allow_html=True,
                )

                chart_df = hindcast_df.set_index("Datetime")[
                    ["Actual_MW", "Predicted_MW"]
                ].rename(
                    columns={
                        "Actual_MW": "Actual Generation",
                        "Predicted_MW": "Modelled Generation",
                    }
                )
                st.line_chart(chart_df)

                preview = hindcast_df.rename(
                    columns={
                        "Actual_MW": "Actual Generation [MW]",
                        "Predicted_MW": "Modelled Generation [MW]",
                        "Residual_MW": "Residual [MW]",
                    }
                )
                st.dataframe(preview.head(30), height=330)

                st.download_button(
                    "📥 Download Hindcast Results CSV",
                    data=hindcast_df.to_csv(index=False),
                    file_name=f"GB_Hindcast_{start_d}_{end_d}.csv",
                    mime="text/csv",
                    key="download_hindcast",
                )

with forecast_tab:
    first_future_date = (data_end + median_step).date()
    default_forecast_start = first_future_date
    default_forecast_end = first_future_date + datetime.timedelta(days=3)

    selected_forecast = st.date_input(
        "Select future start and end dates",
        value=(default_forecast_start, default_forecast_end),
        min_value=first_future_date,
        key="forecast_range",
    )

    st.caption(
        f"Latest historical observation available to the model: "
        f"{data_end.strftime('%d %b %Y %H:%M')}"
    )

    run_forecast = st.button("🚀 Run Future Forecast", key="run_forecast")

    if run_forecast:
        if not (isinstance(selected_forecast, tuple) and len(selected_forecast) == 2):
            st.error("Please select both a start and an end date.")
        else:
            start_d, end_d = selected_forecast
            start_dt = pd.Timestamp(start_d)
            end_dt = pd.Timestamp(end_d) + pd.Timedelta(days=1) - median_step

            with st.spinner("Generating recursive LSTM forecast..."):
                forecast_df = build_future_forecast(start_dt, end_dt)

            if forecast_df.empty:
                st.warning("No future forecast observations were generated for the selected period.")
            else:
                avg_mw = forecast_df["Forecast_MW"].mean()
                peak_mw = forecast_df["Forecast_MW"].max()
                min_mw = forecast_df["Forecast_MW"].min()
                total_gwh = energy_gwh(forecast_df["Forecast_MW"])

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Average Generation", f"{avg_mw:,.0f} MW")
                k2.metric("Predicted Peak", f"{peak_mw:,.0f} MW")
                k3.metric("Minimum Generation", f"{min_mw:,.0f} MW")
                k4.metric("Energy Volume", f"{total_gwh:,.2f} GWh")

                st.markdown(
                    '<div class="section-title">Future Generation Forecast</div>',
                    unsafe_allow_html=True,
                )

                chart_df = forecast_df.set_index("Datetime")[
                    ["Forecast_MW"]
                ].rename(columns={"Forecast_MW": "Forecast Generation"})
                st.line_chart(chart_df)

                preview = forecast_df.rename(
                    columns={"Forecast_MW": "Forecast Generation [MW]"}
                )
                st.dataframe(preview.head(30), height=330)

                st.download_button(
                    "📥 Download Future Forecast CSV",
                    data=forecast_df.to_csv(index=False),
                    file_name=f"GB_LSTM_Forecast_{start_d}_{end_d}.csv",
                    mime="text/csv",
                    key="download_forecast",
                )

st.markdown(
    """
    <div class="app-footer">
        © 2026 Energy Analytics Portfolio · Kamil Ridwan Kehinde<br>
        Data context: National Energy System Operator (NESO) historical generation data<br>
        Research and analytical demonstration, not an operational NESO forecasting service
    </div>
    """,
    unsafe_allow_html=True,
)
