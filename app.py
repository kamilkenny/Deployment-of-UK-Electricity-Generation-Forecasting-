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
import base64

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
            background: #f4f1e8;
            color: #17223b;
        }

        .block-container {
            max-width: 1180px;
            padding-top: 0.9rem;
            padding-bottom: 1.8rem;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        header[data-testid="stHeader"] {
            background: rgba(244, 241, 232, 0.94);
        }

        /* Distinctive navy + amber energy-system identity */
        .top-strip {
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            align-items: center;
            background: #17223b;
            color: #ffffff;
            border-radius: 16px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.85rem;
        }

        .top-strip-left {
            min-width: 0;
        }

        .top-strip-kicker {
            color: #f0c36b;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .top-strip h1 {
            margin: 0;
            font-size: clamp(1.65rem, 2.8vw, 2.45rem);
            line-height: 1.08;
            letter-spacing: -0.025em;
        }

        .top-strip p {
            margin: 0.48rem 0 0 0;
            color: #d8dfeb;
            font-size: 0.9rem;
            line-height: 1.5;
            max-width: 760px;
        }

        .status-chip {
            flex: 0 0 auto;
            background: #f0a202;
            color: #17223b;
            border-radius: 999px;
            padding: 0.55rem 0.78rem;
            font-size: 0.76rem;
            font-weight: 800;
            white-space: nowrap;
        }

        .signal-card {
            min-height: 92px;
            background: #fffdf7;
            border: 1px solid #ded8ca;
            border-top: 4px solid #2d5f8b;
            border-radius: 12px;
            padding: 0.82rem 0.9rem;
        }

        .signal-card.amber {
            border-top-color: #f0a202;
        }

        .signal-card.teal {
            border-top-color: #2a7f8e;
        }

        .signal-label {
            color: #7a7367;
            font-size: 0.68rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.22rem;
        }

        .signal-value {
            color: #17223b;
            font-size: 1.05rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .signal-note {
            color: #7a7367;
            font-size: 0.78rem;
            line-height: 1.35;
            margin-top: 0.22rem;
        }

        .section-heading {
            color: #17223b;
            font-size: 1.18rem;
            font-weight: 850;
            margin-top: 0.65rem;
            margin-bottom: 0.12rem;
        }

        .section-copy {
            color: #716c63;
            font-size: 0.88rem;
            line-height: 1.5;
            margin: 0 0 0.7rem 0;
        }

        .why-strip {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 0.7rem;
            margin: 0.35rem 0 0.7rem 0;
        }

        .why-item {
            background: #ebe6d8;
            border-radius: 10px;
            padding: 0.72rem 0.8rem;
        }

        .why-title {
            color: #17223b;
            font-size: 0.85rem;
            font-weight: 800;
            margin-bottom: 0.15rem;
        }

        .why-copy {
            color: #70695d;
            font-size: 0.76rem;
            line-height: 1.35;
        }

        /* Make the workspace feel like the primary product area */
        div[data-baseweb="tab-list"] {
            gap: 0.35rem;
            background: #e9e4d7;
            padding: 0.3rem;
            border-radius: 12px;
        }

        button[data-baseweb="tab"] {
            border-radius: 9px;
            padding: 0.55rem 0.9rem;
        }

        div.stButton > button {
            width: 100%;
            border: 0;
            border-radius: 10px;
            padding: 0.68rem 1rem;
            background: #17223b;
            color: #ffffff;
            font-weight: 800;
        }

        div.stButton > button:hover {
            background: #223250;
            color: #ffffff;
        }

        div[data-testid="stMetric"] {
            background: #fffdf7;
            border: 1px solid #ded8ca;
            border-radius: 12px;
            padding: 0.78rem 0.9rem;
        }

        div[data-testid="stMetricLabel"] {
            color: #746e64;
        }

        div[data-testid="stMetricValue"] {
            color: #17223b;
        }

        div.stDownloadButton > button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid #17223b;
            background: #fffdf7;
            color: #17223b;
            font-weight: 750;
        }

        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        section[data-testid="stSidebar"] {
            background: #17223b;
        }

        section[data-testid="stSidebar"] * {
            color: #f6f3ea;
        }

        section[data-testid="stSidebar"] div[data-testid="stAlert"] {
            background: rgba(255,255,255,0.08);
        }

        .footer-note {
            margin-top: 1.15rem;
            padding-top: 0.8rem;
            border-top: 1px solid #d8d1c4;
            color: #7b756b;
            font-size: 0.74rem;
            text-align: center;
            line-height: 1.45;
        }

        @media (max-width: 800px) {
            .block-container {
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }

            .top-strip {
                display: block;
            }

            .status-chip {
                display: inline-block;
                margin-top: 0.7rem;
            }

            .why-strip {
                grid-template-columns: 1fr;
            }

            .signal-card {
                min-height: auto;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)





with st.sidebar:
    st.markdown("## ⚡ GB Grid Lab")
    if PROFILE_PATH.exists():
        try:
            sidebar_img = Image.open(PROFILE_PATH)
            st.image(sidebar_img, width=180)
        except Exception:
            pass
    st.markdown("**Kamil Ridwan Kehinde**")
    st.caption("Energy systems forecasting · Data analytics")
    st.divider()
    st.markdown("**Model asset**")
    st.caption("Univariate LSTM")
    st.markdown("**Data source context**")
    st.caption("NESO historical generation data")
    st.markdown("**Modes**")
    st.caption("Historical hindcast + recursive forecast")


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
    <div class="top-strip">
        <div class="top-strip-left">
            <div class="top-strip-kicker">Great Britain · Generation Intelligence</div>
            <h1>Electricity Generation Forecast Lab</h1>
            <p>
                Univariate LSTM inference for Great Britain's historical generation series,
                with compact hindcasting and recursive future forecasting in one workspace.
            </p>
        </div>
        <div class="status-chip">LSTM · LIVE MODEL</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Full-resolution project visual
#
# Important: on desktop the image fits the dashboard width. On mobile it is
# deliberately NOT squeezed into the phone width. It stays at its native pixel
# width inside a horizontally scrollable container, so small labels and charts
# remain much clearer. The original PNG bytes are embedded unchanged.
# -----------------------------------------------------------------------------
if PROFILE_PATH.exists():
    try:
        with Image.open(PROFILE_PATH) as project_image:
            img_width, img_height = project_image.size
            aspect_ratio = img_width / img_height if img_height else 1

        if aspect_ratio >= 1.20:
            original_png = PROFILE_PATH.read_bytes()
            encoded_png = base64.b64encode(original_png).decode("ascii")

            st.markdown(
                f"""
                <div class="project-visual-shell">
                    <div class="project-visual-title">
                        Generation forecasting architecture & analytical overview
                    </div>
                    <div
                        class="project-visual-scroll"
                        style="--native-width: {img_width}px;"
                    >
                        <img
                            src="data:image/png;base64,{encoded_png}"
                            alt="GB electricity generation forecasting architecture and analytical overview"
                        />
                    </div>
                    <div class="project-visual-hint">
                        On mobile, swipe horizontally across the image to view the
                        original-resolution detail without shrinking the labels.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            # A portrait image is kept compact rather than forced into a wide
            # architecture-style viewer.
            project_image = Image.open(PROFILE_PATH)
            image_left, image_main, image_right = st.columns([1.4, 1, 1.4])
            with image_main:
                st.image(
                    project_image,
                    width=min(320, img_width),
                    caption="Kamil Ridwan Kehinde · Lead Modeller",
                )

    except Exception as image_error:
        st.warning(
            "The project image is present but could not be displayed. "
            f"Image error: {image_error}"
        )
else:
    st.info(
        "Project image not found. Add `kamil_profile.png` beside `app.py` "
        "to display it in this section."
    )

sig1, sig2, sig3, sig4 = st.columns(4)

with sig1:
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-label">Model</div>
            <div class="signal-value">Univariate LSTM</div>
            <div class="signal-note">{LOOKBACK}-observation lookback window</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sig2:
    resolution_minutes = int(median_step.total_seconds() / 60)
    resolution_text = (
        f"{resolution_minutes} min"
        if resolution_minutes < 60
        else f"{resolution_minutes / 60:g} hr"
    )
    st.markdown(
        f"""
        <div class="signal-card amber">
            <div class="signal-label">Resolution</div>
            <div class="signal-value">{resolution_text}</div>
            <div class="signal-note">Inferred from deployed history</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sig3:
    st.markdown(
        """
        <div class="signal-card teal">
            <div class="signal-label">Analysis Modes</div>
            <div class="signal-value">2 Workflows</div>
            <div class="signal-note">Hindcast + future forecast</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with sig4:
    st.markdown(
        f"""
        <div class="signal-card">
            <div class="signal-label">Latest Observation</div>
            <div class="signal-value">{data_end.strftime("%d %b %Y")}</div>
            <div class="signal-note">{data_end.strftime("%H:%M")} historical endpoint</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="section-heading">Forecast Workspace</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="section-copy">
        Choose the operational mode below. Hindcast compares model estimates with known history,
        while Future Forecast extends the latest observed sequence recursively.
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

    c_left, c_right = st.columns([1.3, 1])

    with c_left:
        selected_hindcast = st.date_input(
            "Historical start and end dates",
            value=(default_hindcast_start, default_hindcast_end),
            min_value=earliest_hindcast,
            max_value=latest_hindcast,
            key="hindcast_range",
        )

    with c_right:
        st.markdown(
            f"""
            <div class="signal-card amber">
                <div class="signal-label">Available History</div>
                <div class="signal-value">{earliest_hindcast.strftime("%d %b %Y")} → {latest_hindcast.strftime("%d %b %Y")}</div>
                <div class="signal-note">Model evaluation window</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    run_hindcast = st.button("Run Hindcast Analysis", key="run_hindcast")

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
                k3.metric("Bias", f"{bias:,.0f} MW")
                k4.metric("R²", "N/A" if np.isnan(r2) else f"{r2:.4f}")

                chart_df = hindcast_df.set_index("Datetime")[
                    ["Actual_MW", "Predicted_MW"]
                ].rename(
                    columns={
                        "Actual_MW": "Actual",
                        "Predicted_MW": "Modelled",
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
                st.dataframe(preview.head(24), height=300)

                st.download_button(
                    "Download Hindcast CSV",
                    data=hindcast_df.to_csv(index=False),
                    file_name=f"GB_Hindcast_{start_d}_{end_d}.csv",
                    mime="text/csv",
                    key="download_hindcast",
                )

with forecast_tab:
    first_future_date = (data_end + median_step).date()
    default_forecast_start = first_future_date
    default_forecast_end = first_future_date + datetime.timedelta(days=3)

    c_left, c_right = st.columns([1.3, 1])

    with c_left:
        selected_forecast = st.date_input(
            "Future start and end dates",
            value=(default_forecast_start, default_forecast_end),
            min_value=first_future_date,
            key="forecast_range",
        )

    with c_right:
        st.markdown(
            f"""
            <div class="signal-card teal">
                <div class="signal-label">Forecast Seed</div>
                <div class="signal-value">{data_end.strftime("%d %b %Y %H:%M")}</div>
                <div class="signal-note">Latest historical observation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    run_forecast = st.button("Run Future Forecast", key="run_forecast")

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
                k1.metric("Average", f"{avg_mw:,.0f} MW")
                k2.metric("Peak", f"{peak_mw:,.0f} MW")
                k3.metric("Minimum", f"{min_mw:,.0f} MW")
                k4.metric("Energy", f"{total_gwh:,.2f} GWh")

                chart_df = forecast_df.set_index("Datetime")[
                    ["Forecast_MW"]
                ].rename(columns={"Forecast_MW": "Forecast Generation"})
                st.line_chart(chart_df)

                preview = forecast_df.rename(
                    columns={"Forecast_MW": "Forecast Generation [MW]"}
                )
                st.dataframe(preview.head(24), height=300)

                st.download_button(
                    "Download Future Forecast CSV",
                    data=forecast_df.to_csv(index=False),
                    file_name=f"GB_LSTM_Forecast_{start_d}_{end_d}.csv",
                    mime="text/csv",
                    key="download_forecast",
                )

st.markdown(
    """
    <div class="why-strip">
        <div class="why-item">
            <div class="why-title">⚡ Grid balancing</div>
            <div class="why-copy">Supports understanding of expected generation conditions.</div>
        </div>
        <div class="why-item">
            <div class="why-title">🌱 Renewable integration</div>
            <div class="why-copy">Frames variability as low-carbon generation increases.</div>
        </div>
        <div class="why-item">
            <div class="why-title">📊 System planning</div>
            <div class="why-copy">Provides compact analytical evidence for operational studies.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="footer-note">
        © 2026 Kamil Ridwan Kehinde · GB Electricity Generation Forecast Lab ·
        NESO historical generation data context · Research and analytical demonstration
    </div>
    """,
    unsafe_allow_html=True,
)
