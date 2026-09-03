# -*- coding: utf-8 -*-
"""Great Britain Electricity Generation Forecasting Dashboard."""

import datetime
import joblib
import pandas as pd
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="GB Electricity Forecasting | Kamil Ridwan",
    page_icon="⚡",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background:
                radial-gradient(circle at top right, rgba(13, 110, 145, 0.08), transparent 32rem),
                linear-gradient(180deg, #f7faf9 0%, #f1f6f5 100%);
            color: #17332f;
        }

        .block-container {
            max-width: 1240px;
            padding-top: 1.15rem;
            padding-bottom: 2rem;
        }

        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        header[data-testid="stHeader"] {
            background: rgba(247, 250, 249, 0.88);
        }

        .hero {
            background: linear-gradient(135deg, #0d3b36 0%, #145f57 52%, #0d6e91 100%);
            border-radius: 20px;
            padding: 1.8rem 2rem;
            margin-bottom: 1rem;
            box-shadow: 0 12px 34px rgba(13, 59, 54, 0.13);
        }

        .hero-kicker {
            margin: 0 0 0.45rem 0;
            color: #ccece6;
            font-size: 0.77rem;
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
            max-width: 920px;
            margin: 0.72rem 0 0 0;
            color: #e8f4f1;
            font-size: 1rem;
            line-height: 1.6;
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
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            text-transform: uppercase;
            margin-bottom: 0.28rem;
        }

        .info-value {
            color: #123e38;
            font-size: 1.14rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .info-note {
            color: #70817d;
            font-size: 0.81rem;
            line-height: 1.38;
            margin-top: 0.35rem;
        }

        .section-title {
            color: #173f39;
            font-size: 1.35rem;
            font-weight: 800;
            letter-spacing: -0.01em;
            margin-top: 0.65rem;
            margin-bottom: 0.16rem;
        }

        .section-copy {
            color: #677d78;
            font-size: 0.92rem;
            line-height: 1.55;
            margin-top: 0;
            margin-bottom: 0.85rem;
        }

        .pillar {
            min-height: 105px;
            background: #ffffff;
            border: 1px solid #dde8e5;
            border-radius: 14px;
            padding: 0.95rem 1rem;
        }

        .pillar-title {
            color: #164f48;
            font-size: 0.94rem;
            font-weight: 800;
            margin-bottom: 0.28rem;
        }

        .pillar-copy {
            color: #6a7c78;
            font-size: 0.82rem;
            line-height: 1.45;
        }

        div.stButton > button {
            width: 100%;
            border: 0;
            border-radius: 12px;
            padding: 0.74rem 1rem;
            background: linear-gradient(90deg, #155f57, #0d6e91);
            color: #ffffff;
            font-weight: 800;
            box-shadow: 0 6px 18px rgba(13, 110, 145, 0.17);
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dce8e4;
            border-radius: 15px;
            padding: 0.95rem 1rem;
            box-shadow: 0 4px 16px rgba(18, 63, 57, 0.05);
        }

        div[data-testid="stMetricLabel"] {
            color: #627873;
        }

        div[data-testid="stMetricValue"] {
            color: #143f39;
        }

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        div.stDownloadButton > button {
            width: 100%;
            border-radius: 12px;
            border: 1px solid #155f57;
            background: #ffffff;
            color: #155f57;
            font-weight: 750;
        }

        section[data-testid="stSidebar"] {
            background: #f4f8f7;
            border-right: 1px solid #dce7e4;
        }

        .app-footer {
            margin-top: 1.7rem;
            padding-top: 1rem;
            border-top: 1px solid #dce7e4;
            text-align: center;
            color: #73847f;
            font-size: 0.78rem;
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

            .info-card,
            .pillar {
                min-height: auto;
            }
        }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("⚡ GB Grid Analytics")
    st.write("**Kamil Ridwan Kehinde**")
    st.caption("Energy Systems · Forecasting · Data Analytics")
    st.divider()
    st.info(
        "**Project Goal**\n\n"
        "Machine-learning forecasting of Great Britain's electricity "
        "generation using historical NESO generation data."
    )
    st.caption(
        "Portfolio case study demonstrating energy-system analytics, "
        "time-series feature engineering and deployed inference."
    )


@st.cache_resource
def load_model():
    return joblib.load("gb_gen_time_only.pkl")


model = load_model()

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">Great Britain · Electricity System Intelligence</div>
        <h1>GB Electricity Generation Forecasting</h1>
        <p>
            A deployed machine-learning forecasting interface for exploring
            Great Britain's electricity generation profile using historical
            NESO generation data and calendar-based time features.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption("Interface version: GB Forecast Dashboard v2.0 · September 2026")

profile_col, card_1, card_2, card_3 = st.columns([1.15, 1, 1, 1])

with profile_col:
    try:
        img = Image.open("kamil_profile.jpg")
        st.image(
            img,
            width=235,
            caption="Kamil Kehinde · Lead Modeller",
        )
    except Exception:
        st.info("Lead modeller: **Kamil Ridwan Kehinde**")

with card_1:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Forecast Resolution</div>
            <div class="info-value">30 Minutes</div>
            <div class="info-note">Half-hourly generation estimates across the selected horizon.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card_2:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Forecast Drivers</div>
            <div class="info-value">Calendar Features</div>
            <div class="info-note">Hour, weekday, month and year are used by the deployed model.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with card_3:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Data Context</div>
            <div class="info-value">NESO Generation</div>
            <div class="info-note">Great Britain operational generation history used for modelling.</div>
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
        Short-term generation forecasting supports operational understanding of
        electricity-system behaviour as Great Britain integrates more variable
        renewable generation and manages changing demand patterns.
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
                Supports visibility of expected generation requirements as supply
                and demand conditions evolve.
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
                Helps frame the operational challenge of integrating variable
                wind, solar and other low-carbon generation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with p3:
    st.markdown(
        """
        <div class="pillar">
            <div class="pillar-title">📉 Operational Efficiency</div>
            <div class="pillar-copy">
                Provides an analytical basis for understanding generation patterns
                and potential system-planning requirements.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="section-title">Forecast Configuration</div>',
    unsafe_allow_html=True,
)
st.markdown(
    """
    <div class="section-copy">
        Select a start and end date. The deployed model will generate half-hourly
        electricity-generation estimates for the chosen period.
    </div>
    """,
    unsafe_allow_html=True,
)

control_col, context_col = st.columns([1.35, 1])

with control_col:
    start_init = datetime.date(2026, 2, 16)
    end_init = start_init + datetime.timedelta(days=3)

    selected_range = st.date_input(
        "Select Start and End Dates",
        value=(start_init, end_init),
        help="The model will generate a 30-minute interval forecast for this period.",
    )

with context_col:
    st.markdown(
        """
        <div class="info-card">
            <div class="info-label">Inference Mode</div>
            <div class="info-value">User-Selected Horizon</div>
            <div class="info-note">
                Run the model across any valid start and end date selected in the
                forecasting control.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

predict_btn = st.button("🚀 Run Forecast Analysis")

if predict_btn:
    if isinstance(selected_range, tuple) and len(selected_range) == 2:
        start_d, end_d = selected_range

        if start_d > end_d:
            st.error("⚠️ The start date must be earlier than the end date.")
        else:
            with st.spinner("Generating half-hourly GB generation forecast..."):
                idx = pd.date_range(
                    start=datetime.datetime.combine(start_d, datetime.time(0, 0)),
                    end=datetime.datetime.combine(end_d, datetime.time(23, 30)),
                    freq="30min",
                )

                df_pred = pd.DataFrame(index=idx)
                df_pred["hour"] = df_pred.index.hour
                df_pred["dayofweek"] = df_pred.index.dayofweek
                df_pred["month"] = df_pred.index.month
                df_pred["year"] = df_pred.index.year

                df_pred["Forecast_MW"] = model.predict(df_pred)

                avg_mw = df_pred["Forecast_MW"].mean()
                peak_mw = df_pred["Forecast_MW"].max()
                min_mw = df_pred["Forecast_MW"].min()
                total_gwh = (df_pred["Forecast_MW"].sum() * 0.5) / 1000

            st.markdown(
                '<div class="section-title">Forecast Summary</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="section-copy">
                    Key generation indicators calculated across the selected
                    half-hourly forecasting horizon.
                </div>
                """,
                unsafe_allow_html=True,
            )

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Average Generation", f"{avg_mw:,.0f} MW")
            k2.metric("Predicted Peak", f"{peak_mw:,.0f} MW")
            k3.metric("Minimum Generation", f"{min_mw:,.0f} MW")
            k4.metric("Total Energy Volume", f"{total_gwh:,.2f} GWh")

            st.markdown(
                '<div class="section-title">Generation Forecast Profile</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="section-copy">
                    Half-hourly predicted electricity generation from
                    <b>{start_d.strftime("%d %b %Y")}</b> to
                    <b>{end_d.strftime("%d %b %Y")}</b>.
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.line_chart(df_pred["Forecast_MW"])

            st.markdown(
                '<div class="section-title">Forecast Data</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                """
                <div class="section-copy">
                    Review the model output and export the complete half-hourly
                    forecast for further analysis.
                </div>
                """,
                unsafe_allow_html=True,
            )

            display_df = df_pred[["Forecast_MW"]].copy()
            display_df.index.name = "Datetime"
            display_df = display_df.rename(
                columns={"Forecast_MW": "Forecast Generation [MW]"}
            )

            st.dataframe(
                display_df.head(20).style.format(
                    {"Forecast Generation [MW]": "{:,.2f}"}
                ),
                height=330,
            )

            st.download_button(
                "📥 Download Complete Forecast CSV",
                data=df_pred.to_csv(),
                file_name=f"UK_Grid_Forecast_{start_d}.csv",
                mime="text/csv",
            )
    else:
        st.error(
            "⚠️ Please select a valid range with both a start and an end date."
        )

st.markdown(
    """
    <div class="app-footer">
        © 2026 Energy Analytics Portfolio · Kamil Ridwan Kehinde<br>
        Data context: National Energy System Operator (NESO) generation data
    </div>
    """,
    unsafe_allow_html=True,
)
