import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import plotly.express as px


# ==================================================
# CUSTOM POSITIONAL ENCODING LAYER
# ==================================================

@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):

    def call(self, x):
        seq_len = tf.shape(x)[1]

        positions = tf.range(
            start=0,
            limit=seq_len,
            delta=1
        )

        positions = tf.cast(
            positions,
            tf.float32
        )

        return x + tf.expand_dims(
            positions,
            axis=-1
        )

    def get_config(self):
        return super().get_config()


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Fraud Intelligence Dashboard",
    page_icon="💳",
    layout="wide"
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_model():

    import os

    st.write("Current Files:")
    st.write(os.listdir("."))

    try:

        model = tf.keras.models.load_model(
            "attention_model.keras",
            custom_objects={
                "PositionalEncoding": PositionalEncoding
            },
            compile=False
        )

        st.success("Model Loaded Successfully")

        return model

    except Exception as e:

        st.error("MODEL LOAD FAILED")
        st.exception(e)

        raise e

@st.cache_resource
def load_scaler():
    return joblib.load("scaler .pkl")


model = load_model()
scaler = load_scaler()


# ==================================================
# HEADER
# ==================================================

st.title("💳 Fraud Intelligence Dashboard")

st.markdown(
    """
    Upload a transaction CSV file and predict
    potentially fraudulent transactions using
    LSTM + Attention + Positional Encoding.
    """
)


# ==================================================
# FILE UPLOADER
# ==================================================

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)


# ==================================================
# PREDICTION
# ==================================================

if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        if st.button("Analyze Transactions"):

            working_df = df.copy()

            if "Class" in working_df.columns:
                features_df = working_df.drop(
                    columns=["Class"]
                )
            else:
                features_df = working_df

            feature_values = features_df.values

            # Scale Features
            scaled_features = scaler.transform(
                feature_values
            )

            # Create Sequences
            SEQ_LEN = 5

            sequences = []

            for i in range(
                len(scaled_features) - SEQ_LEN
            ):
                sequences.append(
                    scaled_features[
                        i:i + SEQ_LEN
                    ]
                )

            X = np.array(sequences)

            if len(X) == 0:
                st.error(
                    "Dataset must contain at least 6 rows."
                )
                st.stop()

            # Predict
            predictions = model.predict(
                X,
                verbose=0
            )

            fraud_prob = predictions.flatten()

            results = df.iloc[
                SEQ_LEN:
            ].copy()

            results[
                "Fraud_Probability"
            ] = fraud_prob

            # Metrics
            avg_prob = fraud_prob.mean() * 100

            high_risk = results[
                results[
                    "Fraud_Probability"
                ] > 0.80
            ]

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Transactions",
                    len(results)
                )

            with col2:
                st.metric(
                    "Average Fraud Risk",
                    f"{avg_prob:.2f}%"
                )

            with col3:
                st.metric(
                    "High Risk Transactions",
                    len(high_risk)
                )

            # High Risk Table
            st.subheader(
                "🚨 High Risk Transactions"
            )

            if len(high_risk) > 0:
                st.dataframe(high_risk)
            else:
                st.success(
                    "No high-risk transactions detected."
                )

            # Histogram
            st.subheader(
                "Fraud Probability Distribution"
            )

            fig = px.histogram(
                results,
                x="Fraud_Probability",
                nbins=30,
                title="Fraud Probability Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # Top 10 Risky Transactions
            st.subheader(
                "Top 10 Riskiest Transactions"
            )

            top10 = results.sort_values(
                by="Fraud_Probability",
                ascending=False
            ).head(10)

            st.dataframe(top10)

            # Download Results
            csv = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                label="Download Predictions",
                data=csv,
                file_name="fraud_predictions.csv",
                mime="text/csv"
            )

    except Exception as e:

        st.error(
            f"Prediction Error: {str(e)}"
        )
