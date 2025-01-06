import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.impute import SimpleImputer

# Set Streamlit page configuration
st.set_page_config(
    page_title="Weather Anomaly Detection",
    page_icon="🌦️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load Dataset
@st.cache_data
def load_data(file_path = "weather_data_extended.csv"):
    try:
        data = pd.read_csv(file_path)
        return data
    except FileNotFoundError:
        print("File not found. Please check the file path.")
        return None
    except pd.errors.EmptyDataError:
        print("File is empty. Please check the file contents.")
        return None
    except pd.errors.ParserError:
        print("Error parsing the file. Please check the file format.")
        return None

# Preprocess Data
@st.cache_data
def preprocess_data(data):
    numeric_data = data.select_dtypes(include=[np.number])  # Select numeric columns
    imputer = SimpleImputer(strategy='mean')
    imputed_data = imputer.fit_transform(numeric_data)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)
    pca = PCA(n_components=0.95)
    reduced_data = pca.fit_transform(scaled_data)
    return reduced_data, imputer, scaler, pca

# Train Models
@st.cache_data
def train_models(data):
    iso_model = IsolationForest(contamination=0.1, random_state=42)
    iso_model.fit(data)

    dbscan_model = DBSCAN(eps=0.5, min_samples=5)
    dbscan_labels = dbscan_model.fit_predict(data)

    return iso_model, dbscan_model

# Make Predictions
def predict(input_data, imputer, scaler, pca, iso_model, dbscan_model, original_data):
    input_data = np.array(input_data).reshape(1, -1)
    input_data = imputer.transform(input_data)
    input_data = scaler.transform(input_data)
    input_data = pca.transform(input_data)  # Transform to PCA space

    iso_pred = iso_model.predict(input_data)[0]

    # Concatenate the original data with the new input data for DBSCAN prediction
    combined_data = np.vstack([original_data, input_data])
    dbscan_pred = dbscan_model.fit_predict(combined_data)[-1]

    return "Anomaly detected" if iso_pred == -1 else f"Normal, assigned to cluster {dbscan_pred}"

# Main Function
def main():
    st.title("🌦️ Weather Anomaly Detection")
    st.markdown("### Detect anomalies in weather data using Isolation Forest and DBSCAN")
    st.sidebar.header("Enter Weather Data")

    # Load data
    file_path = "weather_data_extended.csv"  # Replace with your actual file path
    data = load_data(file_path)
    reduced_data, imputer, scaler, pca = preprocess_data(data)
    iso_model, dbscan_model = train_models(reduced_data)

    # User Input Section
    temperature = st.sidebar.number_input("Temperature (°C)", value=25.0, step=0.1)
    feels_like = st.sidebar.number_input("Feels Like (°C)", value=24.0, step=0.1)
    humidity = st.sidebar.number_input("Humidity (%)", value=60.0, step=1.0)
    wind_speed = st.sidebar.number_input("Wind Speed (kph)", value=10.0, step=0.1)
    cloud_cover = st.sidebar.number_input("Cloud Cover (%)", value=30.0, step=1.0)
    pressure = st.sidebar.number_input("Pressure (mb)", value=1015.0, step=0.1)
    uv_index = st.sidebar.number_input("UV Index", value=5.0, step=1.0)
    visibility = st.sidebar.number_input("Visibility (km)", value=10.0, step=0.1)

    # Create input array for prediction
    sample_input = [temperature, feels_like, humidity, wind_speed, cloud_cover, pressure, uv_index, visibility]

    # Prediction Button
    if st.sidebar.button("Predict"):
        result = predict(sample_input, imputer, scaler, pca, iso_model, dbscan_model, reduced_data)

        # Display Prediction Result
        if "Anomaly" in result:
            st.error(f"🔴 **Prediction:** {result}")
        else:
            st.success(f"🟢 **Prediction:** {result}")

    # Display Sample Data
    st.subheader("Sample Data from Dataset")
    st.write(data.head())

    # Display Model Insights
    st.subheader("Model Insights")
    st.markdown("""
    - **Isolation Forest**: Detects anomalies based on the isolation of data points.
    - **DBSCAN**: Clusters data points and identifies outliers as noise.
    """)

# Run the app
if __name__ == "__main__":
    main()
