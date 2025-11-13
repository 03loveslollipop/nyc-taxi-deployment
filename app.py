"""
NYC Taxi Trip Duration Prediction - Streamlit App
Loads the trained neural network pipeline and provides an interactive prediction interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px

try:
    from deployment.nn_sklearn_wrapper import NeuralNetworkRegressor  # noqa: F401
    from deployment.pipeline_components import DummyEncoder, MixedScaler  # noqa: F401
except ModuleNotFoundError:
    from nn_sklearn_wrapper import NeuralNetworkRegressor  # noqa: F401
    from pipeline_components import DummyEncoder, MixedScaler  # noqa: F401

BASE_DIR = Path(__file__).resolve().parent

# Page configuration
st.set_page_config(
    page_title="NYC Taxi Trip Duration Predictor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FFD700;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND INFO
# ============================================================================

@st.cache_resource
def load_pipeline():
    """Load the trained pipeline from pickle"""
    pipeline_path = BASE_DIR / 'nn_pipeline.pkl'
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

@st.cache_data
def load_model_info():
    """Load model information"""
    info_path = BASE_DIR / 'model_info.json'
    with open(info_path, 'r') as f:
        info = json.load(f)
    return info

# Load model
try:
    pipeline = load_pipeline()
    model_info = load_model_info()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Please make sure 'nn_pipeline.pkl' and 'model_info.json' are in the same directory as this app.")
    model_loaded = False

# ============================================================================
# HEADER
# ============================================================================

st.markdown('<div class="main-header">🚕 NYC Taxi Trip Duration Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict taxi trip duration using a Neural Network trained on 70% of NYC taxi data</div>', unsafe_allow_html=True)

if model_loaded:
    # Show model performance
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Test RMSE", f"{model_info['test_rmse']:.4f} min")
    with col2:
        st.metric("Test MAE", f"{model_info['test_mae']:.4f} min")
    with col3:
        st.metric("Test R²", f"{model_info['test_r2']:.4f}")

st.markdown("---")

# ============================================================================
# SIDEBAR - MODEL INFO
# ============================================================================

if model_loaded:
    with st.sidebar:
        st.header("📊 Model Information")
        
        st.subheader("Architecture")
        st.write(f"**Type:** {model_info['model_architecture']}")
        st.write(f"**Optimizer:** {model_info['optimizer']}")
        st.write(f"**Learning Rate:** {model_info['learning_rate']}")
        st.write(f"**Activation:** {model_info['activation']}")
        
        st.subheader("Features")
        st.write(f"**Numeric Features:** {len(model_info['numeric_columns'])}")
        st.write(f"**Categorical Features:** {len(model_info['categorical_columns'])}")
        
        with st.expander("View Feature Details"):
            st.write("**Categorical Columns:**")
            for col in model_info['categorical_columns']:
                st.write(f"- {col}")
            
            st.write("**Numeric Columns:**")
            for col in model_info['numeric_columns'][:10]:  # Show first 10
                st.write(f"- {col}")
            if len(model_info['numeric_columns']) > 10:
                st.write(f"... and {len(model_info['numeric_columns']) - 10} more")

# ============================================================================
# MAIN INTERFACE
# ============================================================================

if model_loaded:
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction", "ℹ️ About"])
    
    # ========================================================================
    # TAB 1: SINGLE PREDICTION
    # ========================================================================
    
    with tab1:
        st.header("Enter Trip Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Location & Time")
            
            pickup_borough = st.selectbox(
                "Pickup Borough",
                options=['Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Staten Island'],
                help="Borough where the trip starts"
            )
            
            dropoff_borough = st.selectbox(
                "Dropoff Borough",
                options=['Manhattan', 'Queens', 'Brooklyn', 'Bronx', 'Staten Island'],
                help="Borough where the trip ends"
            )
            
            pickup_hour = st.slider("Pickup Hour (0-23)", 0, 23, 12)
            pickup_day = st.slider("Day of Month (1-31)", 1, 31, 15)
            pickup_weekday = st.slider("Weekday (0=Monday, 6=Sunday)", 0, 6, 2)
            pickup_month = st.slider("Month (1-12)", 1, 12, 6)
        
        with col2:
            st.subheader("🚖 Trip Details")
            
            distance = st.number_input(
                "Trip Distance (miles)",
                min_value=0.01,
                max_value=100.0,
                value=2.5,
                step=0.1,
                help="Distance of the trip in miles"
            )
            
            passenger_count = st.number_input(
                "Passenger Count",
                min_value=1,
                max_value=6,
                value=1,
                help="Number of passengers"
            )
            
            payment_type = st.selectbox(
                "Payment Type",
                options=['credit card', 'cash', 'no charge', 'dispute'],
                help="Method of payment"
            )
            
            fare_amount = st.number_input(
                "Fare Amount ($)",
                min_value=0.0,
                max_value=500.0,
                value=10.0,
                step=0.5,
                help="Base fare amount"
            )
            
            extra = st.number_input(
                "Extra Charges ($)",
                min_value=0.0,
                max_value=50.0,
                value=0.5,
                step=0.25,
                help="Extra charges (rush hour, overnight)"
            )
            
            mta_tax = st.number_input(
                "MTA Tax ($)",
                min_value=0.0,
                max_value=10.0,
                value=0.5,
                step=0.25,
                help="MTA tax"
            )
            
            tip_amount = st.number_input(
                "Tip Amount ($)",
                min_value=0.0,
                max_value=100.0,
                value=2.0,
                step=0.5,
                help="Tip amount"
            )
            
            tolls_amount = st.number_input(
                "Tolls Amount ($)",
                min_value=0.0,
                max_value=50.0,
                value=0.0,
                step=0.5,
                help="Toll charges"
            )
        
        st.markdown("---")
        
        # Predict button
        if st.button("🔮 Predict Trip Duration", type="primary", use_container_width=True):
            # Create input dataframe
            input_data = pd.DataFrame({
                'pickup_borough': [pickup_borough],
                'dropoff_borough': [dropoff_borough],
                'payment_type': [payment_type],
                'passenger_count': [passenger_count],
                'trip_distance': [distance],
                'fare_amount': [fare_amount],
                'extra': [extra],
                'mta_tax': [mta_tax],
                'tip_amount': [tip_amount],
                'tolls_amount': [tolls_amount],
                'pickup_hour': [pickup_hour],
                'pickup_day': [pickup_day],
                'pickup_weekday': [pickup_weekday],
                'pickup_month': [pickup_month]
            })
            
            # Make prediction
            try:
                prediction = pipeline.predict(input_data)[0]
                
                # Display result
                st.success("Prediction Complete!")
                
                # Show prediction with nice formatting
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    st.markdown("### 🕐 Predicted Trip Duration")
                    st.markdown(f"<h1 style='text-align: center; color: #FFD700;'>{prediction:.2f} minutes</h1>", 
                               unsafe_allow_html=True)
                    st.markdown(f"<h3 style='text-align: center; color: #888;'>(~{prediction/60:.1f} hours)</h3>", 
                               unsafe_allow_html=True)
                
                # Additional info
                st.info(f"""
                **Trip Summary:**
                - Route: {pickup_borough} → {dropoff_borough}
                - Distance: {distance:.2f} miles
                - Estimated Speed: {(distance / (prediction/60)):.2f} mph
                - Total Fare: ${fare_amount + extra + mta_tax + tip_amount + tolls_amount:.2f}
                """)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
    
    # ========================================================================
    # TAB 2: BATCH PREDICTION
    # ========================================================================
    
    with tab2:
        st.header("Batch Prediction from CSV")
        
        st.info("""
        Upload a CSV file with the following columns:
        - pickup_borough, dropoff_borough, payment_type
        - passenger_count, trip_distance, fare_amount, extra, mta_tax, tip_amount, tolls_amount
        - pickup_hour, pickup_day, pickup_weekday, pickup_month
        """)
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Load data
                input_df = pd.read_csv(uploaded_file)
                
                st.write(f"Loaded {len(input_df)} rows")
                st.dataframe(input_df.head())
                
                if st.button("🔮 Predict All", type="primary"):
                    # Make predictions
                    with st.spinner("Making predictions..."):
                        predictions = pipeline.predict(input_df)
                    
                    # Add predictions to dataframe
                    results_df = input_df.copy()
                    results_df['predicted_duration'] = predictions
                    
                    st.success("Predictions complete!")
                    
                    # Show statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Mean Duration", f"{predictions.mean():.2f} min")
                    with col2:
                        st.metric("Median Duration", f"{np.median(predictions):.2f} min")
                    with col3:
                        st.metric("Min Duration", f"{predictions.min():.2f} min")
                    with col4:
                        st.metric("Max Duration", f"{predictions.max():.2f} min")
                    
                    # Show results
                    st.dataframe(results_df)
                    
                    # Download button
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Predictions",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
                    
                    # Visualization
                    fig = px.histogram(
                        predictions,
                        nbins=50,
                        title="Distribution of Predicted Trip Durations",
                        labels={'value': 'Duration (minutes)', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
            except Exception as e:
                st.error(f"Error processing file: {e}")
    
    # ========================================================================
    # TAB 3: ABOUT
    # ========================================================================
    
    with tab3:
        st.header("About This Model")
        
        st.markdown("""
        ### 🧠 Neural Network Architecture
        
        This model uses a **Medium-depth neural network** with the following characteristics:
        
        - **Architecture:** 3-layer feedforward network with batch normalization and dropout
        - **Layer sizes:** 256 → 128 → 64 → 1
        - **Activation function:** ReLU
        - **Optimizer:** AdamW with learning rate 0.001
        - **Training:** Trained on 70% of NYC taxi trip data with early stopping
        
        ### 📊 Model Performance
        
        The model was evaluated on a held-out 30% test set:
        """)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{model_info['test_rmse']:.4f} minutes", 
                     help="Root Mean Squared Error - lower is better")
        with col2:
            st.metric("MAE", f"{model_info['test_mae']:.4f} minutes",
                     help="Mean Absolute Error - lower is better")
        with col3:
            st.metric("R² Score", f"{model_info['test_r2']:.4f}",
                     help="Coefficient of determination - closer to 1 is better")
        
        st.markdown("""
        ### 🏆 Why This Model?
        
        This neural network was selected as the best performer among multiple models tested:
        - **Neural Network (Winner):** Best RMSE, MAE, and R² on test set
        - **XGBoost:** Close second, slightly higher error
        - **SVM:** Third place, higher error overall
        
        ### 🔧 Feature Engineering
        
        The model uses carefully engineered features including:
        - **Location:** Borough information (pickup/dropoff)
        - **Temporal:** Hour, day, weekday, month
        - **Trip characteristics:** Distance, passenger count
        - **Financial:** Fare amounts, tips, tolls, taxes
        
        ### 📈 Preprocessing Pipeline
        
        The model includes an end-to-end sklearn pipeline with:
        1. **Mixed scaling:** StandardScaler for some features, PowerTransformer for others
        2. **Categorical encoding:** One-hot encoding for borough and payment type
        3. **Neural network:** PyTorch model wrapped in sklearn-compatible interface
        
        ### 💾 Model Deployment
        
        The entire pipeline (preprocessing + model) is serialized as a single pickle file,
        making deployment simple and ensuring consistency between training and inference.
        """)
        
        st.markdown("---")
        st.markdown("**Built with:** PyTorch, scikit-learn, Streamlit")
        st.markdown("**Data:** NYC Taxi Trip Duration Dataset")

else:
    st.error("Model not loaded. Please ensure all required files are present.")
