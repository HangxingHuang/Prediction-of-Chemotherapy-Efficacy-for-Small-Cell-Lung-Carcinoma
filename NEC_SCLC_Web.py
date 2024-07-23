import streamlit as st
from joblib import load
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import joblib
from scipy import stats

st.title("XGBoost-based Prediction of Chemotherapy Efficacy for Small Cell Lung Carcinoma (SCLC)")
# Create a function to generate HTML for person icons
def generate_person_icons(filled_count, total_count=100):
    # SVG person icon
    icon_svg = """
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="7" r="4" stroke="black" stroke-width="2" fill="none"/>
      <path d="M4 21C4 16.6863 7.68629 13 12 13C16.3137 13 20 16.6863 20 21H4Z" stroke="black" stroke-width="2" fill="none"/>
    </svg>
    """
    
    # Replace fill attribute to change color
    filled_icon = icon_svg.replace('fill="none"', 'fill="lightblue"')
    empty_icon = icon_svg.replace('fill="none"', 'fill="gray"')

    # Generate the HTML for the icons
    icons_html = ''.join([filled_icon if i < filled_count else empty_icon for i in range(total_count)])
    return f"<div style='display: flex; flex-wrap: wrap; width: 480px;'>{icons_html}</div>"
# Load model
loaded_model = load("xgb_SCLC_5%_model.joblib")

# Load saved Scaler
scaler = joblib.load('xgb_SCLC_5%_scaler.joblib')

# Load validation set predictions
validation_predictions = np.load('xgb_SCLC_5%_predictions.npy')
# Ensure validation_predictions is a 1D array
if validation_predictions.ndim > 1:
    validation_predictions = validation_predictions.ravel()

# Define feature order
features = ['Weight', 'CA125', 'P40', 'GLU', 'NSE', 'HDL', 'TP', 'Smoking', 'Ki67']
continuous_features = ['Weight', 'CA125', 'GLU', 'NSE', 'HDL', 'TP', 'Ki67']

# Categorical feature mappings
smoking_options = {0: 'Never', 1: 'Former', 2: 'Current'}
p40_options = {0: 'Negative', 1: 'Positive'}

# Reverse mappings
smoking_reverse = {v: k for k, v in smoking_options.items()}
p40_reverse = {v: k for k, v in p40_options.items()}


# Left column: input form
with st.sidebar:
    st.header("Your information")
    
    weight = st.number_input('Weight (Kg)', min_value=0.0, max_value=200.0, step=1.0, key='weight')
    ca125 = st.number_input('Carbohydrate Antigen 125 (CA125, IU/mL)', min_value=0.0, max_value=1000.0, step=0.1, key='ca125')
    p40 = st.selectbox('Tumor Protein P40', options=list(p40_options.values()), key='p40')
    glu = st.number_input('Blood Glucose (GLU, mmol/L)', min_value=0.0, max_value=50.0, step=0.1, key='glu')
    nse = st.number_input('Neuron-Specific Enolase (NSE, ng/mL)', min_value=0.0, max_value=200.0, step=0.1, key='nse')
    hdl = st.number_input('High Density Lipoprotein (HDL, mmol/L)', min_value=0.0, max_value=10.0, step=0.1, key='hdl')
    tp = st.number_input('Total Protein (TP, g/L)', min_value=0.0, max_value=100.0, step=0.1, key='tp')
    smoking = st.selectbox('Smoking', options=list(smoking_options.values()), key='smoking')
    ki67 = st.number_input('Marker of Proliferation Ki67 (Ki67, %)', min_value=0.0, max_value=100.0, step=1.0, key='ki67')

# Middle column: buttons
with st.container():
    st.write("")  # Placeholder

    # Use custom CSS for button styles
    st.markdown(
        """
        <style>
        .clear-button {
            background-color: transparent;
            color: black;
            border: none;
            text-decoration: underline;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            display: inline-block;
        }
        .clear-button:hover {
            color: red;
        }
        .clear-button:active {
            color: darkred;
        }
        </style>
        """, unsafe_allow_html=True)

    # Use HTML button
    st.markdown(
        """
        <a href="?reset=true" class="clear-button">Clear</a>
        """, unsafe_allow_html=True)

# If the prediction button is clicked
if st.button('Prediction'):
        # Prepare input data
        user_input = pd.DataFrame([[weight, ca125, p40_reverse[p40], glu, nse, hdl, tp, smoking_reverse[smoking], ki67]], columns=features)
        
        # Extract continuous features
        user_continuous_input = user_input[continuous_features]
        
        # Normalize continuous features
        user_continuous_input_normalized = scaler.transform(user_continuous_input)
        
        # Combine normalized data back into the full input
        user_input_normalized = user_input.copy()
        user_input_normalized[continuous_features] = user_continuous_input_normalized

        # Get prediction probability
        prediction_proba = loaded_model.predict_proba(user_input_normalized)[:, 1][0]
        prediction_percentage = round(prediction_proba * 100)

        # Combine user prediction with validation predictions
        combined_predictions = np.concatenate([validation_predictions, np.array([prediction_proba])])

        # Calculate standard deviation and confidence interval
        std_dev = np.std(combined_predictions)
        confidence_level = 0.95
        degrees_of_freedom = len(combined_predictions) - 1
        t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_of_freedom)
        margin_of_error = t_critical * (std_dev / np.sqrt(len(combined_predictions)))
        lower_bound_percentage = max(prediction_percentage - margin_of_error * 100, 0)
        upper_bound_percentage = min(prediction_percentage + margin_of_error * 100, 100)

        lower_bound_percentage = round(lower_bound_percentage)
        upper_bound_percentage = round(upper_bound_percentage)
        
        # Right column: show prediction results
        with st.container():
            st.header("Your result")
            st.markdown(f"The probability that SCLC patients benefit from chemotherapy is (95% confidence interval):")
            result_html = f"""
            <div style="display: flex; align-items: center;">
                <span style="color:red; font-weight:bold; font-size:48px;">{prediction_percentage}%</span>
                <span style="margin-left: 10px;">({lower_bound_percentage}% to {upper_bound_percentage}%)</span>
            </div>
            """
            st.markdown(result_html, unsafe_allow_html=True)
            # Use the function to generate icons based on prediction
            icons_html = generate_person_icons(prediction_percentage)

            # Display the generated icons
            st.markdown(f"""
                <div style="display: flex; align-items: center;">
                </div>
                <div>
                    {icons_html}
                </div>
            """, unsafe_allow_html=True)
            
            # Show additional information
            st.write(f"This result predicts how likely you are to benefit from chemotherapy. The probability means that out of 100 patients with similar characteristics, approximately {prediction_percentage}% may benefit from this therapy. More specifically, we're 95% confident that {lower_bound_percentage} to {upper_bound_percentage} out of 100 patients may benefit from this therapy, based on our training data. However, it's important to recognize that this is just a rough ballpark estimate. Individual patient outcomes can vary significantly, and a healthcare provider can provide a more precise assessment, taking into account a broader range of factors and personal medical history.")
            st.markdown(f"<span style='color:red;'>Disclaimer:</span> This tool is provided for informational purposes only and should NOT be considered as medical advice or a substitute for professional consultation. Users should seek proper medical counsel and discuss their treatment options with a qualified healthcare provider.", unsafe_allow_html=True)
