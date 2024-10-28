from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging

# Initialize Flask app
app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained model
model = joblib.load('corn_yield_model.joblib')  # Ensure this path is correct

# Complete state mappings
state_mapping = {
    'Alabama': 0, 'Alaska': 1, 'Arizona': 2, 'Arkansas': 3, 'California': 4,
    'Colorado': 5, 'Connecticut': 6, 'Delaware': 7, 'Florida': 8, 'Georgia': 9,
    'Hawaii': 10, 'Idaho': 11, 'Illinois': 12, 'Indiana': 13, 'Iowa': 14,
    'Kansas': 15, 'Kentucky': 16, 'Louisiana': 17, 'Maine': 18, 'Maryland': 19,
    'Massachusetts': 20, 'Michigan': 21, 'Minnesota': 22, 'Mississippi': 23,
    'Missouri': 24, 'Montana': 25, 'Nebraska': 26, 'Nevada': 27, 'New Hampshire': 28,
    'New Jersey': 29, 'New Mexico': 30, 'New York': 31, 'North Carolina': 32,
    'North Dakota': 33, 'Ohio': 34, 'Oklahoma': 35, 'Oregon': 36, 'Pennsylvania': 37,
    'Rhode Island': 38, 'South Carolina': 39, 'South Dakota': 40, 'Tennessee': 41,
    'Texas': 42, 'Utah': 43, 'Vermont': 44, 'Virginia': 45, 'Washington': 46,
    'West Virginia': 47, 'Wisconsin': 48, 'Wyoming': 49
}

# Define the possible crops
crop_mapping = {
    'corn': 0, 'corn_production': 1, 'corn_silage_acres': 2, 'corn_silage_yield': 3
}

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Renders HTML template

# Define route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Log the received request data
    logging.debug("Received request data: %s", request.get_json())
    
    # Parse the incoming JSON request data
    data = request.get_json()

    # Extract input data
    state = data.get('state')
    year = data.get('year')
    crop = data.get('crop')

    # These fields are included as parameters but are optional
    acres_harvested = data.get('acres_harvested', 0)  # Default to 0 if not provided
    production_measure = data.get('production_measure', 0)
    silage_acres_harvested = data.get('silage_acres_harvested', 0)
    silage_yield = data.get('silage_yield', 0)

    # Convert state and crop names to numeric values using the mappings
    state_value = state_mapping.get(state)
    crop_value = crop_mapping.get(crop)

    # Validate input values
    if state_value is None or crop_value is None:
        logging.error('Invalid state or crop selected: state=%s, crop=%s', state, crop)
        return jsonify({'error': 'Invalid state or crop selected.'}), 400

    if year < 1867 or year > 2020:
        logging.error('Invalid year: %d', year)
        return jsonify({'error': 'Year must be between 1867 and 2020.'}), 400

    # Prepare features for the prediction, including the new fields
    input_features = np.array([[year, 
                                 state_value, 
                                 acres_harvested, 
                                 production_measure, 
                                 silage_acres_harvested, 
                                 silage_yield, 
                                 crop_value]])

    # Generate the price prediction
    predicted_price = model.predict(input_features)

    # Example accuracy; replace with dynamic model accuracy if available
    accuracy = 0.9725

    # Log the prediction result
    logging.debug("Predicted price: %f, Accuracy: %f", predicted_price[0], accuracy)

    # Return prediction results as JSON
    return jsonify({
        'predicted_price': f"${predicted_price[0]:.2f}/ton",
        'accuracy': f"{accuracy * 100:.2f}%"
    })

# Run the application
if __name__ == '__main__':
    app.run(debug=True, port=5001)
