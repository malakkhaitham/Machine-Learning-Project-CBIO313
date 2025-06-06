import warnings
from sklearn.exceptions import InconsistentVersionWarning

# Suppress sklearn version mismatch warnings
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get features from form and convert to float
        features = [float(x) for x in request.form.values()]
        # Convert to numpy array
        final_features = np.array([features])
        # Make prediction
        prediction = model.predict(final_features)

        return render_template('index.html', prediction_text=f'Diabetes Prediction: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == '__main__':
    import os
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
