from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the pre-trained models
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
logistic_regression_model = pickle.load(open('logistic_regression_model.pkl', 'rb'))
random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    std_data = scaler.transform(input_data_reshaped)

    # SVM Prediction
    svm_prediction = svm_model.predict(std_data)[0]

    # Logistic Regression Prediction
    logistic_regression_prediction = logistic_regression_model.predict(std_data)[0]

    # Random Forest Prediction
    random_forest_prediction = random_forest_model.predict(std_data)[0]

    return render_template('index.html', 
                           svm_result='Diabetic' if svm_prediction else 'Not Diabetic',
                           logistic_result='Diabetic' if logistic_regression_prediction else 'Not Diabetic',
                           random_forest_result='Diabetic' if random_forest_prediction else 'Not Diabetic')

if __name__ == "__main__":
    app.run(debug=True)
