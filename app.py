from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__, template_folder='templates')

# Load the saved models
Deep_Neural_Network_model=joblib.load("dnn_model.pkl")
logistic_reg_model = joblib.load('logistic_regression_model.pkl')
knn_model = joblib.load('knn_model.pkl')
naive_bayes_model = joblib.load('naive_bayes_model.pkl')
svm_model = joblib.load('svm_model.pkl')
decision_tree_model = joblib.load('decision_tree_model.pkl')
xgboost_model = joblib.load('xgboost_model.pkl')
stocahstic_gradient_model=joblib.load('stocahstic_gradient_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    predictions = {}
    if request.method == 'POST':
        input_data = request.form['input_data']
        input_data_list = [float(x.strip()) for x in input_data.split(',')]
        
        # Perform predictions using different models
        predictions['dnn'] = 'Benign' if make_prediction(input_data_list, Deep_Neural_Network_model) == 0 else 'Malicious'
        predictions['logistic_reg'] = 'Benign' if make_prediction(input_data_list, logistic_reg_model) == 0 else 'Malicious'
        predictions['knn'] = 'Benign' if make_prediction(input_data_list, knn_model) == 0 else 'Malicious'
        predictions['naive_bayes'] = 'Benign' if make_prediction(input_data_list, naive_bayes_model) == 0 else 'Malicious'
        predictions['svm'] = 'Benign' if make_prediction(input_data_list, decision_tree_model) == 0 else 'Malicious'
        predictions['decision_tree'] = 'Benign' if make_prediction(input_data_list, stocahstic_gradient_model) == 0 else 'Malicious'
        predictions['xgboost'] = 'Benign' if make_prediction(input_data_list, xgboost_model) == 0 else 'Malicious'
        predictions['stocahstic'] = 'Benign' if make_prediction(input_data_list, stocahstic_gradient_model) == 0 else 'Malicious'
       
        print("Predictions:", predictions)
        
        return render_template('index.html', predictions=predictions)
    
    return render_template('index.html')

def make_prediction(input_data, model):
    try:
        # Perform the prediction using the model
        input_data_reshaped = np.array(input_data).reshape(1, -1)
        prediction = model.predict(input_data_reshaped)
        print("Prediction:", prediction)
        
        return prediction[0]
    except ValueError:
        # Handle the case where input data cannot be converted to numbers
        print(ValueError)
        return None

if __name__ == '__main__':
    app.run(debug=True)
    app.run(host="0.0.0.0, port=80)
