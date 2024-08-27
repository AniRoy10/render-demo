from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your model
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract data from form
        IQ = float(request.form['IQ'])
        CGPA = float(request.form['CGPA'])
        _10th_Marks = float(request.form['10th_Marks'])
        _12th_Marks = float(request.form['12th_Marks'])
        Communication_Skills = float(request.form['Communication_Skills'])
        
        # Prepare feature array
        features = np.array([[IQ, CGPA, _10th_Marks, _12th_Marks, Communication_Skills]])
        
        # Make prediction
        prediction = model.predict(features)
        output = 'Placed' if prediction[0] == 1 else 'Not Placed'

        return render_template('index.html', prediction_text=f'Prediction: {output}')
    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', prediction_text='Error in prediction.')

if __name__ == '__main__':
    app.run(debug=True)
