from flask import Flask, render_template, request
import joblib
import pandas as pd  

app = Flask(__name__)


model = joblib.load('knn_model (1).pkl')


feature_columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']

categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
   
    print("hello")
    
    def make_predict(input_data):

        input_data = pd.DataFrame([input_data], columns=feature_columns)

        print("Input Data Columns:", input_data.columns)

       
        missing_columns = [col for col in categorical_columns if col not in input_data.columns]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
           
            for col in missing_columns:
                input_data[col] = 0

        
        input_data = pd.get_dummies(input_data, columns=categorical_columns)

     
        print("Columns after One-Hot Encoding:", input_data.columns)


        model_columns = model.feature_names_in_  
        missing_cols = set(model_columns) - set(input_data.columns)
        for col in missing_cols:
            input_data[col] = 0 

      
        input_data = input_data[model_columns]

        input_data = input_data.fillna(input_data.median())

        # Make a prediction
        prediction = model.predict(input_data)
        return prediction[0]
    
    # Prepare input data for prediction
    data = [float(request.form['age']), 
            float(request.form['sex']), 
            float(request.form['cp']), 
            float(request.form['trestbps']), 
            float(request.form['chol']), 
            float(request.form['fbs']), 
            float(request.form['restecg']), 
            float(request.form['thalach']), 
            float(request.form['exang']), 
            float(request.form['oldpeak']), 
            float(request.form['slope']), 
            float(request.form['ca']), 
            float(request.form['thal'])]
    
    # Get the prediction
    print(data)
    prediction = make_predict(data)

    # Return the result
    result = "There is a chance of heart disease" if prediction == 1 else "There is no chance of disease"
    return render_template('result.html', prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
