from flask import Flask,request, render_template
import numpy as np
import pandas as pd
from flask_cors import CORS

import sys
from src.utils import load_object
from src.exception import CustomException

from sklearn.preprocessing import StandardScaler
# from src.pipeline.predict_pipeline import predict
application=Flask(__name__)

app=application


preprocessor_path="artifacts/preprocessor.pkl"
model_path="artifacts/model.pkl"
preprocessor=load_object(preprocessor_path)
model=load_object(model_path)

CORS(app)

@app.route('/')
def index():
    return render_template('index.html',prediction=None)



def predict(data):
    try:
        df=pd.DataFrame(data)



        # log.info("Prediction Started...")
        # log.info(f"DATA {data}")
        scaled_data=preprocessor.transform(df)
        prediction=model.predict(scaled_data)

        # log.info(prediction)
        return prediction[0]
    except Exception as e:
        raise CustomException(e,sys)

@app.route('/submit', methods=['POST'])
def submit():
    data =   {
        "gender": [request.form['gender']],
        "race/ethnicity": [request.form['race_ethnicity']],
        "parental level of education":[ request.form['parental_education']],
        "lunch":[ request.form['lunch']],
        "test preparation course":[ request.form['test_preparation']],
        # "math score": [request.form['math_score']],
        "reading score": [request.form['reading_score']],
        "writing score": [request.form['writing_score']]
    }
    # For now, just print the data
    prediction= predict(data)

    # Here, you can process the data or save it as needed
    return render_template('index.html',prediction=prediction)

if __name__ == '__main__':
    app.run()
