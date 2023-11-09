import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from flask import Flask, jsonify, render_template,request

application = Flask(__name__)
app = application

# import ridge regressor and standard scaler pickle
logistic_reg_model = pickle.load(open("./models/Prediction_Model.pkl", "rb"))
standard_scaler = pickle.load(open("./models/StandardScaler.pkl", "rb"))

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method == "POST":
        Pregnancies = float(request.form.get("Pregnancies"))
        Glucose = float(request.form.get("Glucose"))
        BloodPressure = float(request.form.get("BloodPressure"))
        SkinThickness = float(request.form.get("SkinThickness"))
        Insulin = float(request.form.get("Insulin"))
        BMI = float(request.form.get("BMI"))
        DiabetesPedigreeFunction = float(request.form.get("DiabetesPedigreeFunction"))
        Age = float(request.form.get("Age"))

        new_datapoint = standard_scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        result = logistic_reg_model.predict(new_datapoint)

        if result[0] == 1:
            result = "Diabetes"
        else:
            result = "No Diabetes"

        return render_template("home.html",results=result)

    else:
        return render_template("home.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
