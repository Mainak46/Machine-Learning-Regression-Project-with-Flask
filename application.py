from flask import Flask, render_template, request
import numpy as np
import pickle

application = Flask(__name__)
app = application

# Load model and scaler
ridge_model = pickle.load(open("models/ridge.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.route("/predictdata", methods=["POST"])
def predict_datapoints():

    Temperature = float(request.form.get("Temperature"))
    RH = float(request.form.get("RH"))
    Ws = float(request.form.get("Ws"))
    Rain = float(request.form.get("Rain"))
    FFMC = float(request.form.get("FFMC"))
    DMC = float(request.form.get("DMC"))
    DC = float(request.form.get("DC"))
    ISI = float(request.form.get("ISI"))
    BUI = float(request.form.get("BUI"))      
    Classes = float(request.form.get("Classes"))
    Region = float(request.form.get("Region"))

    # Create input array in correct order
    input_data = np.array([[Temperature, RH, Ws, Rain,FFMC, DMC, DC, ISI,BUI,Classes, Region]])

    # Scale input
    input_scaled = scaler.transform(input_data)

    # Predict FWI
    prediction = round(ridge_model.predict(input_scaled)[0], 2)

    print("Predicted FWI:", prediction)

    return render_template("home.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
