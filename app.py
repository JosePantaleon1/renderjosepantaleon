from flask import Flask, render_template, request, jsonify
import os
import pickle

app = Flask(__name__)

# Load the trained Titanic model
model_path = os.path.join(os.getcwd(), "titanic_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Route for the home page with the prediction form
@app.route("/")
def home():
    return render_template("form.html")

# Route to handle form submission and make predictions
@app.route("/predict_form", methods=["POST"])
def predict_form():
    # Collect input data from the form
    pclass = int(request.form["Pclass"])
    sex = 0 if request.form["Sex"] == "male" else 1  # Encode gender
    age = float(request.form["Age"])

    # Make a prediction using the model
    prediction = model.predict([[pclass, sex, age]])[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"

    # Return the result to the user on the form page
    return render_template("form.html", prediction=result)

# API route to handle JSON requests for predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Collect JSON data sent in the request
    data = request.get_json()
    pclass = data["Pclass"]
    sex = 0 if data["Sex"] == "male" else 1  # Encode gender
    age = data["Age"]

    # Make a prediction using the model
    prediction = model.predict([[pclass, sex, age]])[0]
    result = "Survived" if prediction == 1 else "Did Not Survive"

    # Return the result as a JSON response
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)

