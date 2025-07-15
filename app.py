from flask import Flask, render_template, request
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load data and model
car = pd.read_csv(r"E:\Data_Science\Car_Prediction_App\car.csv")
model = pickle.load(open(r"E:\Data_Science\Car_Prediction_App\NewLinearRegressionModel.pkl", "rb"))

# Build company->models mapping for dropdown logic
company_models = car.groupby("company")["name"].unique().apply(list).to_dict()

@app.route("/", methods=["GET", "POST"])
def index():
    companies = sorted(car["company"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_type = car["fuel_type"].unique()
    prediction = None

    # This dictionary holds the current (or default) form values
    form_values = {
        "company": "",
        "car_model": "",
        "year": "",
        "fuel_type": "",
        "kilo_driven": ""
    }

    if request.method == "POST":
        try:
            # Retain submitted values for redisplay
            form_values["company"] = request.form["company"]
            form_values["car_model"] = request.form["car_model"]
            form_values["year"] = request.form["year"]
            form_values["fuel_type"] = request.form["fuel_type"]
            form_values["kilo_driven"] = request.form["kilo_driven"]

            selected_company = form_values["company"]
            selected_model = form_values["car_model"]
            selected_year = int(form_values["year"])
            selected_fuel = form_values["fuel_type"]
            kilo_driven = int(form_values["kilo_driven"])

            input_data = pd.DataFrame([[selected_model, selected_company, selected_year, kilo_driven, selected_fuel]],
                                      columns=["name", "company", "year", "kms_driven", "fuel_type"])

            pred = model.predict(input_data)[0]

            # --- Hardcoded INR→USD conversion ---
            usd_per_inr = 1 / 83  # 1 USD = ₹83 (update as needed)
            pred_usd = pred * usd_per_inr

            if pred_usd < 0:
                prediction = (
                "Predicted Price: $0.00 USD /n"
                "Wrong Entry! Please Try Again."
                            )
            else:
                prediction = f"Predicted Price: ${pred_usd:,.2f} USD"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        companies=companies,
        year=year,
        fuel_type=fuel_type,
        company_models=company_models,
        prediction=prediction,
        form_values=form_values
    )

if __name__ == "__main__":
    app.run(debug=True)
