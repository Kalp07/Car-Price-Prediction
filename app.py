from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Check if model and encoder files exist before loading
if not os.path.exists("linear_regression_car_sale_price_model.pkl") or \
   not os.path.exists("label_encoders.pkl"):
    print("Error: Required model and/or encoder files are missing.")
    exit()

try:
    # Load model + encoders
    model = joblib.load("linear_regression_car_sale_price_model.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    exit()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        input_data = {
            'yr_mfr': int(request.form['yr_mfr']),
            'fuel_type': request.form['fuel_type'],
            'kms_run': int(request.form['kms_run']),
            'transmission': request.form['transmission'],
            'body_type': request.form['body_type'],
            'total_owners': int(request.form['total_owners']),
            'make': request.form['make'],
            'model': request.form['model'],
            'car_rating': request.form['car_rating'],
            'original_price': float(request.form['original_price']),
            'warranty_avail': request.form['warranty_avail']
        }

        df_input = pd.DataFrame([input_data])

        # Apply same encoding as training
        for col in label_encoders:
            if col in df_input.columns:
                df_input[col] = label_encoders[col].transform(df_input[col].astype(str))

        # Ensure input features match the model's expected features
        # This handles cases where the user's input might not have the same order as the training data
        df_input = df_input[model.feature_names_in_]

        # Predict price
        pred_price = model.predict(df_input)[0]

        return render_template('index.html', prediction=f"₹{pred_price:,.0f}")

    except KeyError as e:
        # Handle missing form data
        return f"Error: Missing form field - {e}", 400
    except Exception as e:
        # Handle other potential errors
        return f"An error occurred: {e}", 500

if __name__ == "__main__":
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
