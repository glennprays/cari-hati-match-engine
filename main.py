from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import json

app = Flask(__name__)

kmeans = joblib.load("kmeans_model.pkl")

scaler = joblib.load("scaler.pkl")

with open("passions.json", "r") as f:
    all_passions = json.load(f)

class_names = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E"}


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        user_profile = [
            1 if passion in data["passions"] else 0 for passion in all_passions
        ]
        user_profile_array = np.array([user_profile])
        scaled_user_profile = scaler.transform(user_profile_array)
        cluster_label = kmeans.predict(scaled_user_profile)[0]
        class_name = class_names.get(cluster_label, "Unknown")
        return jsonify({"class_name": class_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)
