
from pathlib import Path
import logging
import math
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g

# Send application logs to stdout so Render and Docker can collect them.
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    force=True,
)

logger = logging.getLogger("titanic-api")

app = Flask(__name__)

MODEL_PATH = Path(__file__).with_name("titanic_model.pkl")

FEATURES = [
    "Pclass", "Sex", "Age", "SibSp", "Parch",
    "Fare", "Embarked", "FamilySize", "IsAlone", "Title"
]

# Load the model once when the application starts.
model = None


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Titanic model file not found.")
    return joblib.load(MODEL_PATH)


logger.info("Starting Titanic survival prediction API")

try:
    model = load_model()
    logger.info("Titanic model loaded successfully")
except Exception:
    logger.exception("Model loading failed; predictions are unavailable")


@app.before_request
def start_request_tracking():
    g.request_id = uuid.uuid4().hex
    g.start_time = time.perf_counter()


@app.after_request
def log_request(response):
    request_id = getattr(g, "request_id", "unknown")
    start_time = getattr(g, "start_time", time.perf_counter())
    duration_ms = (time.perf_counter() - start_time) * 1000

    log_method = logger.info
    if response.status_code >= 400:
        log_method = logger.warning

    # Do not log passenger names or request-body contents.
    log_method(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms,
    )

    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/")
def health_check():
    return jsonify({
        "status": "ok",
        "service": "titanic-survival-prediction"
    })


@app.post("/predict")
def predict():
    global model

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A JSON request body is required"
        }), 400

    required = [
        "Pclass", "Sex", "Age", "SibSp",
        "Parch", "Fare", "Embarked", "Name"
    ]

    missing = [field for field in required if field not in data]
    if missing:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing
        }), 400

    if not isinstance(data["Name"], str) or not data["Name"].strip():
        return jsonify({
            "error": "Name must be a non-empty string"
        }), 400

    try:
        pclass = int(data["Pclass"])
        age = float(data["Age"])
        sibsp = int(data["SibSp"])
        parch = int(data["Parch"])
        fare = float(data["Fare"])
    except (ValueError, TypeError, OverflowError):
        return jsonify({
            "error": "Numeric fields contain invalid values"
        }), 400

    if pclass not in [1, 2, 3]:
        return jsonify({
            "error": "Pclass must be 1, 2, or 3"
        }), 400

    if (
        age < 0
        or sibsp < 0
        or parch < 0
        or fare < 0
        or not math.isfinite(age)
        or not math.isfinite(fare)
    ):
        return jsonify({
            "error": "Numeric values are outside valid ranges"
        }), 400

    sex = str(data["Sex"]).strip().lower()
    embarked = str(data["Embarked"]).strip().upper()

    if sex not in ["male", "female"]:
        return jsonify({
            "error": "Sex must be male or female"
        }), 400

    if embarked not in ["C", "Q", "S"]:
        return jsonify({
            "error": "Embarked must be C, Q, or S"
        }), 400

    name = data["Name"].strip()
    title_match = pd.Series([name]).str.extract(
        r",\s*([^.]*)\."
    )[0].iloc[0]

    if pd.isna(title_match):
        return jsonify({
            "error": "Name must include a title, e.g. Smith, Mr. John"
        }), 400

    family_size = sibsp + parch + 1

    sample = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked,
        "FamilySize": family_size,
        "IsAlone": int(family_size == 1),
        "Title": title_match.strip()
    }], columns=FEATURES)

    if model is None:
        logger.error(
            "Prediction unavailable because the model did not load; "
            "request_id=%s",
            g.request_id,
        )
        return jsonify({
            "error": "Prediction service is temporarily unavailable"
        }), 500

    try:
        prediction = int(model.predict(sample)[0])
    except Exception:
        logger.exception(
            "Prediction failed; request_id=%s",
            g.request_id,
        )
        return jsonify({
            "error": "An internal prediction error occurred"
        }), 500

    logger.info(
        "Prediction completed; request_id=%s prediction_code=%s",
        g.request_id,
        prediction,
    )

    return jsonify({
        "prediction": (
            "SURVIVED" if prediction == 1
            else "DID NOT SURVIVE"
        ),
        "prediction_code": prediction
    })


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
