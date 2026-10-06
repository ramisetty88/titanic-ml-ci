import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def load_data():
    data = pd.read_csv("titanic_passenger_raw_600.csv")
    return data


def train_model():

    print("Loading Titanic dataset...")

    data = load_data()

    print("Dataset loaded successfully.")
    print("Number of records:", len(data))

    # Feature engineering
    data["FamilySize"] = data["SibSp"] + data["Parch"] + 1

    data["IsAlone"] = (data["FamilySize"] == 1).astype(int)

    data["Title"] = (
        data["Name"]
        .str.extract(r",\s*([^.]*)\.")[0]
        .str.strip()
    )

    # Input features
    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "FamilySize",
        "IsAlone",
        "Title"
    ]

    X = data[features]
    y = data["Survived"]

    # Numerical features
    numerical_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "FamilySize",
        "IsAlone"
    ]

    # Categorical features
    categorical_features = [
        "Sex",
        "Embarked",
        "Title"
    ]

    # Numerical preprocessing
    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical preprocessing
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # Random Forest
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=2,
        random_state=42
    )

    # Complete pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training records:", len(X_train))
    print("Testing records :", len(X_test))

    # Train
    print("Training Titanic survival model...")

    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions)

    print("\nModel Evaluation")
    print("----------------")
    print("Accuracy:", round(accuracy, 4))

    print("\nConfusion Matrix:")
    print(matrix)

    # Save model
    joblib.dump(pipeline, "titanic_model.pkl")

    print("\nModel saved as titanic_model.pkl")

    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Metrics saved as metrics.json")

    return pipeline, accuracy


if __name__ == "__main__":
    train_model()
