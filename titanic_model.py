import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def load_data():
    data = pd.read_csv("titanic.csv")
    return data


def train_model():
    data = load_data()

    # Select input features
    features = [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked"
    ]

    X = data[features]
    y = data["Survived"]

    # Numerical and categorical columns
    numerical_features = [
        "Pclass",
        "Age",
        "SibSp",
        "Parch",
        "Fare"
    ]

    categorical_features = [
        "Sex",
        "Embarked"
    ]

    # Preprocessing for numerical data
    numerical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Preprocessing for categorical data
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    # ML model
    model = RandomForestClassifier(
        n_estimators=100,
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

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, predictions)

    print("Titanic Survival Prediction")
    print("---------------------------")
    print("Total records :", len(data))
    print("Training records :", len(X_train))
    print("Testing records :", len(X_test))
    print("Accuracy :", round(accuracy, 4))

    return pipeline, accuracy


if __name__ == "__main__":
    train_model()
