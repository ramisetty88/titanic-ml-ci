import json
import os
import unittest

import joblib
import pandas as pd

from titanic_model import load_data, train_model


class TestTitanicMLPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Train the model once before running the tests
        cls.model, cls.accuracy = train_model()

    def test_dataset_created(self):
        data = load_data()

        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)

    def test_model_created(self):
        self.assertTrue(os.path.exists("titanic_model.pkl"))

    def test_metrics_created(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy_is_valid(self):
        with open("metrics.json", "r") as file:
            metrics = json.load(file)

        accuracy = metrics["accuracy"]

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)

    def test_model_prediction(self):
        model = joblib.load("titanic_model.pkl")

        sample = pd.DataFrame([{
            "Pclass": 1,
            "Sex": "female",
            "Age": 25,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 80,
            "Embarked": "C"
        }])

        prediction = model.predict(sample)[0]

        self.assertIn(int(prediction), [0, 1])

    def test_survival_prediction_for_female(self):
        model = joblib.load("titanic_model.pkl")

        sample = pd.DataFrame([{
            "Pclass": 1,
            "Sex": "female",
            "Age": 25,
            "SibSp": 0,
            "Parch": 0,
            "Fare": 80,
            "Embarked": "C"
        }])

        prediction = model.predict(sample)[0]

        self.assertEqual(int(prediction), 1)

    def test_prediction_for_low_class_male(self):
        model = joblib.load("titanic_model.pkl")

        sample = pd.DataFrame([{
            "Pclass": 3,
            "Sex": "male",
            "Age": 40,
            "SibSp": 1,
            "Parch": 1,
            "Fare": 10,
            "Embarked": "S"
        }])

        prediction = model.predict(sample)[0]

        self.assertEqual(int(prediction), 0)


if __name__ == "__main__":
    unittest.main()
