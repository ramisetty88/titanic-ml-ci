import unittest
import pandas as pd

from titanic_model import load_data, train_model


class TestTitanicModel(unittest.TestCase):

    def test_dataset_exists_and_loads(self):
        data = load_data()

        self.assertIsInstance(data, pd.DataFrame)
        self.assertGreater(len(data), 0)

    def test_required_columns_exist(self):
        data = load_data()

        required_columns = [
            "Pclass",
            "Sex",
            "Age",
            "SibSp",
            "Parch",
            "Fare",
            "Embarked",
            "Survived"
        ]

        for column in required_columns:
            self.assertIn(column, data.columns)

    def test_model_trains_successfully(self):
        model, accuracy = train_model()

        self.assertIsNotNone(model)

    def test_accuracy_is_valid(self):
        model, accuracy = train_model()

        self.assertGreaterEqual(accuracy, 0.0)
        self.assertLessEqual(accuracy, 1.0)


if __name__ == "__main__":
    unittest.main()
