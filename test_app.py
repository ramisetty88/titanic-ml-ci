
import unittest
from app import app


class TestTitanicPredictionAPI(unittest.TestCase):

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_valid_prediction(self):
        response = self.client.post(
            "/predict",
            json={
                "Pclass": 1,
                "Sex": "female",
                "Age": 25,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 80,
                "Embarked": "C",
                "Name": "Smith, Miss. Jane"
            }
        )

        self.assertEqual(response.status_code, 200)
        result = response.get_json()
        self.assertIn(
            result["prediction"],
            ["SURVIVED", "DID NOT SURVIVE"]
        )
        self.assertIn(result["prediction_code"], [0, 1])

    def test_missing_fields(self):
        response = self.client.post(
            "/predict",
            json={"Pclass": 1, "Sex": "female"}
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("missing_fields", response.get_json())

    def test_invalid_passenger_class(self):
        response = self.client.post(
            "/predict",
            json={
                "Pclass": 5,
                "Sex": "female",
                "Age": 25,
                "SibSp": 0,
                "Parch": 0,
                "Fare": 80,
                "Embarked": "C",
                "Name": "Smith, Miss. Jane"
            }
        )

        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()
