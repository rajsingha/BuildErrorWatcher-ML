import unittest
import joblib

class TestErrorClassifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the trained classifier
        cls.model = joblib.load("../error_classifier.joblib")

    def test_error_line(self):
        # A known error line should predict 1 (error)
        sample = " cannot find symbol"
        pred = self.model.predict([sample])[0]
        self.assertEqual(pred, 1, f"Expected error prediction for '{sample}'")

    def test_noise_line(self):
        # A known noise line should predict 0 (noise)
        sample = "Version available"
        pred = self.model.predict([sample])[0]
        self.assertEqual(pred, 0, f"Expected noise prediction for '{sample}'")

    def test_probability_threshold(self):
        # Probabilities for an error line should strongly favor the error class
        sample = "[ERROR] Build failed with exit code 1"
        probs = self.model.predict_proba([sample])[0]
        # probs = [prob_noise, prob_error]
        self.assertGreater(probs[1], 0.5, f"Error probability too low: {probs[1]}")

if __name__ == "__main__":
    unittest.main()
