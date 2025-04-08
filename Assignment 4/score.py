import joblib
from sklearn.pipeline import Pipeline

def score(text: str, model: Pipeline, threshold: float) -> tuple[bool, float]:
    """Score text for spam/ham."""
    propensity = model.predict_proba([text])[0][1]  # P(spam)
    prediction = bool(propensity >= threshold)
    return prediction, propensity

# Example usage (for testing)
if __name__ == "__main__":
    model = joblib.load("spam_model.joblib")
    print(score("WIN A FREE IPHONE!", model, 0.5))  # Should return (True, ~0.99)
    print(score("Hello, how are you?", model, 0.5))  # Should return (False, ~0.01)
