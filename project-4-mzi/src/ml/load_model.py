from joblib import load

model = load("models/mzi_surrogate.joblib")

print(type(model))
print("Model loaded successfully")