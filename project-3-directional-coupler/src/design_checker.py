import numpy as np
import joblib
import os

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "..",
    "results",
    "models",
    "coupler_surrogate.pkl"
)

model = joblib.load(MODEL_PATH)

def check_design(kappa, L, noise=0.1, samples=200, wavelength_shift=0.0):
    kappa_eff = kappa * (1 + wavelength_shift)

    nominal_split = model.predict([[kappa_eff, L]])[0]

    noisy_kappa = kappa_eff * (1 + noise * np.random.randn(samples))
    splits = np.sin(noisy_kappa * L) ** 2

    alpha = 0.01  # propagation loss
    splits *= np.exp(-alpha * L)

    mean_error = np.mean(np.abs(splits - 0.5))
    std_error = np.std(splits)

    verdict = True
    reasons = []

    if mean_error >= 0.03:
        verdict = False
        reasons.append("Mean split deviates from 50:50")

    if std_error >= 0.05:
        verdict = False
        reasons.append("Design is fabrication-sensitive")

    return {
        "nominal_split": nominal_split,
        "mean_error": mean_error,
        "std_error": std_error,
        "passes": verdict,
        "reasons": reasons
    }
