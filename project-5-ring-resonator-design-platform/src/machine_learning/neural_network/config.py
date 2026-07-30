# ==========================================================
# Neural Network Configuration
# ==========================================================

# Dataset
DATASET_PATH = "data/ring_resonator_dataset.csv"

# Model directory
MODEL_DIRECTORY = "models/neural_network"

MODEL_PATH = f"{MODEL_DIRECTORY}/ring_surrogate.pt"
SCALER_PATH = f"{MODEL_DIRECTORY}/input_scaler.pkl"

# Results
RESULTS_DIRECTORY = "results/neural_network"

# Training
TEST_SIZE = 0.20
RANDOM_STATE = 42

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 200

# Early stopping
PATIENCE = 20
MIN_DELTA = 1e-6

# Network architecture
INPUT_SIZE = 4
HIDDEN_LAYERS = [128, 256, 512]
OUTPUT_SIZE = 1024

# Input scaler
SCALER_PATH = f"{MODEL_DIRECTORY}/input_scaler.pkl"

# Output scaler
OUTPUT_SCALER_PATH = f"{MODEL_DIRECTORY}/output_scaler.pkl"