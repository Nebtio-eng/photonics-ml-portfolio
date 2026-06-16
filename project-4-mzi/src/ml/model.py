from sklearn.neural_network import MLPRegressor

def build_model():
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        max_iter=500
    )
    return model
