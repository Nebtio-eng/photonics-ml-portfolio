import torch
import torch.nn as nn

import config


class RingResonatorNetwork(nn.Module):
    """
    Feedforward neural network for predicting
    a ring resonator transmission spectrum.
    """

    def __init__(self):

        super().__init__()

        if len(config.HIDDEN_LAYERS) != 3:
            raise ValueError(
                "config.HIDDEN_LAYERS must contain exactly three values."
            )

        self.network = nn.Sequential(

        nn.Linear(
        config.INPUT_SIZE,
        config.HIDDEN_LAYERS[0]
        ),
        nn.BatchNorm1d(
        config.HIDDEN_LAYERS[0]
        ),
        nn.ReLU(),

        nn.Linear(
            config.HIDDEN_LAYERS[0],
            config.HIDDEN_LAYERS[1]
        ),
        nn.BatchNorm1d(
            config.HIDDEN_LAYERS[1]
        ),
        nn.ReLU(),

        nn.Linear(
            config.HIDDEN_LAYERS[1],
            config.HIDDEN_LAYERS[2]
        ),
        nn.BatchNorm1d(
            config.HIDDEN_LAYERS[2]
        ),
        nn.ReLU(),

        nn.Linear(
            config.HIDDEN_LAYERS[2],
            config.OUTPUT_SIZE
        )

)
    def forward(self, x):
        """
        Forward pass through the network.
        """

        return self.network(x)