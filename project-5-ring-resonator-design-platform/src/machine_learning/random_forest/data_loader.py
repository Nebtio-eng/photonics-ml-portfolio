"""
data_loader.py

Load the generated ring resonator dataset for
machine learning.
"""

import pandas as pd


def load_dataset(path="data/ring_resonator_dataset.csv"):
    """
    Load the dataset.

    Parameters
    ----------
    path : str
        CSV dataset location.

    Returns
    -------
    pandas.DataFrame
    """

    dataframe = pd.read_csv(path)

    return dataframe


def split_features_targets(dataframe):
    """
    Separate input features and output targets.
    """

    feature_columns = [

        "Radius_um",
        "Effective_Index",
        "Self_Coupling_t",
        "Loss_Factor_a"

    ]

    target_columns = [

        "Average_FSR_nm",
        "Average_FWHM_nm",
        "Average_Q",
        "Average_ER_dB",
        "Insertion_Loss_dB",
        "Average_Finesse"

    ]

    X = dataframe[feature_columns]

    y = dataframe[target_columns]

    return X, y