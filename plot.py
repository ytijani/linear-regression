"""Plot the training data and the regression line saved by train.py."""

import matplotlib.pyplot as plt

from constants import DATA_FILE, MILEAGE_KEY, MODEL_FILE, PRICE_KEY
from predict import load_model_parameters
from train import read_data
from utils import estimate_price


def plot_training_data(data_file=DATA_FILE, model_file=MODEL_FILE):
    mileages, prices = read_data(data_file)
    if not mileages:
        raise ValueError("the training data file contains no rows")

    theta0, theta1 = load_model_parameters(model_file)
    x_min = min(mileages)
    x_max = max(mileages)

    plt.scatter(mileages, prices, label="Training data", color="blue")
    plt.plot(
        [x_min, x_max],
        [estimate_price(x_min, theta0, theta1), estimate_price(x_max, theta0, theta1)],
        label="Regression line",
        color="red",
    )
    plt.xlabel(MILEAGE_KEY)
    plt.ylabel(PRICE_KEY)
    plt.title("Car price vs mileage")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_training_data()
