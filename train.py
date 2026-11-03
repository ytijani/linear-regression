import csv

from constants import (
    DATA_FILE,
    INITIAL_THETA0,
    INITIAL_THETA1,
    LEARNING_RATE,
    MAX_ITERATIONS,
    MILEAGE_KEY,
    PRICE_KEY,
)
from utils import estimate_price, fit_z_score


def read_data(filename=DATA_FILE):
   
    mileage = []
    prices = []
    with open(filename, newline="") as csvfile:
        for row in csv.DictReader(csvfile):
            mileage.append(float(row[MILEAGE_KEY]))
            prices.append(float(row[PRICE_KEY]))
    return mileage, prices


def normalize_data(filename=DATA_FILE):

    mileage, prices = read_data(filename)
    normalized_mileage, mileage_mean, mileage_std = fit_z_score(mileage)
    normalized_prices, price_mean, price_std = fit_z_score(prices)
    return (
        normalized_mileage,
        normalized_prices,
        mileage_mean,
        mileage_std,
        price_mean,
        price_std,
    )


def get_mean_squared_error(mileages, prices, theta0, theta1):
    if not mileages or len(mileages) != len(prices):
        raise ValueError("mileage and price data must be non-empty and equal in length")
    return sum(
        (estimate_price(mileage, theta0, theta1) - price) ** 2
        for mileage, price in zip(mileages, prices)
    ) / len(mileages)


def gradient_bias(theta0, theta1, mileages, prices):
    count = len(mileages)
    if count == 0 or count != len(prices):
        raise ValueError("mileage and price data must be non-empty and equal in length")
    return sum(
        estimate_price(mileage, theta0, theta1) - price
        for mileage, price in zip(mileages, prices)
    ) / count


def gradient_slope(theta0, theta1, mileages, prices):
    count = len(mileages)
    if count == 0 or count != len(prices):
        raise ValueError("mileage and price data must be non-empty and equal in length")
    return sum(
        (estimate_price(mileage, theta0, theta1) - price) * mileage
        for mileage, price in zip(mileages, prices)
    ) / count


def train_model(
    mileages,
    prices,
    learning_rate=LEARNING_RATE,
    iterations=MAX_ITERATIONS,
    initial_theta0=INITIAL_THETA0,
    initial_theta1=INITIAL_THETA1,
):

    if not mileages or len(mileages) != len(prices):
        raise ValueError("mileage and price data must be non-empty and equal in length")

    theta0 = initial_theta0
    theta1 = initial_theta1
    for _ in range(iterations):
        bias_gradient = gradient_bias(theta0, theta1, mileages, prices)
        slope_gradient = gradient_slope(theta0, theta1, mileages, prices)
        theta0 -= learning_rate * bias_gradient
        theta1 -= learning_rate * slope_gradient
    return theta0, theta1


if __name__ == "__main__":
    mileage, prices, _, _, _, _ = normalize_data()
    theta0, theta1 = train_model(mileage, prices)
    print(f"Training complete: theta0={theta0:.6f}, theta1={theta1:.6f}")
