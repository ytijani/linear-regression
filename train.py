import json
from constants import (
    DATA_FILE,
    INITIAL_THETA0,
    INITIAL_THETA1,
    LEARNING_RATE,
    MAX_ITERATIONS,
    MILEAGE_KEY,
    PRICE_KEY,
)
from utils import estimate_price, fit_z_score, read_data


def save_model_parameters(theta0,theta1):
    parameters = {
        "theta0": theta0,
        "theta1": theta1,
    }
    with open("model_parameters.json", "w") as file:
        json.dump(parameters, file, indent=4)




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
    m = len(mileages)
    if m == 0 or m != len(prices):
        raise ValueError("mileage and price data must be non-empty and equal in length")
    return sum(
        estimate_price(mileage, theta0, theta1) - price
        for mileage, price in zip(mileages, prices)
    ) / m


def gradient_slope(theta0, theta1, mileages, prices):
    m = len(mileages)
    if m == 0 or m != len(prices):
        raise ValueError("mileage and price data must be non-empty and equal in length")
    return sum(
        (estimate_price(mileage, theta0, theta1) - price) * mileage
        for mileage, price in zip(mileages, prices)
    ) / m


def convert_to_original_scale(
    theta0,
    theta1,
    mileage_mean,
    mileage_std,
    price_mean,
    price_std,
):
    if mileage_std <= 0 or price_std <= 0:
        raise ValueError("Standard deviations must be positive")

    original_theta1 = (theta1 * price_std) / mileage_std

    original_theta0 = price_mean + price_std * (
        theta0 - (theta1 * mileage_mean) / mileage_std
    )

    return original_theta0, original_theta1


def train_model(
    mileages,
    prices,
    learning_rate=LEARNING_RATE,
    iterations=MAX_ITERATIONS,
    initial_theta0=INITIAL_THETA0,
    initial_theta1=INITIAL_THETA1,
):
    if not mileages or len(mileages) != len(prices):
        raise ValueError("Mileage and price data must be non-empty and equal in length")

    if learning_rate <= 0:
        raise ValueError("Learning rate must be positive")

    if iterations <= 0:
        raise ValueError("Iterations must be positive")

    theta0 = initial_theta0
    theta1 = initial_theta1

    for _ in range(iterations):
        bias_gradient = gradient_bias(theta0, theta1, mileages, prices)
        slope_gradient = gradient_slope(theta0, theta1, mileages, prices)

        theta0 -= learning_rate * bias_gradient
        theta1 -= learning_rate * slope_gradient

    return theta0, theta1



if __name__ == "__main__":
    (
        mileage,
        prices,
        mileage_mean,
        mileage_std,
        price_mean,
        price_std,
    ) = normalize_data()

    theta0, theta1 = train_model(mileage, prices)

    original_theta0, original_theta1 = convert_to_original_scale(
        theta0,
        theta1,
        mileage_mean,
        mileage_std,
        price_mean,
        price_std,
    )

    save_model_parameters(original_theta0, original_theta1)
    print(
        f"Training complete: theta0={original_theta0:.6f}, "
        f"theta1={original_theta1:.6f}"
    )

