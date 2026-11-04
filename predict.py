

import json

from utils import estimate_price

from constants import MODEL_FILE


def load_model_parameters(filename=MODEL_FILE):
    with open(filename) as file:
        parameters = json.load(file)

    try:
        theta0 = float(parameters["theta0"])
        theta1 = float(parameters["theta1"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("model file must contain numeric theta0 and theta1") from error

    return theta0, theta1


def predict_price(mileage, theta0, theta1):
    return estimate_price(mileage, theta0, theta1)


if __name__ == "__main__":
    theta0, theta1 = load_model_parameters()

    try:
      raw_mileage = input("Enter mileage (km): ").strip()
      mileage = float(raw_mileage.replace(",", ""))
      if mileage < 0:
         raise ValueError("mileage cannot be negative")
    except ValueError as error:
        raise SystemExit(f"Invalid mileage: {error}")

    price = predict_price(mileage, theta0, theta1)
    print(f"Estimated price: {price:.2f}")
