
from predict import load_model_parameters, predict_price
from utils import read_data


def _validate_values(predictions: list[float], actual_values: list[float]) -> None:
    if not predictions or len(predictions) != len(actual_values):
        raise ValueError(
            "predictions and actual_values must be non-empty and equal in length"
        )


def mean_absolute_error(
    predictions: list[float], actual_values: list[float]
) -> float:
    """Return the average absolute prediction error, in the target's units."""
    _validate_values(predictions, actual_values)
    return sum(
        abs(prediction - actual)
        for prediction, actual in zip(predictions, actual_values)
    ) / len(actual_values)


def calculate_root_mean_squared_error(
    predictions: list[float], actual_values: list[float]
) -> float:

    _validate_values(predictions, actual_values)
    mean_squared_error = sum(
        (prediction - actual) ** 2
        for prediction, actual in zip(predictions, actual_values)
    ) / len(actual_values)
    return mean_squared_error ** 0.5


def calculate_r_squared(
    predictions: list[float], actual_values: list[float]
) -> float:

    _validate_values(predictions, actual_values)
    actual_mean = sum(actual_values) / len(actual_values)
    total_sum_of_squares = sum(
        (actual - actual_mean) ** 2 for actual in actual_values
    )
    if total_sum_of_squares == 0:
        raise ValueError("R² is undefined when all actual values are identical")

    residual_sum_of_squares = sum(
        (actual - prediction) ** 2
        for prediction, actual in zip(predictions, actual_values)
    )
    return 1 - residual_sum_of_squares / total_sum_of_squares


def evaluate_model(data_file=None, model_file=None):

    mileages, actual_prices = read_data(data_file) if data_file else read_data()
    if not mileages:
        raise ValueError("the training data file contains no rows")

    theta0, theta1 = (
        load_model_parameters(model_file) if model_file else load_model_parameters()
    )
    predictions = [predict_price(mileage, theta0, theta1) for mileage in mileages]

    return {
        "mae": mean_absolute_error(predictions, actual_prices),
        "rmse": calculate_root_mean_squared_error(predictions, actual_prices),
        "r_squared": calculate_r_squared(predictions, actual_prices),
    }


if __name__ == "__main__":
    metrics = evaluate_model()
    print("Evaluation on the training data:")
    print(f"Mean absolute error (MAE): {metrics['mae']:.2f}")
    print(f"Root mean squared error (RMSE): {metrics['rmse']:.2f}")
    print(f"R² score: {metrics['r_squared']:.4f}")
