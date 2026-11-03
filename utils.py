



def estimate_price(mileage: float, theta0: float, theta1: float) -> float:
    return theta0 + theta1 * mileage


def mean(values):
    if not values:
        raise ValueError("values must not be empty")
    return sum(values) / len(values)


def standard_deviation(values):
    avg = mean(values)
    variance = sum((x - avg) ** 2 for x in values) / len(values)
    return variance ** 0.5


def fit_z_score(values):
    values_mean = mean(values)
    values_std = standard_deviation(values)
    if values_std == 0:
        raise ValueError("cannot z-score values with zero standard deviation")
    normalized = [(value - values_mean) / values_std for value in values]
    return normalized, values_mean, values_std


def apply_z_score(value, values_mean, values_std):
    if values_std == 0:
        raise ValueError("standard deviation must not be zero")
    return (value - values_mean) / values_std


def reverse_z_score(value, values_mean, values_std):
    return value * values_std + values_mean


def normalize(values):
    normalized, _, _ = fit_z_score(values)
    return normalized
