import matplotlib.pyplot as plt

from predict import load_model_parameters
from utils import estimate_price, fit_z_score, read_data

DATA_COLOR = "#2f6fdb"
LINE_COLOR = "#e03e3e"


def compute_r2(mileages, prices, theta0, theta1):

    mean_price = sum(prices) / len(prices)
    ss_res = sum((y - estimate_price(x, theta0, theta1)) ** 2
                 for x, y in zip(mileages, prices))
    ss_tot = sum((y - mean_price) ** 2 for y in prices)
    return 1 - ss_res / ss_tot


def scale_thetas(theta0, theta1, x_mean, x_std, y_mean, y_std):

    scaled_theta0 = (theta0 + theta1 * x_mean - y_mean) / y_std
    scaled_theta1 = theta1 * x_std / y_std
    return scaled_theta0, scaled_theta1


def plot_regression(ax, xs, ys, theta0, theta1, *, title, xlabel, ylabel):
    x_min, x_max = min(xs), max(xs)

    ax.scatter(xs, ys, color=DATA_COLOR, alpha=0.8, edgecolor="white",
               linewidth=0.6, s=60, label="Data points", zorder=3)
    ax.plot(
        [x_min, x_max],
        [estimate_price(x_min, theta0, theta1),
         estimate_price(x_max, theta0, theta1)],
        color=LINE_COLOR, linewidth=2.5, label="Regression line", zorder=2,
    )

    # Equation box in the corner
    ax.text(
        0.97, 0.95,
        f"price = {theta0:.4g} + {theta1:.4g} × x",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="#cccccc"),
    )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="lower left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.4)


def plot_data(save_path="regression.png"):
    mileages, prices = read_data()
    if not mileages:
        raise ValueError("the training data file contains no rows")

    theta0, theta1 = load_model_parameters()
    r2 = compute_r2(mileages, prices, theta0, theta1)

    scaled_mileages, x_mean, x_std = fit_z_score(mileages)
    scaled_prices, y_mean, y_std = fit_z_score(prices)
    scaled_theta0, scaled_theta1 = scale_thetas(
        theta0, theta1, x_mean, x_std, y_mean, y_std
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_original, ax_scaled) = plt.subplots(1, 2, figsize=(14, 6))

    plot_regression(
        ax_original, mileages, prices, theta0, theta1,
        title="Original data",
        xlabel="Mileage (km)",
        ylabel="Price",
    )
    plot_regression(
        ax_scaled, scaled_mileages, scaled_prices, scaled_theta0, scaled_theta1,
        title="Z-score scaled data",
        xlabel="Scaled mileage (z-score)",
        ylabel="Scaled price (z-score)",
    )

    fig.suptitle(f"Car price regression  (R² = {r2:.3f})",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    plot_data()