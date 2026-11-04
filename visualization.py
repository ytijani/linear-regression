import matplotlib.pyplot as plt

from utils import read_data


def plot_data():
    mileages, prices = read_data()

    plt.scatter(mileages, prices)

    plt.xlabel("Mileage (km)")
    plt.ylabel("Price")
    plt.title("Car Mileage vs Price")

    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    plot_data()
