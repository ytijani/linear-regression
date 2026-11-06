# ft_linear_regression

A small Python implementation of linear regression that predicts a car's price from its mileage. The model is trained with gradient descent and uses only Python's standard library, except for the optional graph visualization, which uses Matplotlib.

## Requirements

- Python 3.9 or newer
- Matplotlib for `visualization.py`

Install Matplotlib with:

```bash
python3 -m pip install matplotlib
```

## Dataset

The training data is in `data/data.csv`. It must be a CSV file with these columns:

```csv
km,price
240000,3650
139800,3800
```

The data file path and column names are configured in `constants.py`.

## Train the model

From the project directory, run:

```bash
python3 train.py
```

Training normalizes the mileage and price values, learns the model parameters with gradient descent, converts the parameters back to the original units, and saves them in `model_parameters.json`.

The model uses this equation:

```text
predicted price = theta0 + theta1 * mileage
```

## Predict a price

Train the model first, then run:

```bash
python3 predict.py
```

Enter the car's mileage in kilometres when prompted. Commas are accepted, for example `240,000`.

## Visualize the data and regression

After training, run:

```bash
python3 visualization.py
```

This displays two graphs side by side: the original mileage and price values, and their z-score scaled values. Both graphs show the data points and regression line. The visualization also displays the model's R² score and saves the graph to `regression.png`.

## Evaluate the model

After training, run:

```bash
python3 evaluate.py
```

The program reports:

- **MAE**: average absolute difference between predicted and actual prices.
- **RMSE**: square root of the average squared prediction error; larger errors count more heavily.
- **R²**: how well the predictions fit the observed prices; a value closer to 1 indicates a better fit.

These metrics are calculated on the same dataset used for training, so they describe the model's fit to that data.

## Project files

| File | Purpose |
| --- | --- |
| `data/data.csv` | Mileage and price training data |
| `constants.py` | Dataset path, CSV column names, and training settings |
| `utils.py` | Data loading, normalization, and price estimation helpers |
| `train.py` | Trains the model and saves its parameters |
| `model_parameters.json` | Saved `theta0` and `theta1` values |
| `predict.py` | Predicts a price for mileage entered by the user |
| `visualization.py` | Plots original and scaled data with the regression line |
| `evaluate.py` | Calculates MAE, RMSE, and R² |
