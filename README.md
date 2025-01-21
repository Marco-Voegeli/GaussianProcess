# Gaussian Process Regression for Pollution Prediction

This repository contains a Python implementation of a Gaussian Process Regression model designed to predict pollution concentrations based on geographic locations. The code includes functionalities for training the model, making predictions, and visualizing results.

## Features

- **Gaussian Process Regression** using `sklearn` with customizable kernels and hyperparameters.
- **Data Reduction** through k-Means clustering to address computational complexity.
- **Custom Cost Function** for evaluating prediction performance with penalties for over/under-predictions.
- **Visualization** tools for exploring predictions, GP mean, and GP standard deviations.
- **Cross-validation** for kernel and hyperparameter optimization.
- **Extended Evaluation** that generates visual outputs stored as a PDF.

## Prerequisites

To run this project, you need the following Python packages:

- `numpy`
- `scikit-learn`
- `matplotlib`

You can install them using:

```bash
pip install numpy scikit-learn matplotlib
```

## Usage

1. Place your training data in `train_x.csv` and `train_y.csv`, and your test data in `test_x.csv`. Ensure the data is properly formatted as comma-separated values (CSV).
2. Run the main script:

```bash
python solution.py
```

3. If `EXTENDED_EVALUATION` is set to `True` in the script, a PDF file `extended_evaluation.pdf` will be generated in the output directory.

## File Structure

- `solution.py`: Main script containing the implementation of the Gaussian Process Regression model and related functionalities.
- `train_x.csv`, `train_y.csv`, `test_x.csv`: Example CSV files for input data.
- `extended_evaluation.pdf`: Generated visualization report.

## Extended Evaluation

The `extended_evaluation.pdf` file contains:

- Predictions across a grid of geographic locations.
- GP posterior mean and standard deviation visualizations in both 2D and 3D.
- Color-coded plots showing model confidence.

### Example PDF Display

![Extended Evaluation](extended_evaluation.pdf)

## Author

This project was implemented to address a task involving pollution prediction using advanced machine learning methods.

## License

This project is licensed under the MIT License.
