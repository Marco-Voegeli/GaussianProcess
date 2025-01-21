import os
import typing

from sklearn.gaussian_process.kernels import *
import numpy as np
import matplotlib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.model_selection import train_test_split
# Set `EXTENDED_EVALUATION` to `True` in order to visualize your predictions.
EXTENDED_EVALUATION = True
EVALUATION_GRID_POINTS = 300  # Number of grid points used in extended evaluation
EVALUATION_GRID_POINTS_3D = 50  # Number of points displayed in 3D during evaluation

# Cost function constants
THRESHOLD = 35.5
COST_W_NORMAL = 1.0
COST_W_OVERPREDICT = 5.0
COST_W_THRESHOLD = 20.0


class Model(object):
    """
    Model for this task.
    You need to implement the fit_model and predict methods
    without changing their signatures, but are allowed to create additional methods.
    """

    def __init__(self):
        """
        Initialize your model here.
        We already provide a random number generator for reproducibility.
        """
        self.kernel = 1.0 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))
        self.alpha = 1e-9
        self.rng = np.random.default_rng(seed=0)
        self.model = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True,
                                              random_state=self.rng.integers(0, 1))
        # TODO: Add custom initialization for your model here if necessary

    def set_kernel_and_alpha(self, kernel, alpha):
        self.kernel = kernel
        self.alpha = alpha
        self.model = GaussianProcessRegressor(kernel=self.kernel, normalize_y=True,
                                              random_state=self.rng.integers(0, 1))

    def predict(self, x: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict the pollution concentration for a given set of locations.
        :param x: Locations as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :return:
            Tuple of three 1d NumPy float arrays, each of shape (NUM_SAMPLES,),
            containing your predictions, the GP posterior mean, and the GP posterior stddev (in that order)
        """
        gp_mean, gp_std = self.model.predict(x, return_std=True)
        # GP to estimate the posterior mean and stddev for each location here

        predictions = gp_mean.copy()

        # Adjust the predictions to account for penalties for overprediction and underprediction
        for i, mean in enumerate(gp_mean):
            if mean < THRESHOLD:
                predictions[i] = mean + 1.96 * gp_std[i] # Using 95% confidence interval
        
        return predictions, gp_mean, gp_std
        
    def fit_model(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Fit your model on the given training data.
        :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
        :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
        """
        sub_train_x, sub_train_y = kmeansReduction(train_x, train_y) # Reduce the training set due to O(n^3) training complexity
        self.model.fit(sub_train_x, sub_train_y)


def kmeansReduction(train_x: np.ndarray, train_y: np.ndarray, n_clusters: int = 2000) -> tuple:
    """
    Using kMeans to reduce the training set due to GP complexity
    :param train_x: Training features as a 2d NumPy float array of shape (NUM_SAMPLES, 2)
    :param train_y: Training pollution concentrations as a 1d NumPy float array of shape (NUM_SAMPLES,)
    """
    cluster = KMeans(n_clusters = n_clusters)

    cluster.fit(train_x)
    sub_train_x = cluster.cluster_centers_
    sub_train_y = np.zeros(sub_train_x.shape[0], dtype=float)
    for i in range(sub_train_x.shape[0]):
        sub_train_y[i] = np.mean(train_y[np.where(cluster.labels_ == i)])
    return sub_train_x, sub_train_y

def cost_function(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Calculates the cost of a set of predictions.

    :param y_true: Ground truth pollution levels as a 1d NumPy float array
    :param y_predicted: Predicted pollution levels as a 1d NumPy float array
    :return: Total cost of all predictions as a single float
    """
    assert y_true.ndim == 1 and y_predicted.ndim == 1 and y_true.shape == y_predicted.shape

    # Unweighted cost
    cost = (y_true - y_predicted) ** 2
    weights = np.zeros_like(cost)

    # Case i): overprediction
    mask_1 = y_predicted > y_true
    weights[mask_1] = COST_W_OVERPREDICT

    # Case ii): true is above threshold, prediction below
    mask_2 = (y_true >= THRESHOLD) & (y_predicted < THRESHOLD)
    weights[mask_2] = COST_W_THRESHOLD

    # Case iii): everything else
    mask_3 = ~(mask_1 | mask_2)
    weights[mask_3] = COST_W_NORMAL

    # Weigh the cost and return the average
    return np.mean(cost * weights)


def perform_extended_evaluation(model: Model, output_dir: str = '/results'):
    """
    Visualizes the predictions of a fitted model.
    :param model: Fitted model to be visualized
    :param output_dir: Directory in which the visualizations will be stored
    """
    print('Performing extended evaluation')
    fig = plt.figure(figsize=(30, 10))
    fig.suptitle('Extended visualization of task 1')

    # Visualize on a uniform grid over the entire coordinate system
    grid_lat, grid_lon = np.meshgrid(
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
        np.linspace(0, EVALUATION_GRID_POINTS - 1, num=EVALUATION_GRID_POINTS) / EVALUATION_GRID_POINTS,
    )
    visualization_xs = np.stack((grid_lon.flatten(), grid_lat.flatten()), axis=1)
    print("Visualization_xs: ", visualization_xs.shape)
    # Obtain predictions, means, and stddevs over the entire map
    predictions, gp_mean, gp_stddev = model.predict(visualization_xs)
    predictions = np.reshape(predictions, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_mean = np.reshape(gp_mean, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))
    gp_stddev = np.reshape(gp_stddev, (EVALUATION_GRID_POINTS, EVALUATION_GRID_POINTS))

    vmin, vmax = 0.0, 65.0
    vmax_stddev = 35.5
    vmax_stddev = max(gp_stddev.flatten())

    # Plot the actual predictions
    ax_predictions = fig.add_subplot(1, 3, 1)
    predictions_plot = ax_predictions.imshow(predictions, vmin=vmin, vmax=vmax)
    ax_predictions.set_title('Predictions')
    fig.colorbar(predictions_plot)
    colormap = matplotlib.colormaps['viridis']
    # Plot the raw GP predictions with their stddeviations
    ax_gp = fig.add_subplot(1, 3, 2, projection='3d')
    ax_gp.plot_surface(
        X=grid_lon,
        Y=grid_lat,
        Z=gp_mean,
        facecolors=colormap(gp_stddev / vmax_stddev),
        rcount=EVALUATION_GRID_POINTS_3D,
        ccount=EVALUATION_GRID_POINTS_3D,
        linewidth=0,
        antialiased=False
    )
    ax_gp.set_zlim(vmin, vmax)
    ax_gp.set_title('GP means, colors are GP stddev')

    # Plot the standard deviations
    ax_stddev = fig.add_subplot(1, 3, 3)
    print("gp_stddev: ", gp_stddev)
    stddev_plot = ax_stddev.imshow(gp_stddev, vmin=vmin, vmax=vmax_stddev)
    ax_stddev.set_title('GP estimated stddev')
    fig.colorbar(stddev_plot)

    # Save figure to pdf
    figure_path = os.path.join(output_dir, 'extended_evaluation.pdf')
    fig.savefig(figure_path)
    print(f'Saved extended evaluation to {figure_path}')

    plt.show()


def cross_val(model, train_x, train_y):
    kf = KFold(n_splits=4)
    kernels = [2 * RBF(length_scale=100.0, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1)),
                1.0 * RBF(length_scale=300.0, length_scale_bounds=(1e-3, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+2)),
                Matern(1.0, (1e-4, 1e4), nu=0.5) + WhiteKernel(1.0, (1e-4, 1e2)),
                Matern(1.0, (1e-4, 1e4), nu=1.5) + WhiteKernel(1.0, (1e-4, 1e2)),
                RationalQuadratic(1.0, 1.0, (1e-5, 1e3), (1e-5, 1e3)) + WhiteKernel(1.0, (1e-4, 1e2)),
        #ExpSineSquared() + WhiteKernel(1.0, (1e-4, 1e2))
        ]
    alphas = [1e-9, 1e-10, 1e-11]
    best_cost = 1000
    best_kernel = kernels[0]
    best_alpha = alphas[0]
    print("Validation")
    best_model = model
    for kernel in kernels:
        print("Kernel: ", kernel)
        for alpha in alphas:
            current_model = Model()
            current_model.set_kernel_and_alpha(kernel, alpha)
            avg_cost = 0
            for train_index, test_index in kf.split(train_x):
                X_train, y_train = train_x[train_index], train_y[train_index]
                X_val, y_val = train_x[test_index], train_y[test_index]
                best_model.fit_model(X_train, y_train)
                y_hat = best_model.predict(X_val)[0]
                avg_cost += cost_function(y_val, y_hat)
            print("avg_cost: ")
            print(avg_cost / 3)
            avg_cost /= 3
            if avg_cost < best_cost:
                best_cost = avg_cost
                best_kernel = kernel
                best_alpha = alpha
        print("Best cost: ")
        print(best_cost)
        print(best_kernel)
        print(best_alpha)
    return best_model

def plot_data(train_x: np.ndarray, train_y: np.ndarray):
        fig = plt.figure()
        ax = plt.axes()
        ax.scatter(train_x[:,0], train_x[:,1], c=train_y)
        plt.show()

def plot_prediction_vs_true(train_x: np.ndarray, train_y: np.ndarray, predictions: np.ndarray, predicted_mean: np.ndarray, predicted_std: np.ndarray):
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))
    # Plot the predicted values and the confidence window
    ax[0].scatter(train_x[:, 0], train_x[:, 1], c=predictions, cmap='viridis', label='Predictions')
    ax[0].set_title('Predicted Values with Confidence Interval')
    ax[0].legend()

    # Plot the true train_y values
    ax[1].scatter(train_x[:, 0], train_x[:, 1], c=train_y, cmap='viridis', label='True Values')
    ax[1].set_title('True Values')
    ax[1].legend()

    plt.tight_layout()
    plt.show()

    
def main():
    # Load the training dateset and test features
    train_x = np.loadtxt('train_x.csv', delimiter=',', skiprows=1)
    train_y = np.loadtxt('train_y.csv', delimiter=',', skiprows=1)

    sub_train_x, sub_test_x, sub_train_y, sub_test_y = train_test_split(train_x, train_y, train_size=0.6)

    test_x = np.loadtxt('test_x.csv', delimiter=',', skiprows=1)

    # Fit the model
    #best_model = cross_val(Model(), sub_train_x, sub_train_y)
    print("Training on full dataset: ", train_x.shape)
    #TODO Here is the main part where we train the model on the entire dataset, idk if we should only train it on a subsample as well.

    best_model = Model()
    # kernel = Matern(1.0, (1e-5, 1e4), nu=2.5) + WhiteKernel(1.0, (1e-5, 1e2))
    # best_model.set_kernel_and_alpha(kernel, 1e-10)
    best_model.set_kernel_and_alpha(1.41**2 * RBF(length_scale=100) + WhiteKernel(noise_level=1), 1e-10)
    train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size=0.2)
    print("Training on: ", train_x.shape)

    best_model.fit_model(train_x, train_y)

    # Plot comparision of val_y and predicted_y
    predictions_val, mean_val, std_val = best_model.predict(val_x)
    print(predictions_val.shape)
    print(mean_val.shape)
    print(std_val.shape)
    print(val_y.shape)
    plot_prediction_vs_true(val_x, val_y, predictions_val, mean_val, std_val)
    # Predict on the test features
    print('Predicting on test features')
    predicted_y = best_model.predict(test_x)
    if EXTENDED_EVALUATION:
        perform_extended_evaluation(best_model, output_dir='.')


if __name__ == "__main__":
    main()


