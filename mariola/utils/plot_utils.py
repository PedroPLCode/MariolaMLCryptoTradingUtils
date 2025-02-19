import matplotlib.pyplot as plt
from utils.logger_utils import log
from utils.exception_handler import exception_handler
from typing import Union


@exception_handler()
def visualise_model_prediction(y_pred: Union[list, np.ndarray]) -> None:
    """
    Visualizes the predicted values for new data.

    This function creates a line plot of the predicted values to help
    assess the behavior and trend of predictions for new, unseen data.

    Arguments:
        y_pred (array-like): The predicted values from the model.

    Returns:
        None. Displays the plot of predictions.

    Example:
        visualise_prediction(predictions)
    """
    log("Visualizing predictions.")
    plt.plot(y_pred, label="Predictions", color="orange")
    plt.title("Predictions for New Data")
    plt.xlabel("Index")
    plt.ylabel("Prediction Value")
    plt.legend()
    plt.show()


@exception_handler()
def visualise_model_performance(
    y_test: Union[list, np.ndarray],
    y_pred: Union[list, np.ndarray],
    result_marker: str,
    regression: bool,
    classification: bool,
) -> None:
    """
    Visualizes the training results, including distributions, predictions vs actual,
    residuals, and confusion matrix based on the type of model (regression or classification).

    The function creates several plots:
        1. Distribution of predicted values (for regression or classification).
        2. Predictions vs Actual values (for regression).
        3. Residuals (errors) distribution (for regression).
        4. Confusion matrix (for classification).

    Arguments:
        y_test (array-like): The true values for the test data.
        y_pred (array-like): The predicted values from the model.
        result_marker (str): The name of the target variable.
        regression (bool): Whether the model is a regression model.
        classification (bool): Whether the model is a classification model.

    Returns:
        None. Displays multiple plots based on the model type.

    Example:
        visualise_training_results(y_test, y_pred, 'Price', True, False)
    """
    log("Visualizing target distribution.")
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=50, alpha=0.7, color="blue")
    plt.title(f"Distribution of Target Variable: {result_marker}")
    plt.xlabel(result_marker)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    log("Visualizing model performance.")

    if regression:
        plot_regression_performance(y_test, y_pred)

    elif classification:
        plot_classification_performance(y_test, y_pred)

    log("Visualizations completed.")


@exception_handler()
def plot_regression_performance(
    y_test: Union[list, np.ndarray], y_pred: Union[list, np.ndarray]
) -> None:
    """
    Visualizes regression model performance by plotting predicted vs actual values
    and the distribution of residuals (errors).

    This function creates two plots:
        1. A scatter plot of predicted vs actual values.
        2. A histogram of residuals (errors) to show the error distribution.

    Arguments:
        y_test (array-like): The true values for the test data.
        y_pred (array-like): The predicted values from the regression model.

    Returns:
        None. Displays the plots for regression model performance.

    Example:
        plot_regression(y_test, y_pred)
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color="green")
    plt.plot(
        [min(y_test), max(y_test)],
        [min(y_test), max(y_test)],
        color="red",
        linestyle="--",
    )
    plt.title("Predictions vs Actual (Regression)")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.grid(True)
    plt.show()

    errors = y_pred - y_test
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color="orange")
    plt.title("Residuals (Errors) Distribution")
    plt.xlabel("Error")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()


@exception_handler()
def plot_classification_performance(
    y_test: Union[list, np.ndarray], y_pred: Union[list, np.ndarray]
) -> None:
    """
    Visualizes classification model performance by displaying a confusion matrix.

    This function generates a heatmap of the confusion matrix to visualize
    the performance of the classification model in terms of true positives,
    false positives, true negatives, and false negatives.

    Arguments:
        y_test (array-like): The true class labels for the test data.
        y_pred (array-like): The predicted class labels from the model.

    Returns:
        None. Displays the confusion matrix plot.

    Example:
        plot_classification(y_test, y_pred)
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
