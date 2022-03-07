import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from h5_w7_helper_functions import linear_decision_function, nonlinear_decision_function, plot_perceptron_boundary


class Datapreprocessor:
    def __init__(self, path):
        """
        :param path: path to the '.csv' files
        """
        self.path = path

    def read_data(self):
        """
        :param: path
        :return: data
        """
        data = np.array(pd.read_csv(self.path, header=None))
        # feature_labels = data[0, :]
        data = np.delete(data, 0, 0)  # removing from the feature_labels
        return data

    def data_preprocess(self, data, augmentation):
        """
        :param: dataset loaded from above
        :return: augmented data points and labels that are shuffled
        """
        # np.random.shuffle(data)  # shuffling the data set
        D = data.shape[1] - 1
        x = np.array(data[:, 0:D], dtype=float)
        y = np.array(data[:, -1], dtype=float)

        # data augmentation
        if augmentation:
            x_augmented = np.zeros([x.shape[0], x.shape[1] + 1])
            x_augmented[:, 0] = np.ones(x.shape[0])
            x_augmented[:, 1:x.shape[1] + 1] = x

        else:
            x_augmented = x

        return x_augmented, y

    def feature_expansion(self, x):
        """
        :param x: data points
        :return: non-augmented data points in expanded feature space
        """
        D1 = 5
        x_expanded = np.zeros([x.shape[0], 5])
        x_expanded[:, 0:2] = x
        x_expanded[:, 2] = np.multiply(x[:, 0], x[:, 1])
        x_expanded[:, 3] = np.power(x[:, 0], 2)
        x_expanded[:, 4] = np.power(x[:, 1], 2)

        return x_expanded

    def scatter_plot(self, x, y):
        """
        :param x: data points
        :param y: labels
        :return: scatter point in non-augmented feature space
        """

        plt.figure()
        plt.scatter(x[np.where(y == 1), 0], x[np.where(y == 1), 1], color='red', label='class1')
        plt.scatter(x[np.where(y == 2), 0], x[np.where(y == 2), 1], color='blue', label='class2')
        plt.legend(loc='upper center')
        plt.xlabel("$x_{1}$")
        plt.ylabel("$x_{2}$")


# Data Loading and Pre processing
train_path = "h5w7_data.csv"
Data_loader = Datapreprocessor(train_path)
train_data = Data_loader.read_data()
x_train, y_train = Data_loader.data_preprocess(train_data, augmentation=False)

# ------------- SCATTER PLOT OF ORIGINAL DATASET -------------- #
Data_loader.scatter_plot(x_train, y_train)
plt.title("Scattered data points")
plt.savefig("figures/Problem3(a).png")

# -------------- PERCEPTRON ALGORITHM --------------- #
perceptron = Perceptron(fit_intercept=False)
perceptron.fit(x_train, y_train)
pred_labels = perceptron.predict(x_train)
weights = perceptron.coef_[0]  # perceptron weights
accuracy = perceptron.score(x_train, y_train)
print(f'classification accuracy in original feature space: {accuracy}')
Data_loader.scatter_plot(x_train, pred_labels)
plt.title("Perceptron Trained data points")
plt.savefig("figures/Problem3(c).png")

fig = plot_perceptron_boundary(x_train, y_train, weights, linear_decision_function)
plt.title("Linear Decision Boundary from Perceptron training")
plt.savefig("figures/Problem3(d).png")

# --------- EXPANDED FEATURE SPACE - PERCEPTRON ALGORITHM -------- #

phi_x_train = Data_loader.feature_expansion(x_train)
perceptron.fit(phi_x_train, y_train)
pred_labels_expanded = perceptron.predict(phi_x_train)
weights_expanded = perceptron.coef_[0]
class_accuracy = perceptron.score(phi_x_train, y_train)
print(f'classification accuracy in expanded feature space: {class_accuracy}')
print(f'Weights for expanded feature space: ', weights_expanded)

# ---------------- BEST FEATURES --------------- #
best_feature_indices = abs(weights_expanded).argsort()[3:5]
print(f'Indices corresponding to the best two features: ', best_feature_indices)
weights_best_2 = abs(weights_expanded[best_feature_indices])
print(f'The two best features: {weights_best_2}')
phi_x_best_2 = np.zeros([phi_x_train.shape[0], 2])
for i in range(2):
    phi_x_best_2[:, i] = phi_x_train[:, best_feature_indices[i]]

# plot the best two features and the decision boundary
fig1 = plot_perceptron_boundary(phi_x_best_2, y_train, weights_best_2, linear_decision_function)
plt.title(f'Linear Decision Boundary of best features, $x_{1}x_{2}$, $x_{2}^{2}$')
plt.savefig(f'figures/Problem3(e).png')

# ------------- NON LINEAR DECISION BOUNDARY ----------------- #
plot_perceptron_boundary(x_train, y_train, weights_expanded, nonlinear_decision_function)
plt.title(f'Non-linear Decision Boundary')
plt.savefig(f'figures/Problem3(f).png')
plt.show()
