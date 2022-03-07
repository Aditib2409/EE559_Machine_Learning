import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(42)

SAVE_PLOTS = False
SHOW_PLOTS = True

class Datapreprocessor:
    def __init__(self, path):
        """
        :param path: path to the '.csv' files
        """
        self.path = path

    def pre_process(self):
        """
        :return: x_augmented, y
        """
        data = np.array(pd.read_csv(self.path, header=None))
        # feature_labels = data[0, :]
        data = np.delete(data, 0, 0)  # removing from the feature_labels

        np.random.shuffle(data)  # shuffling the data set

        x = np.array(data[:, 0:4], dtype=float)
        y = np.array(data[:, -1], dtype=float)

        # data augmentation
        x_augmented = np.zeros([x.shape[0], x.shape[1] + 1])
        x_augmented[:, 0] = np.ones(x.shape[0])
        x_augmented[:, 1:x.shape[1] + 1] = x

        return x_augmented, y


class Regression:
    def __init__(self, a, b):
        """
        :param a: learning_rate parameter1
        :param b: learning_rate parameter2
        """
        self.a = a
        self.b = b
        self.epochs = 100
        self.len_A = 5
        self.len_B = 4

    def fit(self, x, y):

        N, D = x.shape
        m = np.arange(0, self.epochs, 1)
        E_rms = np.zeros([self.epochs, ])
        weight = np.zeros([self.epochs, D])

        itr = 0

        w = np.random.uniform(-0.1, +0.1, D)

        for epoch in m:

            J = (np.sum(np.power(np.dot(x, w) - y, 2))) / N

            E_rms[epoch] = np.sqrt(J)

            # the following updates are for 1 epoch
            for n in range(N):
                learning_rate = self.a / (self.b + itr)
                w = w - learning_rate * (np.dot(x[n, :], w) - y[n]) * x[n, :]
                itr += 1

            weight[epoch, :] = w

            # halting condition 1 -
            if E_rms[epoch] < 0.001 * E_rms[0]:
                print("Halting condition 1 satisfied. %d epochs done" % (epoch + 1))
                print("Learning rate parameters: A = %d; B = %d" % (a, b))
                break

        final_weight = weight[-1, :]

        return final_weight, E_rms

    def predict(self, x, w):
        """
        :param x: Augmented test dataset
        :param w: Optimized weight vector
        :return: predictions
        """
        y = np.matmul(x, w)
        return y

    def compute_rms(self, predicted, actual):
        """
        :param mean: y_mean
        :return:
        """
        return np.sqrt((np.sum(np.power(predicted - actual, 2))) / len(actual))


    def plot_learning_curves(self, Error, lr1, lr2):
        """
        :param Error: E_rms
        :param lr1: learning_rate parameter 1
        :param lr2: learning_rate parameter 2
        :return: learning curves subplots
        """
        plt.gca().plot(np.arange(1, self.epochs, 1), Error, label="B = %d" % lr2)
        plt.gca().set_title("A = %.2f" % lr1)
        plt.gca().set_xlabel("epochs")

        plt.legend()


# get the training dataset
train_path = "h5w7_pr1_power_train.csv"
Data_loader = Datapreprocessor(train_path)
x_train_augmented, y_train = Data_loader.pre_process()

# get the test dataset
test_path = "h5w7_pr1_power_test.csv"
Data_loader_test = Datapreprocessor(test_path)
x_test_augmented, y_test = Data_loader_test.pre_process()

# Question 1(b)
A = [0.01, 0.1, 1, 10, 100]
B = [1, 10, 100, 1000]
epochs = 100

# RMS error for all learning rates
rms_lr = np.zeros([len(A), len(B)])

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(13, 5))
fig.supylabel("$E_{rms}^{epoch}$")

for i in range(len(A)):
    for j in range(len(B)):
        model = Regression(A[i], B[j])
        optimized_weight, E_RMS = model.fit(x_train_augmented, y_train)
        rms_lr[i, j] = E_RMS[-1]  # rms for each A, B combinations

        plt.sca(axes[i])
        model.plot_learning_curves(E_RMS[1:], A[i], B[j])  # plot the learning curves

# if the halting condition 2 is satisfied
if rms_lr[4, 3] != 0:
    print("Halting condition 2 satisfied. %d epochs have been completed!" % epochs)

# Question 1(d)
# Best learning rates
best_learning_rate = np.unravel_index(rms_lr.argmin(), rms_lr.shape) # gives an ordered pair
print("Best Learning rate parameters: A = %.2f, B = %.2f" % (A[best_learning_rate[0]], B[best_learning_rate[1]]))

# defining model for the best learning rate parameters
best_model = Regression(A[best_learning_rate[0]], B[best_learning_rate[1]])

# Best weights for the above learning parameters
best_weight, _ = best_model.fit(x_train_augmented, y_train)
print("Best Weights corresponding to the best learning rates: ", best_weight)

# Predict on the training set
y_predict_train = best_model.predict(x_train_augmented, best_weight)
train_error = best_model.compute_rms(y_predict_train, y_train)
print("Train classification: ", train_error)

# Predict on the test dataset
y_predict_test = best_model.predict(x_test_augmented, best_weight)
test_error = best_model.compute_rms(y_predict_test, y_test) # test E_rms from regressor

# Question 1(e)
y_mean = np.mean(y_train)
trivial_error = best_model.compute_rms(y_mean, y_test) # E-rms on trivial case
print(f'Test Error from regressor: {test_error}% | Trivial test error: {trivial_error}%')

if SAVE_PLOTS:
    plt.savefig("figures/Problem1(b).png")

if SHOW_PLOTS:
    plt.show()