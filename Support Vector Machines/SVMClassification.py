import numpy as np
import matplotlib.pyplot as plt


def calculate_rho(u1, u2, u3, z1, z2, z3):
    """
    calculating rho ([lambda1, lambda2, lambda3, mu]) using inverse method
    """
    A = np.array([[z1**2*np.dot(u1, np.transpose(u1)), z1*z2*np.dot(u1, np.transpose(u2)), z1*z3*np.dot(u1, np.transpose(u3)), -z1],
                  [z2*z1*np.dot(u2, np.transpose(u1)), z2**2*np.dot(u2, np.transpose(u2)), z2*z3*np.dot(u2, np.transpose(u3)), -z2],
                  [z3*z1*np.dot(u3, np.transpose(u1)), z3*z2*np.dot(u3, np.transpose(u2)), z3**2*np.dot(u3, np.transpose(u3)), -z3],
                  [z1, z2, z3, 0]])

    b = np.transpose([1, 1, 1, 0])

    A_inv = np.linalg.inv(A)
    rho = np.matmul(A_inv, b)

    return rho, A, b


def check_KKT(z, rho, weight, w0, u):
    """
    checking the four KKt conditions
    """
    count = 0
    # check all lambdas are non-negative
    for i in range(u.shape[0]):
        if rho[i] >= 0:
            count += 1
            if count == 3:
                print(f'KKT1: lambdas are non-negative')
        else:
            print(f'KKT1: lambdas are negative')

    # Second KKT condition - sum(lambda*z) = 0
    condition2 = int(np.sum(rho[0:3]*z))
    if condition2 == 0:
        print('KKT2: Sum of lambda & z is 0. New KKT condition satisfied')
    else:
        print('KKT2: Sum of lambda & z is not 0. New KKT condition is not satisfied')

    # check if weights and bias satisfy the last 2 KKT conditions
    for i in range(u.shape[0]):
        if z[i]*(np.dot(weight, u[i, :]) + w0) - 1 >= 0:
            count += 0
    if count > 1:
        print(f'The weights and bias follow the 2 other KKT conditions')

    return rho


def calculate_optimal_weights(rho, z, u):
    """
    calculate the weight vector
    """
    weight = rho[0:3]*z
    weight = np.matmul(np.transpose(u), weight)
    return weight


def calculate_bias(weight, u, z, rho):
    """
    Calculating the bias term
    """
    w0 = np.zeros(u.shape[0])
    for i in range(u.shape[0]):
        if rho[i] != 0:
            w0[i] = (1 - z[i]*np.dot(weight, u[i, :]))/z[i]
    bias = w0[0]
    return round(bias, 4)


def recalculating_rho(rho, A, b):
    """
    recalculating rho, when any lambda is negative
    """
    for i in range(len(rho)):
        if rho[i] < 0:
            A = np.delete(A, i, axis=0)
            A = np.delete(A, i, axis=1)
            b = np.delete(b, i)
            index = i

    A_inv = np.linalg.inv(A)
    rho_new = np.matmul(A_inv, b)
    rho_new = np.insert(rho_new, index, 0)
    return rho_new


def dataset_evaluation(u1, u2, u3, z1, z2, z3):
    """
    This function evaluates each dataset -
    1. Inverts the A matrix to calculate rho
    2. Check if lambda vector satisfies the KKT conditions
        involving lambda
    3. Calculate optimal weight vector and optimal bias value
        one of the KKT conditions
    4. Check if weight vector an bias satisfy the remaining
        KKT conditions
    5. If any lambda fails to satisfy (2), then re-calculate A
        and rho and re-evaluate the KKt conditions
    6. Finally plot the decision boundary as suggested
    """
    u = np.array([u1, u2, u3])
    z = np.array([z1, z2, z3])
    rho_data, A, b = calculate_rho(u1, u2, u3, z1, z2, z3)
    print(f'rho vector: {rho_data}')
    for i in rho_data[0:3]:
        if i < 0:
            print(f'One of the lambdas is negative')
            print(f're-evaluating KKT conditions')
            rho_data = recalculating_rho(rho_data, A, b)
    optimal_weights = calculate_optimal_weights(rho_data, z, u)
    print(f'optimal weights: {optimal_weights}')
    optimal_bias = calculate_bias(optimal_weights, u, z, rho_data)
    print(f'optimal bias: {optimal_bias}')

    check_KKT(z, rho_data, optimal_weights, optimal_bias, u)

    plot_decision_boundary(u, optimal_weights, optimal_bias)

    print('\n')


def plot_decision_boundary(u, weight, bias):
    """
    plots the decision regions and the boundary along with the
    margins. The points are all reflected here.
    """
    x_line = np.arange(-10, 10)
    arrow_index = int(len(x_line)/2 + 1)
    x2 = np.zeros(x_line.shape)
    x3 = np.zeros(x_line.shape)
    for i in range(len(x_line)):
        x2[i] = (-weight[0]*x_line[i] - bias + 1)/(weight[1])
        x3[i] = (-weight[0]*x_line[i] - bias - 1)/(weight[1])
    plt.scatter(u[0:2, 0], u[0:2, 1], c='blue', marker='o')
    plt.scatter(u[2, 0], u[2, 1], c='red', marker='x')
    plt.legend(['S1', 'S2'], loc="lower left")
    plt.plot(x_line, x2, 'b:')
    plt.plot(x_line, x3, ':')
    plt.arrow(x_line[arrow_index], x2[arrow_index], 0.5, 0.5, width=0.01, head_width=0.3)
    plt.arrow(x_line[arrow_index], x3[arrow_index], -0.5, -0.5, width=0.01, head_width=0.3)
    # plotDecBoundaries(_,_,_)
    max_x = 5
    min_x = -5
    max_y = +5
    min_y = -5
    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005

    # generate grid coordinates. this will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1] + inc / 100, inc), np.arange(yrange[0], yrange[1] + inc / 100, inc))

    # size of the (x, y) image, which will also be the size of the
    # decision boundary image that is used as the plot background.
    image_size = x.shape
    xy = np.hstack((x.reshape(x.shape[0] * x.shape[1], 1, order='F'), y.reshape(y.shape[0] * y.shape[1], 1, order='F')))

    func = weight[1] * xy[:, 1] + weight[0] * xy[:, 0] + bias > 0

    pred_label = func[:]

    # reshape the idx (which contains the class label) into an image.
    decisionmap = pred_label.reshape(image_size, order='F')

    plt.imshow(decisionmap, extent=[xrange[0], xrange[1], yrange[0], yrange[1]], origin='lower', alpha=0.9, cmap='Set3')

    plt.title('Decision Boundary in Non-augmented Feature Space')


if __name__ == "__main__":
    print(f'****** For dataset 1 ******')
    u11 = np.array([1, 2])
    u21 = np.array([2, 1])
    u31 = np.array([0, 0])
    z1 = 1
    z2 = 1
    z3 = -1
    dataset_evaluation(u11, u21, u31, z1, z2, z3)
    plt.savefig('figures/dataset1.png')
    plt.show()

    print(f'****** For dataset 2 ******')
    u32 = np.array([1, 1])
    dataset_evaluation(u11, u21, u32, z1, z2, z3)
    plt.savefig('figures/dataset2.png')
    plt.show()

    print(f'****** For dataset 3 ******')
    u33 = np.array([0, 1.5])
    dataset_evaluation(u11, u21, u33, z1, z2, z3)
    plt.savefig('figures/dataset3.png')
    plt.show()