import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data(path):
    data = np.array(pd.read_csv(path))
    return data
                   
def get_data(data, num_features):
    input_data = data[:,0:num_features]
    labels = data[:,-1]
                    
    return input_data, labels

def shuffle_data(X):
    return np.random.shuffle(X)

# get the data
train_data = load_data("synthetic1_train.csv")
test_data = load_data("synthetic1_test.csv")

D = 2 # NUMBER OF FEATURES
N = train_data.shape[0] # NUMBER OF DATA POINTS

X_train, y_train = get_data(train_data, D) # split the train dataset
X_test, y_test = get_data(test_data, D) # split the test dataset


XZ_train_augmented = np.ones([N, D+1])
XZ_train_augmented[np.where(y_train == 2),0] = -1 
XZ_train_augmented[:,1:D+1] = X_train
XZ_train_augmented[np.where(y_train == 2),1] = -1*(X_train[np.where(y_train == 2),0])
XZ_train_augmented[np.where(y_train == 2),2] = -1*(X_train[np.where(y_train == 2),1])
XZ_train_augmented = XZ_train_augmented[48:52,:]

class PERCEPTRON:
    def __init__(self, lr):
        self.lr = lr
        #self.w = np.zeros([3,1])
       
    # DISCRIMINANT FUNCTION
    def discriminant_function(self, x, w):
        g = np.matmul(x,w) # scalar for one data point
        #print(g)
        return g

    # INDICATOR FUNCTION
    def indicator_function(self,g):
        h = 0
        for i in range(len(g)):
            if g[i] > 0:
                h = 1 # (1,1) vector of 0 or 1
        return h
        
       
    def perceptron_algorithm(self, X, w, h, itr, num_data):
        for n in range(num_data):
            i = (itr*num_data) + n
            if g <= 0:
                w[i+1,:] = w[i,:] + (self.lr*h)
            else:
                w[i+1,:] = w[i,:]
            #return w  



model = PERCEPTRON(lr=1)
lr = 1
num_iterations = 10000
a = 0.1
weight = a*np.ones((D+1,))
FINAL_WEIGHTS = np.zeros([num_iterations+1,D+1])
FINAL_WEIGHTS[0,:] = weight
FINAL_J = np.zeros([500])

i = 0
# --------- BASIC SEQUENTIAL GRADIENT DESCENT --------- #
while i < num_iterations - 1:
    H = np.zeros([4])
    for j in range(4):
        
        g = np.inner(FINAL_WEIGHTS[i,:],XZ_train_augmented[j,:]) 
         
        if g <= 0:
            FINAL_WEIGHTS[i+1,:] = FINAL_WEIGHTS[i,:] + (lr*XZ_train_augmented[j,:])
            H[j] = 1
        else:
            FINAL_WEIGHTS[i+1,:] = FINAL_WEIGHTS[i,:]
        i += 1
    model.perceptron_algorithm(XZ_train_augmented, FINAL_WEIGHTS, h, m, N)
    J = np.dot(np.transpose( np.matmul(XZ_train_augmented, weight)),H) # scalar per iteration per (3,1) vector
    
    ### HALTING CONDITIONS
    
    if np.sum(H) == 4:
        print("Data is Linearly Separable")
        break
      
    if m >= (num_iterations - 500)-1:
        FINAL_WEIGHTS[m - (num_iterations - 500)-1,:] = weight.reshape((1,3))
        FINAL_J[m - (num_iterations - 500)-1] = J
      
