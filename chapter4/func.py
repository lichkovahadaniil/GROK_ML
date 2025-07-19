import pandas as pd
import numpy as np 
import random
import matplotlib.pyplot as plt
random.seed(0) # for reproducibility
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

'''
func trains a polinomial regression model, returns wieghts, plots rsults

sklearn always awaiting the features in the form of a column of matrix
'''

def train_plot_reg(X, Y, degree):
    # X must be the table of features [[x1], [x2], ... ]
    X = np.array(X).reshape(-1, 1) # reshape: 1  for 1 column (1 feature), -1 for automatical number of rows
    Y = np.array(Y)

    poly = PolynomialFeatures(degree = degree) # set the degree of the polynomial
    X_poly = poly.fit_transform(X) # create the polynomial features from the features

    model = LinearRegression()
    model.fit(X_poly, Y) # train the model

    # generating predicted values for plotting the graph of curve
    X_plot = np.linspace(min(X), max(X), 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot) # only 'transform' without "fit" beacause we already trained the model
    # creating the polynomial features for X_plot
    Y_plot_poly = model.predict(X_plot_poly) # predict the values of the polynomial features for X_poly

    # plotting the orginal points of data
    plt.scatter(X, Y, color = 'b', label = 'original data')

    # plot the polynomial regression curve
    plt.plot(X_plot, Y_plot_poly, color = 'r', label = f'polynomial regression with degree {degree}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'polynomial regression with degree {degree}')
    plt.legend()
    plt.grid(True)
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))

    plt.show()

    return model.coef_, model.intercept_ # return the wieghts and the free member
    

def predict_evaluate(model_coefs, degree, X_train, Y_train, X_test, Y_test):
    '''
    makes predictions using a polynomial regression model, calculates the square loss,
    plots training and testing set points, plots regression curve

    returns the mean squared error
    '''
    
    X_train = np.array(X_train).reshape(-1, 1)
    X_test = np.array(X_test).reshape(-1, 1)
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)

    coefs = model_coefs[0]
    intercept = model_coefs[1]

    poly = PolynomialFeatures(degree = degree)
    poly.fit(np.concatenate((X_train, X_test))) # default concatenation on axis = 0 ([[1], [2]], [[3]]) -> [[1], [2], [3]]
    # poly.fit calculate the polynomial features for X_train and X_test
    X_test_poly = poly.transform(X_test) # transform applies the polynomial features to X_test
    ''''
    X_test_poly @ coefs = matrix multiplication
    [[1], [x1], [x2]...    ]   [   ]    [[1] * 0, [x1] * a, [x2] * b...] 
    [[1], [x1^2], [x2^2]...] * [[a]] =  [[1] * 0, [x1^2] * a, [x2^2] * b...]
    [[1], [x1^3], [x2^3]...]   [[b]]    ...

    if here's no a coef for single column -> it will be *0
    '''
    # manual calculation, no model.predict()
    Y_pred = X_test_poly @ coefs.reshape(-1, 1) + intercept

    mse = mean_squared_error(Y_test, Y_pred)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train, Y_train, color = 'blue', label = 'training data')
    plt.scatter(X_test, Y_test, color = 'red', marker = '^', label = 'testing data')
    
    # generate points for plotting the regression curve
    full_X = np.concatenate((X_train, X_test))
    X_plot = np.linspace(np.min(full_X), np.max(full_X), 100).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)

    # Calculate predictions for the plot
    Y_plot_poly = X_plot_poly @ coefs.reshape(-1, 1) + intercept

    # plotting :D
    plt.plot(X_plot, Y_plot_poly, color = 'green', label = f'polynomial regression with degree {degree}')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f'polynomial regression with degree {degree}, mse = {mse}')
    plt.legend()
    plt.grid(True)

    # set bounds, based on the range of all data (train + test)
    full_Y = np.concatenate((Y_train, Y_test))
    plt.xlim(np.min(full_X), np.max(full_X))
    plt.ylim(np.min(full_Y) - 0.1 * (np.max(full_Y) - np.min(full_Y)), np.max(full_Y) + 0.1 * (np.max(full_Y) - np.min(full_Y)))

    plt.show()

    return mse