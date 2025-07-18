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
    plt.plot(X_plot_poly, Y_plot_poly, color = 'r', label = f'polynomial regression with degree {degree}')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'polynomial regression with degree {degree}')
    plt.grid(True)
    plt.xlim(np.min(X), np.max(X))
    plt.ylim(np.min(Y), np.max(Y))

    plt.show()

    return model.coef_, model.intercept_ # return the wieghts and the free member
    
