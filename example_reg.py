# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from levenberg_marquardt import PDistr, Prior
from levenberg_marquardt import LevenbergMarquardtReg

if __name__ == '__main__':
    df = pd.read_csv("sample-data\\nonlinear-data.csv")
    df = df.sort_values(by = 'x').reset_index(drop = True)
    X, y = df[['x']].values, df['y'].values
    sorted_ixs = np.argsort(X[:, 0])
    theta_actual = np.array([15., .1, .4]) # <--- value that was used to generate the data
    
    # Define nonlinear model and declare LevenbergMarquardtReg class
    def f(X, theta):
        return theta[0] * np.tanh(theta[1] + theta[2] * X[:, 0])
    lr = LevenbergMarquardtReg(model_fn = f)    
    
    priors = [Prior(PDistr.GAUSSIAN, mean = 15., precision = 0.01),
              Prior(PDistr.GAUSSIAN, mean = 1. , precision = 0.01),
              Prior(PDistr.GAUSSIAN, mean = 1. , precision = 0.01)]
    
    theta_init = [15., 1., 1.]
    lr.fit(X[:10], y[:10], theta_init = theta_init, priors = priors)
    yEst_reg  = lr.predict(X)[sorted_ixs]
    
    lr.fit(X[:10], y[:10], theta_init = theta_init)
    yEst_nreg = lr.predict(X)[sorted_ixs]
    
    # Display results
    expected_values = lr.__get_current_status__(theta_actual)
    print("\n*** RESULTS")
    print("Estimated theta: {}".format(lr.theta))
    print("WSS for estimated theta: {}".format(lr.current_status.WSS))
    print("WSS for real value of theta: {}".format(expected_values.WSS))
    
    plt.figure()
    
    plt.scatter(X[:, 0], y, color = 'black', marker = '.')
    plt.scatter(X[:10, 0], y[:10], color = 'red', marker = 'o', label = 'available observations')
    plt.plot(X[sorted_ixs, 0], yEst_reg, color = 'red', label = 'regularized model')
    plt.plot(X[sorted_ixs, 0], yEst_nreg, color = 'green', label = 'non-regularized model')
    plt.legend()
    plt.title("Levenberg-Marquardt algorithm")