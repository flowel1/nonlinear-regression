# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from levenberg_marquardt import LevenbergMarquardtReg

if __name__ == '__main__':

    # Load training data
    df = pd.read_csv("sample-data\\nonlinear-data.csv")
    X, y = df[['x']].values, df['y'].values
    theta_actual = np.array([15., 0.1, 0.4]) # <--- values that were used to generate the data
    sigma_actual = 0.6
    
    # Define nonlinear model and declare LevenbergMarquardtReg class
    def f(X, theta):
        return theta[0] * np.tanh(theta[1] + theta[2] * X[:, 0])
    lr = LevenbergMarquardtReg(model_fn = f)    

    # Fit model
    lr.fit(X, y, theta_init = np.ones(3)) # starting point = [1., 1., 1.]
    
    # Display results
    expected_values = lr.__get_optimization_status__(theta_actual)
    print("\n*** RESULTS")
    print("Estimated theta: {}".format(lr.theta))
    print("WSS for estimated theta: {}".format(lr.current_status.WSS))
    print("WSS for real value of theta: {}".format(expected_values.WSS))
    
    plt.figure()
    sorted_ixs = np.argsort(X[:, 0])
    plt.scatter(X[:, 0], y, color = 'black', marker = '.')
    plt.plot(X[sorted_ixs, 0], lr.current_status.yEst[sorted_ixs], color = 'red'  , label = 'estimated')
    plt.plot(X[sorted_ixs, 0],   expected_values.yEst[sorted_ixs], color = 'green', label = 'target')
    plt.legend()
    plt.title("Levenberg-Marquardt algorithm", fontsize = 20)