# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from levenberg_marquardt import Prior
from levenberg_marquardt import LevenbergMarquardtReg

if __name__ == '__main__':
    
    # Load training data
    df = pd.read_csv("sample-data\\nonlinear-data.csv")
    df = df.sort_values(by = 'x').reset_index(drop = True)
    X, y = df[['x']].values, df['y'].values
    
    train_data_ixs = range(10 + 1)
    X_train, y_train = X[train_data_ixs], y[train_data_ixs]
    sorted_ixs = np.argsort(X[:, 0])
    theta_actual = np.array([15., .1, .4]) # <--- value that was used to generate the data
    
    # Define nonlinear model and declare LevenbergMarquardtReg class
    def f(X, theta):
        return theta[0] * np.tanh(theta[1] + theta[2] * X[:, 0])
    lr    = LevenbergMarquardtReg(model_fn = f)
    lrReg = LevenbergMarquardtReg(model_fn = f)
    
    priors = [Prior(Prior.LOGNORMAL, mu = 20., bta = 10.), #Prior(Prior.GAUSSIAN, mu = 15., bta = 0.01),
              Prior(Prior.GAUSSIAN , mu = 1. , bta = 0.1),
              Prior(Prior.GAUSSIAN , mu = 1. , bta = 0.1)]
    
    # plot priors to get a qualitative idea
    fig, axs = plt.subplots(1, len(priors))
    for i in range(len(axs)):
        if priors[i].density == Prior.GAUSSIAN:
            sigma = 1. / np.sqrt(priors[i].bta)
            xx = np.linspace(priors[i].mu - 2. * sigma, priors[i].mu + 2 * sigma, 100)
        elif priors[i].density == Prior.LOGNORMAL:
            sigma = np.sqrt(1. / priors[i].bta)
            hlim = priors[i].mu * np.exp(sigma * (np.log(priors[i].mu) + 2 * sigma))
            xx = np.linspace(1E-5, hlim, 100.)
        axs[i].plot(xx, priors[i].pdf(xx))
        axs[i].set_title("parameter #{}".format(i), fontsize = 20)
    plt.suptitle("Prior distributions", fontsize = 25)
        
    theta_init = [np.mean(y_train), 0., 1.]
    print("theta_init = {}".format(theta_init))
    
    lr   .fit(X_train, y_train, theta_init = theta_init)
    lrReg.fit(X_train, y_train, theta_init = theta_init, priors = priors)
    yEst_reg  = lr.predict(X)[sorted_ixs]
    
    # Display results
    expected_values = lr.__get_optimization_status__(theta_actual)
    print("\n*** RESULTS")
    print("Estimated theta:\n\t{} (reg)\n\t{} (non reg)".format(lrReg.theta, lr.theta))
    print("WSS for estimated theta:\n\t{} (reg)\n\t{} (non reg)".format(lrReg.current_status.WSS,
                                                                   lr   .current_status.WSS))
    print("WSS for real value of theta: {}".format(expected_values.WSS))
    
    plt.figure()
    plt.scatter(X[:, 0], y, color = 'black', marker = '.')
    plt.scatter(X_train[:, 0], y_train, color = 'red', marker = 'o', label = 'available observations')
    plt.plot(X[sorted_ixs, 0], lrReg.predict(X)[sorted_ixs], color = 'red'  , label = 'regularized model')
    plt.plot(X[sorted_ixs, 0], lr   .predict(X)[sorted_ixs], color = 'green', label = 'non-regularized model')
    plt.legend()
    plt.title("Levenberg-Marquardt algorithm", fontsize = 20)