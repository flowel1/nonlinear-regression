# -*- coding: utf-8 -*-

import collections
import numpy as np
import sys

from enum import Enum

# y = f(X, theta) + eps

class PDistr(Enum):
    GAUSSIAN  = 1
    LOGNORMAL = 2
    
class Prior:
    
    def __init__(self, distrib, mean = 0., precision = 1.):
        
        self.distrib   = distrib
        self.mean      = mean
        self.precision = precision
        
        
LMStepOutput = collections.namedtuple('LMStepOutput',
                                      ['yEst',
                                       'err',
                                       'WSS',
                                       'sigma2',
                                       'Objective'])

# f: Rk ---> RN ==> Jf: RN ---> Rk

class LevenbergMarquardtReg: # frozen parameters
    
    def __init__(self, model_fn, lbda = .1, step_init = 1., min_displacement = 1E-3,
                 max_lbda = 1., step_mult_down = 0.8,
                 lbda_mult_up = 2., lbda_mult_down = 1.5,
                 check_every = 10, min_norm = 1E-5, max_iter = None):
        
        self.model_fn  = model_fn
        self.lbda      = lbda
        self.step_init = step_init
        self.min_displacement = min_displacement
        self.max_lbda  = max_lbda
        self.step_mult_down = step_mult_down
        self.lbda_mult_up   = lbda_mult_up
        self.lbda_mult_down = lbda_mult_down
        self.check_every = check_every
        self.min_norm    = min_norm
        self.max_iter    = max_iter
        
        self.current_status = None
        
    def fit(self, X, y, theta_init, bounds = None, priors = None, weights = None):
        
        self.nObs, self.nParams = X.shape
        
        self.X, self.y = X, y
        self.weights = np.ones(self.nObs) if weights is None else weights # not necessarily normalized
        self.lower, self.upper = self.__set_bounds__(bounds)
        self.priors = priors
                
        self.theta  = theta_init.copy()
        
        self.total_displacement = 0.
        self.step = self.step_init
        
        self.current_status = self.__get_current_status__(theta_init)
        print("Initial WSS: {}".format(self.current_status.WSS))
        
        nIter = 0
        while True:
            descent_direction = self.__find_descent_direction__()
            flg_theta_updated = self.__move_to_new_theta__(descent_direction)
            nIter += 1
            if nIter % self.check_every == 0:
                norm_theta = np.linalg.norm(self.theta)
                if norm_theta == 0.:
                    raise Exception("Theta was set to 0")
                perc_displacement = (self.total_displacement / self.check_every) / norm_theta
                print("Check after {nIter} iterations: % displacement = {perc_displacement}, norm_theta = {norm_theta}" \
                      .format(nIter = nIter, perc_displacement = perc_displacement, norm_theta = norm_theta))
                self.step = perc_displacement # FIXME da rivedere
                
                if perc_displacement < self.min_norm:
                    break
                self.total_displacement = 0.

            if nIter == self.max_iter:
                break
            
    def predict(self, X):
        return self.model_fn(X, self.theta)

    def __find_descent_direction__(self):
        
        # Calculate descent direction from current theta
        J  = self.Jf_theta(self.X, self.theta)
        JT = np.transpose(J)
        self.JTJ = np.dot(JT, np.dot(np.diag(self.weights), J))
        self.constant_term = np.dot(JT, self.current_status.err) # = - gradient of the objective function
        if self.priors is not None:
            self.__add_priors__()
            
        self.JTJ += self.lbda * np.diag(np.diag(self.JTJ)) # Marquardt
            
        success = False
        for n_trial in [0, 1, 2]:
            if np.linalg.cond(self.JTJ) < 1. / sys.float_info.epsilon:
                descent_direction = np.linalg.solve(self.JTJ, self.constant_term)
                success = True
                break
            self.__improve_JTJ_cond__(n_trial = n_trial)
            
        if not success:
            raise Exception("Could not invert JTJ matrix")
            
        if np.dot(self.constant_term, descent_direction) < 0.:
            raise Exception("Direction found is not a descent direction")
            
        return descent_direction
    
    def __add_priors__(self):
        
        self.JTJ /= self.current_status.sigma2
        for i in range(len(self.priors)):
            pr = self.priors[i]
            if pr.distrib == PDistr.GAUSSIAN:
                self.JTJ[i][i]        += pr.precision
                self.constant_term[i] -= pr.precision * (self.theta[i] - pr.mean)
            if pr.distrib == PDistr.LOGNORMAL:
                log_theta_over_mu = np.log(self.theta[i] / pr.mean)
                self.JTJ[i][i]        += - (1. + (log_theta_over_mu - 1.) * pr.precision) / np.pow(self.theta[i], 2.)
                self.constant_term[i] -= pr.precision * (log_theta_over_mu + 1. / pr.precision) / self.theta[i]

    def __move_to_new_theta__(self, descent_direction):
            
        norm_desc_dir = np.linalg.norm(descent_direction)
        
        self.status = self.__get_current_status__(self.theta)

        flg_step_decreased = False
        flg_theta_updated  = False
        while True:
            theta_new = self.theta + self.step * descent_direction
            if self.lower is not None:
                theta_new = np.clip(theta_new, self.lower, self.upper)
                
            new_status = self.__get_current_status__(theta_new)
            
            if self.current_status.Objective - new_status.Objective > 1E-5 * self.current_status.Objective:
                self.current_status = new_status
                self.theta = theta_new
                self.total_displacement += self.step * norm_desc_dir
                flg_theta_updated = True
            else:
                self.step *= self.step_mult_down
                flg_step_decreased = True
            
            if self.step * norm_desc_dir < self.min_displacement:
                break
        
        if flg_step_decreased: # update lambda
            self.lbda = min(self.max_lbda, self.lbda * self.lbda_mult_up)
            
        return flg_theta_updated
                
    def __get_current_status__(self, theta):
        
        yEst = self.model_fn(self.X, theta)
        err  = self.y - yEst
        WSS  = sum(self.weights * np.power(err, 2.))
        sigma2 = WSS # FIXME rivedere, va divisa per nObs per ottenere varianza stimata
        
        Objective = WSS
        if self.priors is not None:
            Objective = 0.5 * self.nObs * np.log(sigma2) + 0.5 * WSS / sigma2
            for i in range(len(self.priors)):
                pr = self.priors[i]
                mu, bta = pr.mean, pr.precision
                if pr.distrib == PDistr.GAUSSIAN:
                    Objective += 0.5 * bta * np.power(theta[i] - mu, 2.)
                elif pr.distrib == PDistr.LOGNORMAL:
                    Objective += np.log(theta[i]) + 0.5 * bta * np.power(theta[i] / mu, 2.)
        
        return LMStepOutput(yEst      = yEst,
                            err       = err,
                            WSS       = WSS,
                            sigma2    = WSS,
                            Objective = Objective)
        
    def Jf_theta(self, X, theta, h = 1E-5):
        k = len(theta)
        
        Jf = []
        for i in range(len(theta)):
            Jf.append((self.model_fn(X, theta + h * np.eye(1, k, i)[0]) - self.model_fn(X, theta)) / h)
            
        return np.transpose(np.array(Jf))

    
    def __improve_JTJ_cond__(self, n_trial = 1):
        
        if n_trial == 1:
            # se JTJ non è invertibile perché ha una o più righe nulle, si mette un 1
            # sulla diagonale per ogni riga nulla (ci si avvicina alla direzione del gradiente)
            
            zero_rows = np.where(np.max(np.abs(self.JTJ), axis = 1) < 1E-5)[0]
            for i in zero_rows:
                self.JTJ[i][i] = 1.
                
        elif n_trial == 2:
            # mette 1 sulla diagonale della matrice, usata come estrema ratio
            # se matrice non è invertibile
            for i in range(len(self.JTJ)):
                self.JTJ[i][i] = 1.
                
    def __set_bounds__(self, bounds):
        
        if bounds is None:
            lower = None
            upper = None
        else:
            lower, upper = [np.array(x) for x in zip(*bounds)]
            lower[lower == None] = -1E+30
            upper[upper == None] = +1E+30

        return lower, upper
    

        
    
            
'''
df = pd.read_csv("data.csv")

y = df['y'].values
X = df[['x1', 'x2', 'x3']].values

theta = np.array([5., 4., 6., 0.7, 46.12])

def f(X, theta):
    
    X_transf = np.concatenate((np.ones((X.shape[0], 1)), X[:, :2]), axis = 1)
    X_transf[:, 2] = X_transf[:, 2] ** 2.
    
    atan_arg = np.dot(X_transf, theta[:3]) + X[:, 2]
    
    return theta[4] * np.power(1. - 2. / np.pi * np.arctan(atan_arg), theta[3])
        
theta_init = np.array([4.5,  2.3,  5.1,  1., 60.])
theta_opt  = np.array([ 5.,  4.,  6.,  0.7, 46.12])
#priors = [Prior(PDistr.GAUSSIAN)] * len(theta_init)
'''









