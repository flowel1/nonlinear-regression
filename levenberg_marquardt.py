# -*- coding: utf-8 -*-

import collections
import numpy as np
import sys

# y = f(X, theta) + eps

class Prior:
    
    GAUSSIAN  = 'GAUSSIAN'
    LOGNORMAL = 'LOGNORMAL'
    
    def __init__(self, density, mu = 0., bta = 1.):
        
        assert density in [Prior.GAUSSIAN, Prior.LOGNORMAL], \
               "Invalid density {}".format(density)
        
        self.density = density
        self.mu      = mu
        self.bta     = bta
        
    def pdf(self, x):
        
        if self.density == Prior.GAUSSIAN:
            return np.sqrt(self.bta / (2. * np.pi)) * np.exp(-0.5 * np.power(x - self.mu, 2.))

        elif self.density == Prior.LOGNORMAL:
            return np.sqrt(self.bta / (2. * np.pi)) / x * np.exp(-0.5 * np.power(np.log(x / self.mu), 2.))

        
        
LMStepOutput = collections.namedtuple('LMStepOutput',
                                      ['yEst',
                                       'err',
                                       'WSS',
                                       'sigma2',
                                       'Objective'])

# f: Rk ---> RN ==> Jf: RN ---> Rk

class LevenbergMarquardtReg: # frozen parameters
    
    def __init__(self, model_fn, lbda = .1, step_init = 1., min_displacement = 1E-5,
                 max_lbda = 1., step_mult_down = 0.8, step_mult_up = 1.2,
                 lbda_mult_up = 2., lbda_mult_down = 1.5,
                 check_every = 10, min_norm = 1E-5, max_iter = None):
        
        self.model_fn  = model_fn
        self.lbda      = lbda
        self.step_init = step_init
        self.min_displacement = min_displacement
        self.max_lbda  = max_lbda
        self.step_mult_down = step_mult_down
        self.step_mult_up   = step_mult_up 
        self.lbda_mult_up   = lbda_mult_up
        self.lbda_mult_down = lbda_mult_down
        self.check_every = check_every
        self.min_norm    = min_norm
        self.max_iter    = max_iter
        
        self.current_status = None
        
    def fit(self, X, y, theta_init, bounds = None, priors = None, weights = None):
        
        assert X.shape[0] == len(y), "Illegal input dimensions"
        
        self.nObs, self.nParams = X.shape
        
        self.X, self.y = X, y
        self.weights = np.ones(self.nObs) if weights is None else weights # not necessarily normalized
        self.lower, self.upper = self.__set_bounds__(bounds)
        self.priors = priors
                
        self.theta  = theta_init.copy()
        
        self.total_displacement = 0.
        self.step = self.step_init
        
        self.current_status = self.__get_optimization_status__(theta_init)
        print("Initial WSS: {}".format(self.current_status.WSS))
        
        nIter = 0
        while True:
            descent_direction = self.__find_descent_direction__()
            self.__move_to_new_theta__(descent_direction)
            nIter += 1
            if nIter % self.check_every == 0:
                norm_theta = np.linalg.norm(self.theta)
                if norm_theta == 0.:
                    raise Exception("Theta was set to 0")
                perc_displacement = self.total_displacement / norm_theta
                print("Check after {nIter} iterations: % displacement = {perc_displacement}, norm_theta = {norm_theta}" \
                      .format(nIter = nIter, perc_displacement = perc_displacement, norm_theta = norm_theta))
                
                if perc_displacement < self.min_norm:
                    break
                self.total_displacement = 0.

            if nIter == self.max_iter:
                break
            
    def predict(self, X):
        return self.model_fn(X, self.theta)

    def __find_descent_direction__(self):
        # descent direction solves a linear system Ax = b
        
        # Calculate descent direction from current theta
        JTWT = self.Jf_theta(self.X, self.theta)
        JTWT = np.dot(np.transpose(JTWT), np.sqrt(np.diag(self.weights)))
        A = np.dot(JTWT, np.transpose(JTWT)) # JT*WT*W*J
        b = np.dot(JTWT, self.current_status.err) # = - gradient of the objective function
        if self.priors is not None:
            A, b = self.__add_priors__(A, b)
            
        A += self.lbda * np.diag(np.diag(A)) # Marquardt
            
        success = False
        while True:
            if np.linalg.cond(A) < 1. / sys.float_info.epsilon:
                descent_direction = np.linalg.solve(A, b)
                success = True
                break
            if not self.__improve_conditioning__(A):
                break
            
        if not success:
            raise Exception("Could not calculate descent direction (singular matrix)")
            
        if np.dot(b, descent_direction) < -1E-10:
            raise Exception("Direction found is not a descent direction")
            
        return descent_direction
    
    def __add_priors__(self, A, b):
        
        A /= self.current_status.sigma2
        for j in range(len(self.priors)):
            pr = self.priors[j]
            if pr.density == Prior.GAUSSIAN:
                A[j][j] += pr.bta
                b[j]    -= pr.bta * (self.theta[j] - pr.mu)
            if pr.density == Prior.LOGNORMAL:
                log_theta_over_mu = np.log(self.theta[j] / pr.mu)
                A[j][j] += pr.bta / np.power(self.theta[j], 2.) #- (1. + (log_theta_over_mu - 1.) * pr.precision) / np.power(self.theta[i], 2.)
                b[j]    -= pr.bta * (log_theta_over_mu + 1. / pr.bta) / self.theta[j]

        return A, b

    def __move_to_new_theta__(self, descent_direction):
            
        norm_desc_dir = np.linalg.norm(descent_direction)
        descent_direction = descent_direction / norm_desc_dir
        
        self.status = self.__get_optimization_status__(self.theta)

        flg_theta_updated  = False
        while True:
            theta_new = self.theta + self.step * descent_direction
            if self.lower is not None:
                theta_new = np.clip(theta_new, self.lower, self.upper)
                
            new_status = self.__get_optimization_status__(theta_new)
            
            if new_status.Objective < self.current_status.Objective * (1. - 1E-5): # there has been a significant % decrease
                self.current_status = new_status
                self.theta = theta_new
                self.total_displacement += self.step * norm_desc_dir
                flg_theta_updated = True
                self.step *= self.step_mult_up
            else:
                if flg_theta_updated:
                    break
                self.step *= self.step_mult_down # try to decrease the step
            
            if self.step < self.min_displacement:
                break
        
        if not flg_theta_updated: # update lambda
            self.lbda = min(self.max_lbda, self.lbda * self.lbda_mult_up)
            
                
    def __get_optimization_status__(self, theta):
        
        yEst = self.model_fn(self.X, theta)
        err  = self.y - yEst
        WSS  = sum(self.weights * np.power(err, 2.))
        sigma2 = WSS # FIXME rivedere, va divisa per nObs per ottenere varianza stimata
        
        Objective = WSS
        if self.priors is not None:
            Objective = 0.5 * self.nObs * np.log(sigma2) + 0.5 * WSS / sigma2
            for j in range(len(self.priors)):
                pr = self.priors[j]
                if pr.density == Prior.GAUSSIAN:
                    Objective += 0.5 * pr.bta * np.power(theta[j] - pr.mu, 2.)
                elif pr.density == Prior.LOGNORMAL:
                    Objective += np.log(theta[j]) + 0.5 * pr.bta * np.power(np.log(theta[j] / pr.mu), 2.)
        
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

    def __improve_conditioning__(self, A):
        
        flg_matrix_changed = False
        if max(abs(np.diag(A)) - 1.) > 1E-5:
            # Are there any zero rows in A? If so, put a 1. on their diagonal for those rows only.
            zero_rows = np.where(np.max(np.abs(A), axis = 1) < 1E-5)[0]
            if len(zero_rows) > 0:
                A.put([(A.shape[1] + 1) * i for i in zero_rows], 1.)
            else:
                # Last attempt: set all elements on the diagonal = 1.
                np.fill_diagonal(A, 1.)
                
            flg_matrix_changed = True

        return flg_matrix_changed
                
    def __set_bounds__(self, bounds):
        
        if bounds is None:
            lower = None
            upper = None
        else:
            lower, upper = [np.array(x) for x in zip(*bounds)]
            lower[lower == None] = -1E+30
            upper[upper == None] = +1E+30

        return lower, upper
