# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 09:21:09 2022

@author: Kirsch
"""

import numpy as np
import random

def theta_init(X):
    """ Generate an initial value of vector θ from the original independent variables matrix
         Parameters:
          X:  independent variables matrix
        Return value: a vector of theta filled with initial guess
    """
    theta = np.random.randn(len(X[0])+1, 1)
    return theta

def generateXvector(X):
    """ Taking the original independent variables matrix and add a row of 1 which corresponds to x_0
        Parameters:
          X:  independent variables matrix
        Return value: the matrix that contains all the values in the dataset, not include the outcomes variables. 
    """
    vectorX = np.c_[np.ones((len(X), 1)), X]
    return vectorX

def federated_sgd(data, iterations, N, S, n_feature, learning_rate,w_model):
    cost_lst = []
    for i in range(0,iterations):
        #Random selection of clients
        a = list(range(1,N))
        sel = random.sample(a,S)

        grad = [None]*S
        loc = 0

        #Clients iteration
        for j in sel:

            w_cli = w_model

            grad[loc] = 2/len(data[j][0]) * data[j][0].T.dot(data[j][0].dot(w_cli) - data[j][1].reshape(len(data[j][1]),1))

            loc +=1

        g = np.stack(grad, axis=0)
        grad_mean = g.mean(axis = 0)


        w_model = w_model  - learning_rate*grad_mean
        
    return w_model, cost_lst