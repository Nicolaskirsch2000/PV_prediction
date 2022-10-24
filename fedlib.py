# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 09:21:09 2022

@author: Kirsch
"""

import numpy as np
import random
import torch
#from torch import nn
import torch.optim as optim

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

from torch import nn

def fed_sgd(data, iterations, N, S, n_feat):
    
    #grad_mean_hist = []
    
    #Definition of the server model
    model_server = nn.Linear(n_feat, 1)
    optimizer_server = optim.Adam(model_server.parameters())
    #loss_fn = nn.MSELoss()


    #One model per clients
    models = [None]*N
    opti = [None]*N
    loss_fct = [None]*N

    for i in range(len(models)):
        models[i] = nn.Linear(n_feat, 1)
        opti[i] = optim.Adam(models[i].parameters())
        loss_fct[i] = nn.MSELoss()

        
    #Iterations
    for i in range(0,iterations):

        #Random selection of clients
        a = list(range(0,N))
        sel = random.sample(a,S)

        grad = []
        bias = []
        loc = 0

        loss = [None]*N
        losses = 0

        #Clients iteration
        for j in sel:
            opti[loc].zero_grad()
            models[loc].weight = model_server.weight
            models[loc].weight.grad = None
            models[loc].bias = model_server.bias
            models[loc].bias.grad = None

            predictions = models[loc](data[j][0])

            loss[loc] = loss_fct[loc](predictions, data[j][1])
            loss[loc].backward()

            grad.append(models[loc].weight.grad.numpy())
            bias.append(models[loc].bias.grad.numpy())
            loc+=1
        grad = np.stack(grad, axis=0)
        grad_mean = torch.FloatTensor(grad.mean(axis = 0))
        
        grad_mean_hist.append(grad_mean)
        

        bias_mean = torch.FloatTensor([np.mean(bias)])
        model_server.weight.grad = grad_mean
        model_server.bias.grad = bias_mean
        optimizer_server.step()
    return model_server, grad_mean_hist
