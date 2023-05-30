# This module is writen according to "An Introduction to Computational Stochastic PDE's"
# from Gabrial J. Lord. All credit goes to him!

import h5py
import click
import numpy as np
import os
import math
from scipy.special import kv


def initialize_SDEsolver(params):
    
    # The space discretization needs to be equidistant. We will re-map it for the variable: "mixing lenght"    
    # Space points
    params.s_grid  = np.linspace(params.z0, params.z0 + params.Hs, params.Ns_n)
    # dz = params.s_grid[1] - params.s_grid[0] 

    M = 1
    q = 15
    
    stoch_solver = SDEsolver(params.Ns_n, M, params.dz_s, q) 
    
    return stoch_solver, params

class SDEsolver:
    def __init__(self, dof, ext_dof, dz, q):
        self.__dof = dof
        self.__ext_dof = ext_dof
        self.__dz = dz
        self.__q = q
        self.__state = np.zeros( self.__dof )

    def __circ_embed_approx(self, vector):
        # mirror the vector without last element
        vector_mir = vector[-2:0:-1]

        # extend vector to make it conjugate symmetric
        vector_circ = np.concatenate( (vector, vector_mir), axis=0)

        N_circ = np.size(vector_circ)

        # compute the egenvalues with ifft
        d = np.real( np.fft.ifft(vector_circ) ) * N_circ

        # split the eigenvalues
        d_min = np.copy(d)
        d_min[ d_min > 0] = 0

        d_pos = np.copy(d) 
        d_pos[ d_pos < 0] = 0

        # inform if the matrix is non-negative
        if (np.max(-d_min) > 1e-9):
            print('Covariance matrix is not non-negative.', 'Max negative value: ', np.max(-d_min)) 

        # generate complex gaussian
        xi = np.dot( np.random.randn(N_circ,2), np.array([1, 1j]) )

        # sample with fft
        dxi = np.multiply( np.power(d_pos,0.5), xi)
        Z = np.fft.fft( dxi  ) / np.sqrt(N_circ)

        # select the sample paths
        N = np.size(vector)
        X = np.real(Z[0:N])
        Y = np.imag(Z[0:N])

        return X,Y
    
    
    def __matern_sampling(self, N, M, dt, q):
        #  N: is length of the sampled space
        #  M: is extantion to better approx the non-negative matrix
        # dt: space incriment
        #  q: parameter of matern covariance. ~ length of correlation
        N_dash = N+M-1
        c = np.zeros( N_dash+1 )
        t = dt* np.arange(N_dash)

        # matern covariance constants
        c[0] = 1;
        const = np.power( 2,(q-1) ) * math.gamma(q)

        # construct the matern covariance
        for i in range(1,N_dash):
            c[i] = ( np.power(t[i],q) * kv(q, t[i]) )/const

        X,Y = self.__circ_embed_approx(c)
        X = X[0:N]
        Y = Y[0:N]
        t = t[0:N]

        return t, X, Y, c


    def __gauss_sampling(self, N, M, dt, q):
        #  N: is length of the sampled space
        #  M: is extantion to better approx the non-negative matrix
        # dt: space incriment
        #  q: parameter of matern covariance. ~ length of correlation
        N_dash = N+M-1
        c = np.zeros( N_dash+1 )
        t = dt* np.arange(N_dash)

        c = np.exp(-(t/q)**2 )

        X,Y = self.__circ_embed_approx(c)
        X = X[0:N]
        Y = Y[0:N]
        t = t[0:N]
        
        return t, X, Y, c

    
    def evolve(self, x0, dt, a1, a2, a3, flag, v_sqrt, lz):

        if flag is False:
            x, dW, dW2, c = self.__gauss_sampling( self.__dof, self.__ext_dof, self.__dz, lz)
            flag = True
        else:
            dW = np.copy(dW2)
            flag = False

        R = (1.0 + a1 * x0 - a2 * np.power(x0, 2)) / 3600
        g1 = a3 * x0 / 60
        g1_prim = a3 / 60 
        
        x1  = x0 + R * dt + g1 * v_sqrt * dW + 0.5 * g1 * g1_prim * (np.power(v_sqrt * dW, 2)  - dt)
        #x1  = x0 + R * dt + g * v_sqrt * dW 
        
        self.__state = x1

    def getState(self):
        return self.__state

    
def A1(x):
    a = 9.32123
    b = 0.908809
    # c = 0.0738121
    c = 0.0538121
    d = 8.32201
    
    out = a * np.tanh(b*x + c) + d
    return out  


def A2(x): 
    a = 0.429429
    b = 0.1749

    out = a * x + b
    
    return out  


def A3(x, params):
    a = 0.806905
    b = 0.60448
    c = 0.836781
    # c = 0.936781
    #d = -0.00
    out = a * np.tanh(b*x + c) + params.d
    
    return out


def l_s(U):
    a = 0.06009875048315001
    b = 1.2319964831612578
    return 10**( a*U + b ) 
