#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 20:54:19 2021

@author: slava
"""

# standard imports
import matplotlib.pyplot as plt
from tqdm import tqdm
from fenics import *
import numpy as np
import h5py

# project related imports
from lib import fenics_utility_functions as fut 
from lib import utility_functions as ut

def save_current_result(output, params, fparams, i, us, vs, Ts, ks, phi_s):
       
    # convert 2 numpy array and save
    output.U_save[:, i] = np.flipud( interpolate(us, fparams.Q).vector().get_local() )
    output.V_save[:, i] = np.flipud( interpolate(vs, fparams.Q).vector().get_local() )
    output.T_save[:, i] = np.flipud( interpolate(Ts, fparams.Q).vector().get_local() )
    output.k_save[:, i] = np.flipud( interpolate(ks, fparams.Q).vector().get_local() )
    
    # Write Ri number
    calc_Ri = project(fut.Ri(fparams, params), fparams.Q)
    output.Ri_save[:, i] = np.flipud(calc_Ri.vector().get_local())
    
    # Write Km
    calc_Km = project(fut.K_m(fparams, params), fparams.Q)
    output.Km_save[:, i] = np.flipud(calc_Km.vector().get_local())
    
    # Write Kh
    calc_Kh = project(fut.K_h(fparams, params), fparams.Q)
    output.Kh_save[:, i] = np.flipud(calc_Kh.vector().get_local())
    
    # Write the stochastic phi_0 (currently its  a numpy array)
    calc_phi = project(fut.l_m(fparams, params), fparams.Q)
    output.phi_stoch[:, i] = np.flipud(calc_phi.vector().get_local())
    
    return output

def calc_variables_np(params, fparams, us, vs):
       
    # convert 2 numpy array and save
    U_save = np.flipud( interpolate(us, fparams.Q).vector().get_local() )
    V_save = np.flipud( interpolate(vs, fparams.Q).vector().get_local() )
    #T_save = np.flipud( interpolate(Ts, fparams.Q).vector().get_local() )
    #k_save = np.flipud( interpolate(ks, fparams.Q).vector().get_local() )
    
    # Write Ri number
    #calc_Ri = project(fut.Ri(fparams, params), fparams.Q)
    #Ri_save = np.flipud(calc_Ri.vector().get_local())
    
    # Write Km
    #calc_Km = project(fut.K_m(fparams, params), fparams.Q)
    #Km_save = np.flipud(calc_Km.vector().get_local())
    
    # Write Kh
    calc_Kh = project(fut.K_h(fparams, params), fparams.Q)
    Kh_save = np.flipud(calc_Kh.vector().get_local())
    
    return U_save, V_save, Kh_save

def initialize(output, params):
    output.U_save = np.zeros((params.Nz, params.Save_tot))
    output.V_save = np.zeros((params.Nz, params.Save_tot))
    output.T_save = np.zeros((params.Nz, params.Save_tot))
    output.k_save = np.zeros((params.Nz, params.Save_tot))
    output.Ri_save = np.zeros((params.Nz, params.Save_tot))
    output.Kh_save = np.zeros((params.Nz, params.Save_tot))
    output.Km_save = np.zeros((params.Nz, params.Save_tot))
    output.phi_stoch = np.zeros((params.Nz, params.Save_tot))
    
    return output


def save_solution(output, params, fparams):
    
    if params.save_ini_cond:
        np.save(params.initCondStr + '_u', output.U_save[:,-2])
        np.save(params.initCondStr + '_v', output.V_save[:,-2])
        np.save(params.initCondStr + '_T', output.T_save[:,-2])
        np.save(params.initCondStr + '_k', output.k_save[:,-2])
        print('\n Current solution saved as initial condition')

    saveFile = h5py.File('solution/'+ params.solStringName +'.h5','w')
    
    U_ds = saveFile.create_dataset('/U',(params.Nz, params.Save_tot), h5py.h5t.IEEE_F64BE)
    V_ds = saveFile.create_dataset('/V',(params.Nz, params.Save_tot), h5py.h5t.IEEE_F64BE)
    T_ds = saveFile.create_dataset('/Temp',(params.Nz, params.Save_tot), h5py.h5t.IEEE_F64BE)
    k_ds = saveFile.create_dataset('/TKE', (params.Nz, params.Save_tot), h5py.h5t.IEEE_F64BE)
    Ri_ds = saveFile.create_dataset('/Ri',(params.Nz, params.Save_tot), h5py.h5t.IEEE_F64BE)
    Kh_ds = saveFile.create_dataset('/Kh',(params.Nz, params.Save_tot), h5py.h5t.IEEE_F64BE)
    Km_ds = saveFile.create_dataset('/Km',(params.Nz, params.Save_tot), h5py.h5t.IEEE_F64BE)
    phi_ds = saveFile.create_dataset('/phi',(params.Nz, params.Save_tot), h5py.h5t.IEEE_F64BE)
    
    z_ds = saveFile.create_dataset('/z', (np.size(fparams.z),1) , h5py.h5t.IEEE_F64BE)
    
    SimStart_ds = saveFile.create_dataset('/SimStart', (1,1) , h5py.h5t.IEEE_F64BE)
    SimEnd_ds = saveFile.create_dataset('/SimEnd', (1,1) , h5py.h5t.IEEE_F64BE)
    
    dt_ds = saveFile.create_dataset('/dt', (1,1) , h5py.h5t.IEEE_F64BE)
    dts_ds = saveFile.create_dataset('/sdt', (1,1) , h5py.h5t.IEEE_F64BE)
    
    U_ds[...] = output.U_save
    V_ds[...] = output.V_save
    T_ds[...] = output.T_save
    k_ds[...] = output.k_save
    Ri_ds[...] = output.Ri_save
    Kh_ds[...] = output.Kh_save
    Km_ds[...] = output.Km_save
    phi_ds[...] = output.phi_stoch
    
    z_ds[...] = fparams.z
    dt_ds[...] = params.Save_dt
    SimStart_ds[...] = 1
    SimEnd_ds[...] = params.Save_tot
    dts_ds[...] = params.SimSav
    
    saveFile.close()
    print('simulation is done')