from numpy import linalg as LA
import numpy as np
import h5py


def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    return np.abs(a - a0).argmin()


solStringName = 'test_reference_solution'

file = h5py.File('solution/' + solStringName + '.h5' , 'r+')
u_ref = file['U'][:]
v_ref = file['V'][:]
T_ref = file['Temp'][:]
k_ref = file['TKE'][:]
file.close()


solStringName = 'def_solution'

file = h5py.File('solution/' + solStringName + '.h5' , 'r+')
u_def = file['U'][:]
v_def = file['V'][:]
T_def = file['Temp'][:]
k_def = file['TKE'][:]
file.close()


def test_fields_difference(a, b, str_name='variable'):
    print('Testing: ' + str_name)
    EPS = 1e-13
    
    diff = LA.norm(a - b)
    if diff < EPS:
        print('PASS')
    else:
        print('FAIL: ' + 'norm = ' + str(diff) + ' > ' + str(EPS))


test_fields_difference(u_def, u_ref, 'u component')
test_fields_difference(v_def, v_ref, 'v component')
test_fields_difference(T_def, T_ref, 'Temperature')
test_fields_difference(k_def, k_ref, 'TKE variable')

