#!/usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/cartoGRAPHs_app/cartoGRAPHs_app/")

print('CSDEBUG: path inserted successfully')

#from app import myServer as application
#application.secret_key = 'Add your secret key'

print('CSDEBUG: app_main imports complete')
from numba import config, njit, threading_layer
print('CSDEBUG: numba imports successful')
# set the threading layer before any parallel target compilation
config.THREADING_LAYER = 'omp'
print('threading layer set')

@njit('float64(float64[::1], float64[::1])')
def foo(a, b):
    return a[1] + b[2]
print('COMPILED OK')

x = np.arange(10.)
y = x.copy()

# # this will force the compilation of the function, select a threading layer
# # and then execute in parallel
print(foo(x, y))
print('EXECUTED OK')
print('CSDEBUG: function compilation successful')

print('CSDEBUG: wsgi run')
