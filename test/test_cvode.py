# Use the CVODE integrator to solve the Van der Pol equation

from pandas import options
from sksundae.cvode import CVODE
import numpy as np

def rhsfn(t, y, yp, userdata=()):
    a=userdata[0]
    #print('a:',a)
    yp[0] = y[1]+a[0]
    yp[1] = 1000*(1 - y[0]**2)*y[1] - y[0]+userdata[1]

options = {'rtol': 1e-6, 'atol': 1e-10}
# time series from  t from 1 to 3000
# iniitial conditions y0=2, y1=0
t=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
y0=[2,0]
# set userdata to be 2*t and 0
a=0.0000002*np.array(t)
solver = CVODE(rhsfn,userdata=(a,0), **options)
solver.init_step(t[0], [2, 0])
results_a=[]
for i in range(10):
    result=solver.step(t[i])
    results_a.append((result.t, result.y))

print(results_a)