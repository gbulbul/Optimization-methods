# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 13:13:15 2024

@author: gbulb
"""

"""
Finding min on an unconstrained Convex function
"""
import matplotlib.pyplot as plt
from numpy import arange
from scipy.optimize import minimize_scalar

 
# objective function
def convex_objective(x):
	return (5.0 + x)**2.0
def plotting(f,inputs,targets,convex):
    plt.plot(inputs, targets, '--')
    # plot the optima
    plt.plot([opt_x], [opt_y], 's', color='r')
    plt.title('Finding minimum for a' +" "+str(convex)+ " "+'function')
    # show the plot
    plt.show()
# minimize the function
result = minimize_scalar(convex_objective, method='brent')
# summarize the result
opt_x, opt_y = result['x'], result['fun']
print('Optimal Input x: %.6f' % opt_x)
print('Optimal Output f(x): %.6f' % opt_y)
# define the bounds
r_min, r_max = -10.0, 10.0
# prepare inputs
inputs = arange(r_min, r_max, 0.1)
# compute targets
targets = [convex_objective(x) for x in inputs]
f=plt.figure(figsize=(10, 6))
plotting(f,inputs,targets,convex='convex')



"""
Finding min on non-Convex function
"""
# optimize non-convex objective function
 
# objective function
def Non_convex_objective(x):
	return (x - 2.0) * x * (x + 2.0)**2.0
 
# minimize the function
result = minimize_scalar(Non_convex_objective, method='brent')
# summarize the result
opt_x, opt_y = result['x'], result['fun']
print('Optimal Input x: %.6f' % opt_x)
print('Optimal Output f(x): %.6f' % opt_y)
# define the range
r_min, r_max = -3.0, 2.5
# prepare inputs
inputs = arange(r_min, r_max, 0.1)
# compute targets
targets = [Non_convex_objective(x) for x in inputs]

plotting(f,inputs,targets,convex='nonconvex')
