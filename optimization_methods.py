# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 20:31:28 2024

@author: gbulb
"""
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo
"""
Convex Optimization
"""

def fm(x, y):
    return (np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2)
f = plt.figure(figsize=(9, 6))
x = np.linspace(-10, 10, 50)
y=x
def plotting_3D(fig,x,y):
    X, Y = np.meshgrid(x, y)
    Z = fm(X, Y)
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=mpl.cm.coolwarm, linewidth=0.5, antialiased=True)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x, y)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.title('Convex optimization')
    plt.show()


plotting_3D(f,x,y)
"""
Global Optimization
"""
# simulated annealing global optimization for a multimodal objective function
from scipy.optimize import dual_annealing
 
# objective function
def objective(v):
    x,y=v
    return np.sin(x) + 0.05 * x ** 2 + np.sin(y) + 0.05 * y ** 2
 

r_min, r_max = -5.0, 5.0
# bounds
bounds = [[r_min, r_max], [r_min, r_max]]
# perform the simulated annealing search
result = dual_annealing(objective, bounds)
print(result)
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))

f = plt.figure(figsize=(9, 6))
x = np.linspace(-10, 10, 50)
y=x
v=(x,y)
#plt.plot(objective(v))
V =X,Y= np.meshgrid(x, y)
Z = objective(V)
ax = f.gca(projection='3d')
CS1 = ax.contour(X, Y, Z) 

ax.set_title('Global Optimization Example') 
plt.show() 
"""
Local Optimization
"""
from scipy.optimize import minimize
from numpy.random import rand
 
# objective function
def objective(x):
	return np.sin(x[0]) + 0.05 *x[0]**2 + np.sin(x[1]) + 0.05 *x[1]**2
 
# bound
r_min, r_max = -5.0, 5.0
# initial guess
initial_guess = r_min + rand(2) * (r_max - r_min)
# perform the l-bfgs-b algorithm search
result = minimize(objective, initial_guess, method='L-BFGS-B')
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print(result)

f = plt.figure(figsize=(9, 6))
x = np.linspace(-15, 15, 50),np.linspace(-15, 15, 50)

v=(x,x)
#plt.plot(objective(v))
V =X,Y= np.meshgrid(x, y)
Z = objective(V)
ax = f.gca(projection='3d')
CS1 = ax.contour(X, Y, Z)   
ax.set_title('Local Optimization Example') 
plt.show() 
