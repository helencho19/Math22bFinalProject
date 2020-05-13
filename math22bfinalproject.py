#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 20:17:28 2020

@author: helencho
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# Define data matrix
A = np.array([[3.0, 3.5, 3.3, 3.8, 3.6], [6, 7, 9, 8, 8]])
# Number of rows
m = A.shape[0]
# Numer of columns
n = A.shape[1]
#print(A)

# Find sample mean 
M = ((1 / n) * (np.sum(A, axis=1))).reshape(2, 1)
print("sample mean:\n", M) 

# Define data matrix in mean-deviation form
B = np.array([])
for i in range(n):
    if B.size == 0:
        B = A[:,i].reshape(2, 1) - M
    else:
        B = np.c_[B, A[:,i].reshape(2, 1) - M]
print("mean deviation form:\n", B)
        
# Define sample covariance matrix
S = (1 / (n - 1)) * (B.dot(B.transpose()))
print("sample covariance matrix:\n", S)

# Diagonalize S
eigvals, eigvecs = la.eig(S)
eigvals = eigvals.real
print("eigenvalues: ", eigvals)
print("eigenvectors:\n", eigvecs)
eigval_vecs_dict = {}
for i in range(S.shape[1]):
    eigval_vecs_dict[eigvals[i]] = eigvecs[:,i].reshape(2,1)
sorted_eigval_vecs_lst = sorted(eigval_vecs_dict, reverse=True)
P = np.array([])
for e in sorted_eigval_vecs_lst:
    if P.size == 0:
        P = eigval_vecs_dict[e]
    else:
        P = np.c_[P, eigval_vecs_dict[e]]
print("P:\n", P)
dotprod = P[:,0] @ P[:,1]
print("dot product of e'vectors: ", dotprod)

# Visualize Data
# https://www.earthdatascience.org/courses/earth-analytics-bootcamp/numpy-arrays/manipulate-summarize-plot-numpy-arrays/
# Original Data
plt.rcParams["figure.figsize"] = (8, 8)
fig, ax = plt.subplots()
ax.scatter(A[0], A[1])
ax.set(title="Harvard Students Data")
ax.set(xlabel="GPA", ylabel="Avg Hours of Sleep per Night")

# Centered Data
plt.xlim((-3, 3))
plt.ylim((-3, 3))
fig, ax = plt.subplots()
ax.scatter(B[0], B[1])
ax.set(title="Harvard Students Data (Centered)")
ax.set(xlabel="GPA", ylabel="Avg Hours of Sleep per Night")

# Add Principal Components
# https://stackoverflow.com/questions/42281966/how-to-plot-vectors-in-python-using-matplotlib
origin = [0], [0]
plt.quiver(*origin, -1 * P[:,0], -1 * P[:,1], color=['r', 'g'], scale=5)
