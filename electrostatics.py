#!/usr/bin/env python

# This Python script implements a simple numerical procedure for solving
# Maxwell's electrostatic equations for the case of a spherical conductor with
# a conical aperture. The problem is discussed in the following paper:
#
#       Lamberti & Prato, 1991. Am. J. Phys., vol. 59, p. 68

import numpy as np
import math
import matplotlib.pyplot as plt
import sys

# Set up grid of angles.
theta_i = np.arange(0.0, math.pi, 0.01)

# Set parameters of problem. (These are the ones used in Lamberti & Prato).
alpha = 2.44     # angle from opposite pole of aperture to edge of aperture (same as \pi - [angle of aperture])
a = 1.0          # radius of spherical cavity (unitless)
V = -10.0         # potential on surface of cavity (unitless)

# Sanity checks:
if (alpha < 0):
    sys.exit("alpha < 0!")
elif (alpha > math.pi):
    sys.exit("alpha > pi!")

if (a < 0):
    sys.exit("radius of the sphere cannot be negative!")

# Calculate i for which theta_i <= alpha.
n = len([i for i in theta_i if i <= alpha])

# Total number of angle points defines the size of the system of linear equations to solve.
N = len(theta_i)

M = np.zeros([N,N])

from scipy.special import eval_legendre     # Legendre polynomials

# Set up LHS of matrix equation (see Eq. 6 of Lamberti & Prato).
for i in range(N):
    for j in range(N):
        # for angles outside the aperture
        if (i >= 0 and i <= n):
            M[i][j] = math.pow(a,j) * eval_legendre(j, math.cos(theta_i[i]))
        # for angles inside the aperture
        else:
            M[i][j] = (2.0*j + 1.0) * math.pow(a, j-1) * eval_legendre(j, math.cos(theta_i[i]))

# Set up RHS vector.
V_rhs = np.zeros(N)
for i in range(N):
    if (i >= 0 and i <= n):
        V_rhs[i] = V

# Solve linear system for the coefficients A_i in Eqs. 1-2 of Lamberti & Prato.
A = np.linalg.solve(M, V_rhs)

# Calculate the surface charge density on the conductor surface (see Eq. 7 of Lamberti & Prato).
sigma = np.zeros(N)
# Note that we only consider angles with theta_i <= alpha, but we include
# contributions from all A_i, not just from A_i where theta_i < alpha.
for i in range(n):
    for j in range(N):
        sigma[i] += (2.0*j + 1.0) * A[j] * math.pow(a, j-1) * eval_legendre(j, math.cos(theta_i[i]))
sigma *= (1.0 / (4.0 * math.pi))


# Set up figure for pretty plots.
golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
fig = plt.figure( figsize=( 11,0, 11.0/golden_ratio ), dpi=128 )
ax = fig.add_subplot(111)

ax.plot(theta_i, sigma)

ax.set_xlabel(r'$\theta$')
ax.set_ylabel(r'$\sigma$')
ax.set_title('Surface charge density on a spherical conductor with an aperture')

figname = 'surf_charge_dens__alpha_%.2f_a_%.2f_V_%.2f' % (alpha, a, V)
fig.savefig(figname + '.png')
