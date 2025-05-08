from sympy.physics.mechanics import dynamicsymbols
import sympy as sp
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg
import scipy as scpy

from sympy.physics.vector import init_vprinting
Axes3D = Axes3D

init_vprinting(use_latex='mathjax', pretty_print=False)

a, alpha, d, theta, theta1, theta2, theta3, theta4, theta5, theta6, l1, a2, a3, l4, l5, l6 = dynamicsymbols(
    'a alpha d theta theta1 theta2 theta3 theta4 theta5 theta6 l1 a2 a3 l4 l5 l6')

def scos(x): return sp.cos(x).evalf()
def ssin(x): return sp.sin(x).evalf()

def cross(A, B):
    return [A[1]*B[2] - A[2]*B[1], A[2]*B[0] - A[0]*B[2], A[0]*B[1] - A[1]*B[0]]

# Homogeneous Transformation
def DH_transformation(q):
    l1 = 128; a2 = 612.7; a3 = 571.6; l4 = 163.9; l5 = 115.7; l6 = 192.2
    theta1, theta2, theta3, theta4, theta5, theta6 = q

    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), sp.Matrix([[0, 0, 0, 1]]))

    t01 = t.subs({a: 0, alpha: -sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: sp.pi, d: 0, theta: theta2 - sp.pi/2})
    t23 = t.subs({a: a3, alpha: sp.pi, d: 0, theta: theta3})
    t34 = t.subs({a: 0, alpha: sp.pi/2, d: l4, theta: theta4 + sp.pi/2})
    t45 = t.subs({a: 0, alpha: -sp.pi/2, d: l5, theta: theta5})
    t56 = t.subs({a: 0, alpha: 0, d: l6, theta: theta6})

    return t01 * t12 * t23 * t34 * t45 * t56

def DH_jcb(q):
    l1 = 128; a2 = 612.7; a3 = 571.6; l4 = 163.9; l5 = 115.7; l6 = 192.2
    theta1, theta2, theta3, theta4, theta5, theta6 = q

    rot = sp.Matrix([[scos(theta), -ssin(theta)*scos(alpha), ssin(theta)*ssin(alpha)],
                     [ssin(theta), scos(theta)*scos(alpha), -scos(theta)*ssin(alpha)],
                     [0, ssin(alpha), scos(alpha)]])
    trans = sp.Matrix([a*scos(theta), a*ssin(theta), d])
    t = sp.Matrix.vstack(sp.Matrix.hstack(rot, trans), sp.Matrix([[0, 0, 0, 1]]))

    t01 = t.subs({a: 0, alpha: -sp.pi/2, d: l1, theta: theta1})
    t12 = t.subs({a: a2, alpha: sp.pi, d: 0, theta: theta2 - sp.pi/2})
    t23 = t.subs({a: a3, alpha: sp.pi, d: 0, theta: theta3})
    t34 = t.subs({a: 0, alpha: sp.pi/2, d: l4, theta: theta4 + sp.pi/2})
    t45 = t.subs({a: 0, alpha: -sp.pi/2, d: l5, theta: theta5})
    t56 = t.subs({a: 0, alpha: 0, d: l6, theta: theta6})

    return [t01, t01*t12, t01*t12*t23, t01*t12*t23*t34,
            t01*t12*t23*t34*t45, t01*t12*t23*t34*t45*t56]

def jcb(T):
    z = [sp.Matrix([0, 0, 1])]
    for i in T:
        z.append(i[:3, 2])
    o = [sp.Matrix([0, 0, 0])]
    for i in T:
        o.append(i[:3, 3])
    J = sp.zeros(6, 6)
    for i in range(6):
        J[0, i] = sp.Matrix(cross(z[i], o[-1] - o[i]))
    for i in range(6):
        J[3:6, i] = z[i]
    return J
