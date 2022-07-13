from params import *
import sympy as sm
from sympy import Symbol
from sympy.physics.mechanics import dynamicsymbols

class MDynamics:
    def __init__(self) -> None:
        m1,m2,m3,l0,l1,l2,r,g = sm.symbols('m1 m2 m3 l0 l1 l2 r g', real =True)
        t = Symbol('t')
        q1, q2, q3 = dynamicsymbols('q1 q2 q3')
        q1d = dynamicsymbols('q1',1)
        q2d = dynamicsymbols('q2',1)
        q3d = dynamicsymbols('q3',1)
        q1dd = dynamicsymbols('q1',2)
        q2dd = dynamicsymbols('q2',2)
        q3dd = dynamicsymbols('q3',2)

        I0 = m1*(r**2)/2
        k0 = 1/2 * I0 *q1d**2
        p0 = 1/2*m1*g*l0

        x2 = (l1*sm.cos(q2))*sm.cos(q1)
        y2 = (l1*sm.cos(q2))*sm.sin(q1)
        z2 = l0 + l1*sm.sin(q2)
        x2_d = sm.diff(x2,t)
        y2_d = sm.diff(y2,t)
        z2_d = sm.diff(z2,t)
        I1 = 1/12*m1*(l1**2) + 1/4*m1*(r**2)
        v2_2 = (x2_d**2)+(y2_d**2)+(z2_d**2)
        w2 = q2d
        k1 = 1/2*m1*v2_2 + 1/2*I1*(w2**2)
        p1 = (l0+1/2*l1*sm.sin(q2))*m2*g

        x3 = (l1*sm.cos(q2)+l2*sm.cos(q2+q3))*sm.cos(q1)
        y3 = (l1*sm.cos(q2)+l2*sm.cos(q2+q3))*sm.sin(q1)
        z3 = l0 + l1*sm.sin(q2) + l2*sm.sin(q2+q3)
        x3_d = sm.diff(x3,t)
        y3_d = sm.diff(y3,t)
        z3_d = sm.diff(z3,t)
        I3 = 1/12*m3*(l2**2) + 1/4*m3*(r**2)
        v3_2 = (x3_d**2)+(y3_d**2)+(z3_d**2)
        w3 = q2d+q3d
        k2 = 1/2*m3*v3_2 + 1/2*I1*(w3**2)
        p2 = (l0+l1*sm.sin(q2)+1/2*l2*sm.sin(q2+q3))*m2*g

        L = k0+k1+k2-p0-p1-p2

        f1 = sm.diff(sm.diff(L,q1d),t) - sm.diff(L,q1)
        f2 = sm.diff(sm.diff(L,q2d),t) - sm.diff(L,q2)
        f3 = sm.diff(sm.diff(L,q3d),t) - sm.diff(L,q3)
        f1sim = sm.expand(sm.simplify(f1))
        f2sim = sm.expand(sm.simplify(f2))
        f3sim = sm.expand(sm.simplify(f3))

        m11 = f1sim.coeff(q1dd)
        m12 = f1sim.coeff(q2dd)
        m13 = f1sim.coeff(q3dd)

        m21 = f2sim.coeff(q1dd)
        m22 = f2sim.coeff(q2dd)
        m23 = f2sim.coeff(q3dd)

        m31 = f3sim.coeff(q1dd)
        m32 = f3sim.coeff(q2dd)
        m33 = f3sim.coeff(q3dd)

        M = sm.Matrix([[m11, m12, m13], [m21, m22, m23], [m31, m32 ,m33]])

    def compute_mass_matrix(self,pos):
        pass
    
    def compute_coriolis_matrix(self,s):
        pass

    def compute_gravity_matrix(self,pos):
        pass