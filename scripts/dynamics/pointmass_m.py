from itertools import combinations_with_replacement
from params import *
import sympy as sm
from sympy import Symbol,lambdify
from sympy.physics.mechanics import dynamicsymbols

class MDynamics:
    def __init__(self) -> None:
        self.m1,self.m2,self.m3,self.l0,self.l1,self.l2,self.r,self.g = sm.symbols('m1 m2 m3 l0 l1 l2 r g', real =True)
        t = Symbol('t')
        q1, q2, q3 = dynamicsymbols('q1 q2 q3')
        q1d = dynamicsymbols('q1',1)
        q2d = dynamicsymbols('q2',1)
        q3d = dynamicsymbols('q3',1)
        q1dd = dynamicsymbols('q1',2)
        q2dd = dynamicsymbols('q2',2)
        q3dd = dynamicsymbols('q3',2)

        print('evaluating dynamics')

        x1 = (l1*sm.cos(q2))*sm.cos(q1)
        y1 = (l1*sm.cos(q2))*sm.sin(q1)
        z1 = l0 + l1*sm.sin(q2)

        x2 = (l1*sm.cos(q2)+l2*sm.cos(q2+q3))*sm.cos(q1)
        y2 = (l1*sm.cos(q2)+l2*sm.cos(q2+q3))*sm.sin(q1)
        z2 = l0 + l1*sm.sin(q2) + l2*sm.sin(q2+q3)

        x1_dot = sm.diff(x1,t)
        x2_dot = sm.diff(x2,t)
        y1_dot = sm.diff(y1,t)
        y2_dot = sm.diff(y2,t)
        z1_dot = sm.diff(z1,t)
        z2_dot = sm.diff(z2,t)

        # kinetic energy
        ke = (sm.Rational(1/2))*m1*(x1_dot**2)
        ke += (sm.Rational(1/2))*m1*(y1_dot**2)
        ke += (sm.Rational(1/2))*m2*(x2_dot**2)
        ke += (sm.Rational(1/2))*m2*(y2_dot**2)
        
        ke += (sm.Rational(1/2))*m2*(z1_dot**2)
        ke += (sm.Rational(1/2))*m2*(z2_dot**2)

        # potential energy
        pe = m1*g*z1
        pe += m2*g*z2


        L = ke - pe

        f1 = sm.diff(sm.diff(L,q1d),t) - sm.diff(L,q1)
        f2 = sm.diff(sm.diff(L,q2d),t) - sm.diff(L,q2)
        f3 = sm.diff(sm.diff(L,q3d),t) - sm.diff(L,q3)
        f1sim = sm.expand(sm.simplify(f1))
        f2sim = sm.expand(sm.simplify(f2))
        f3sim = sm.expand(sm.simplify(f3))

        print('generating mass matrix template')
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
        self.M_f=lambdify([self.m1,self.m2,self.m3,self.l0,self.l1,self.l2,self.r,self.g,q1,q2,q3],M, "numpy")

        print('generating coriolis matrix template')
        #centrifugal / Coriolis force,
        rows = []
        coordinates = list(combinations_with_replacement([q1d,q2d,q3d], 2))
        c_v = [sm.prod(i) for i in coordinates]
        for f in [f1sim,f2sim,f3sim]:
            columns = []
            for v in c_v:
                columns.append(f.coeff(v)*v)
            rows.append(sum(columns))
        C = sm.Matrix(rows)
        self.C_f=lambdify([self.m1,self.m2,self.m3,self.l0,self.l1,self.l2,self.r,self.g,q1,q2,q3,q1d,q2d,q3d],C, "numpy")

        print('generating gravity matrix template')
        g1 = f1sim - sm.expand(m11*q1dd) - sm.expand(m12*q2dd) - sm.expand(m13*q3dd) - sm.expand(C[0])
        g2 = f2sim - sm.expand(m21*q1dd) - sm.expand(m22*q2dd) - sm.expand(m23*q3dd) - sm.expand(C[1])
        g3 = f3sim - sm.expand(m31*q1dd) - sm.expand(m32*q2dd) - sm.expand(m33*q3dd) - sm.expand(C[2])
        G = sm.Matrix([g1, g2, g3])
        self.G_f=lambdify([self.m1,self.m2,self.m3,self.l0,self.l1,self.l2,self.r,self.g,q1,q2,q3],G, "numpy")


    def compute_mass_matrix(self,s):
        q1v = s[0]
        q2v = s[2]
        q3v = s[4]
        M = self.M_f(m1,m2,m3,l0,l1,l2,r,g,q1v,q2v,q3v)
        return M
    
    def compute_coriolis_matrix(self,s):
        q1v = s[0]
        q2v = s[2]
        q3v = s[4]
        q1_dotv = s[1]
        q2_dotv = s[3]
        q3_dotv = s[5]
        C = self.C_f(m1,m2,m3,l0,l1,l2,r,g,q1v,q2v,q3v,q1_dotv,q2_dotv,q3_dotv)
        return C

    def compute_gravity_matrix(self,s):
        q1v = s[0]
        q2v = s[2]
        q3v = s[4]
        G = self.G_f(m1,m2,m3,l0,l1,l2,r,g,q1v,q2v,q3v)
        return G