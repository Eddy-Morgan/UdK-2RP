import sympy as sm
from sympy import  Symbol, cos, Rational
from sympy.physics.mechanics import dynamicsymbols
from itertools import combinations_with_replacement

class MDtensor():
    def __init__(self):
        self.m1, self.m2, self.l1, self.l2, self.g ,self.tau1, self.tau2,self.tau3,self.tau4, self.r1,self.r2 = sm.symbols('m1 m2 l1 l2 g tau1 tau2 tau3 tau4 c1 c2 c3 c4 L r1 r2', real =True)
        self.t = Symbol('t')
        self.theta1, self.theta2, self.phi1, self.phi2 =  dynamicsymbols('theta1 theta2 phi1 phi2')

        self.theta1d = dynamicsymbols('theta1',1)
        self.theta2d = dynamicsymbols('theta2',1)
        self.phi1d = dynamicsymbols('phi1',1)
        self.phi2d = dynamicsymbols('phi2',1)
        self.theta1dd = dynamicsymbols('theta1',2)
        self.theta2dd = dynamicsymbols('theta2',2)
        self.phi1dd = dynamicsymbols('phi1',2)
        self.phi2dd = dynamicsymbols('phi2',2)

        self.x1_ = 0.5*self.l1*sm.cos(self.theta1)*sm.sin(self.phi1)
        self.y1_ = 0.5*self.l1*sm.sin(self.theta1)*sm.sin(self.phi1)
        self.z1_ = 0.5*self.l1*sm.cos(self.phi1)

        self.x2_ = self.l1*sm.cos(self.theta1)*sm.sin(self.phi1) + 0.5*self.l2*sm.cos(self.theta2)*sm.sin(self.phi2)
        self.y2_ = self.l1*sm.sin(self.theta1)*sm.sin(self.phi1) + 0.5*self.l2*sm.sin(self.theta2)*sm.sin(self.phi2)
        self.z2_ = self.l1*sm.cos(self.phi1) + 0.5*self.l2*cos(self.phi2 )

        x1_dot = sm.diff(self.x1_,self.t)
        y1_dot = sm.diff(self.y1_,self.t)
        z1_dot = sm.diff(self.z1_,self.t)

        x2_dot = sm.diff(self.x2_,self.t)
        y2_dot = sm.diff(self.y2_,self.t)
        z2_dot = sm.diff(self.z2_,self.t)

        I1 = Rational('1/12')*self.m1*self.l1**2 + Rational('1/4')*self.m1*self.r1**2

        v1 = Rational('0.5')*self.l1*self.theta1d
        w1 = self.theta1d
        k1 = Rational('1/2')*self.m1*v1**2 + Rational('1/2')*I1*w1**2
        v2 = x2_dot**2 + y2_dot**2 + z2_dot**2
        w2 = self.theta1d + self.theta2d
        I2 = Rational('1/12')*self.m2*self.l2**2 + Rational('1/4')*self.m2*self.r2**2
        k2 = Rational('1/2')*self.m2*v2**2 + Rational('1/2')*I2*w2**2
        p1 = self.m1*self.g*self.z1_
        p2 = self.m2*self.g*self.z2_

        self.ke = k1 + k2
        self.pe = p1 + p2
        self.Ls = self.ke - self.pe

        self.f1 = (sm.diff(sm.diff(self.Ls,self.theta1d),self.t) - sm.diff(self.Ls,self.theta1)).expand()
        self.f2 = (sm.diff(sm.diff(self.Ls,self.theta1d),self.t) - sm.diff(self.Ls,self.theta2)).expand()
        self.f3 = (sm.diff(sm.diff(self.Ls,self.phi1d),self.t) - sm.diff(self.Ls,self.phi1)).expand()
        self.f4 = (sm.diff(sm.diff(self.Ls,self.phi2d),self.t) - sm.diff(self.Ls,self.phi2)).expand()

        self.forces = [self.f1,self.f2,self.f3,self.f4]

    def inertia_matrix(self):
        rows = []
        for f in self.forces:
            columns = []
            for v in [self.theta1dd,self.theta2dd,self.phi1dd,self.phi2dd]:
                columns.append(f.coeff(v))
            rows.append(columns)
        M = sm.Matrix(rows)
        return M

    def coriolis_matrix(self):
        #centrifugal / Coriolis force term,
        rows = []
        coordinates = list(combinations_with_replacement([self.theta1d,self.theta2d,self.phi1d,self.phi2d], 2))
        c_v = [sm.prod(i) for i in coordinates]
        for f in self.forces:
            columns = []
            for v in c_v:
                columns.append(f.coeff(v)*v)
            rows.append(sum(columns))
        self.C = sm.Matrix(rows)
        return self.C

    def gravity_matrix(self):
        # gravity term
        rows = []
        for f in self.forces:
            columns = []
            for v in [self.theta1dd,self.theta2dd,self.phi1dd,self.phi2dd]:
                columns.append(f.coeff(v)*v)
            rows.append(sum(columns))
        self.G = sm.Matrix(self.forces) - sm.Matrix(rows) - self.C
        return self.G