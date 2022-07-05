import sympy as sm
from sympy import  Matrix, Symbol, cos, Rational,lambdify,diag, pi
from sympy.physics.mechanics import dynamicsymbols
from itertools import combinations_with_replacement
import numpy as np

class MDynamics:
    def __init__(self, g, m,l,tau,r, rho):
        self.g_v = g
        self.m1_v = m[0]
        self.m2_v = m[1]
        self.l1_v = l[0]
        self.l2_v = l[1]
        self.tau1_v = tau[0]
        self.tau2_v = tau[1]
        self.tau3_v = tau[2]
        self.tau4_v = tau[3]
        self.r1_v = r[0]
        self.r2_v = r[1]
        self.rho = rho

        # theta - yaw
        # phi - pitch
        # roll is neglected

        self.m1, self.m2, self.l1, self.l2, self.g ,self.tau1, self.tau2,self.tau3,self.tau4, self.r1,self.r2 = sm.symbols('m1 m2 l1 l2 g tau1 tau2 tau3 tau4 r1 r2',real =True)

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

        self.x1_f = sm.lambdify((self.theta1, self.theta2, self.phi1, self.phi2, self.l1, self.l2), self.x1_)
        self.y1_f = sm.lambdify((self.theta1, self.theta2, self.phi1, self.phi2, self.l1, self.l2), self.y1_)
        self.z1_f = sm.lambdify((self.theta1, self.theta2, self.phi1, self.phi2, self.l1, self.l2), self.z1_)
        self.x2_f = sm.lambdify((self.theta1, self.theta2, self.phi1, self.phi2, self.l1, self.l2), self.x2_)
        self.y2_f = sm.lambdify((self.theta1, self.theta2, self.phi1, self.phi2, self.l1, self.l2), self.y2_)
        self.z2_f = sm.lambdify((self.theta1, self.theta2, self.phi1, self.phi2, self.l1, self.l2), self.z2_)

        x2_dot = sm.diff(self.x2_,self.t)
        y2_dot = sm.diff(self.y2_,self.t)
        z2_dot = sm.diff(self.z2_,self.t)

        I1 = Rational('1/12')*self.m1*self.l1**2 + Rational('1/4')*self.m1*self.r1**2
        v1 = Rational('0.5')*self.l1*self.theta1d
        w1 = self.theta1d
        k1 = Rational('1/2')*self.m1*v1**2 + Rational('1/2')*I1*w1**2
        p1 = self.m1*self.g*self.z1_

        v2 = x2_dot**2 + y2_dot**2 + z2_dot**2
        w2 = self.theta1d + self.theta2d
        I2 = Rational('1/12')*self.m2*self.l2**2 + Rational('1/4')*self.m2*self.r2**2
        k2 = Rational('1/2')*self.m2*v2**2 + Rational('1/2')*I2*w2**2
        p2 = self.m2*self.g*self.z2_

        ke = k1 + k2
        pe = p1 + p2
        self.Ls = ke - pe

        self.f1 = (sm.diff(sm.diff(self.Ls,self.theta1d),self.t) - sm.diff(self.Ls,self.theta1)).expand()
        self.f2 = (sm.diff(sm.diff(self.Ls,self.theta1d),self.t) - sm.diff(self.Ls,self.theta2)).expand()
        self.f3 = (sm.diff(sm.diff(self.Ls,self.phi1d),self.t) - sm.diff(self.Ls,self.phi1)).expand()
        self.f4 = (sm.diff(sm.diff(self.Ls,self.phi2d),self.t) - sm.diff(self.Ls,self.phi2)).expand()

        self.forces = [self.f1,self.f2,self.f3,self.f4]
        self.args = (self.theta1, self.theta2, self.phi1, self.phi2 ,self.theta1d ,self.theta2d,self.phi1d ,self.phi2d)

    def inertia_dynamics_np(self):
        rows = []
        for f in self.forces:
            columns = []
            for v in [self.theta1dd,self.theta2dd,self.phi1dd,self.phi2dd]:
                columns.append(f.coeff(v))
            rows.append(columns)
        M = sm.Matrix(rows)
        return lambdify(self.args , M.subs({self.g: self.g_v ,self.m1: self.m1_v ,
            self.m2:self.m2_v, self.l1:self.l1_v, self.l2:self.l2_v,
            self.tau1:self.tau1_v, self.tau2:self.tau2_v,self.tau3:self.tau3_v,
            self.tau4:self.tau4_v, self.r1:self.r1_v,self.r2:self.r2_v}),modules='numpy')

    def coriolis_dynamics_np(self):
        #centrifugal / Coriolis force term,
        rows = []
        coordinates = list(combinations_with_replacement([self.theta1d,self.theta2d,self.phi1d,self.phi2d], 2))
        c_v = [sm.prod(i) for i in coordinates]
        for f in self.forces:
            columns = []
            for v in c_v:
                columns.append(f.coeff(v)*v)
            rows.append(sum(columns))
        C = sm.Matrix(rows)
        return lambdify(self.args ,C.subs({self.g: self.g_v ,self.m1: self.m1_v ,
            self.m2:self.m2_v, self.l1:self.l1_v, self.l2:self.l2_v,
            self.tau1:self.tau1_v, self.tau2:self.tau2_v,self.tau3:self.tau3_v,
            self.tau4:self.tau4_v, self.r1:self.r1_v,self.r2:self.r2_v}),modules='numpy')

    def coriolis_dynamics(self):
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

    def gravity_dynamics_np(self):
        # gravity term
        rows = []
        for f in self.forces:
            columns = []
            for v in [self.theta1dd,self.theta2dd,self.phi1dd,self.phi2dd]:
                columns.append(f.coeff(v)*v)
            rows.append(sum(columns))
        cd = self.coriolis_dynamics()
        G = sm.Matrix(self.forces) - sm.Matrix(rows) - cd
        return lambdify(self.args ,G.subs({self.g: self.g_v ,self.m1: self.m1_v ,
            self.m2:self.m2_v, self.l1:self.l1_v, self.l2:self.l2_v,
            self.tau1:self.tau1_v, self.tau2:self.tau2_v,self.tau3:self.tau3_v,
            self.tau4:self.tau4_v, self.r1:self.r1_v,self.r2:self.r2_v}),modules='numpy')

    def added_mass(self):
        #By approximating the manipulator as slow moving and symmetric
        theta1_ma = self.rho*np.pi*self.r1_v**2*self.l1_v**3*1/12 #for manipulator 1 yaw
        theta2_ma = self.rho*np.pi*self.r2_v**2*self.l2_v**3*1/12 #for manipulator 2 yaw
        phi1_ma = self.rho*np.pi*self.r1_v**2*self.l1_v**3*1/12 #for manipulator 2 pitch
        phi2_ma = self.rho*np.pi*self.r2_v**2*self.l2_v**3*1/12 #for manipulator 2 pitch
        arr = np.array([theta1_ma,theta2_ma,phi1_ma,phi2_ma])
        ImA = np.diag(arr)
        return ImA


    def drag_force(self):
        pass