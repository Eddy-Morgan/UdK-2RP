import sympy as sm
from sympy import Symbol,lambdify
from sympy.physics.mechanics import dynamicsymbols
from params import *


class spatial_eclipsoid_Constraints:
    def __init__(self) -> None:
        self.l0,self.l1,self.l2 = sm.symbols('l0 l1 l2', real =True)
        t = Symbol('t')
        self.q1, self.q2, self.q3 = dynamicsymbols('q1 q2 q3')
        self.q1d = dynamicsymbols('q1',1)
        self.q2d = dynamicsymbols('q2',1)
        self.q3d = dynamicsymbols('q3',1)
        self.q1dd = dynamicsymbols('q1',2)
        self.q2dd = dynamicsymbols('q2',2)
        self.q3dd = dynamicsymbols('q3',2)
    
        x = (l1*sm.cos(self.q2)+l2*sm.cos(self.q2+self.q3))*sm.cos(self.q1)
        y = (l1*sm.cos(self.q2)+l2*sm.cos(self.q2+self.q3))*sm.sin(self.q1)
        z = l0 + l1*sm.sin(self.q2) + l2*sm.sin(self.q2+self.q3)

        endefX = -x + (1 + 0.25*sm.cos(t))
        endefY = -y + (-0.3*sm.cos(t)+sm.sin(t))
        endefZ = -z + (1+0.5*sm.sin(t))

        endefXdd = sm.diff(sm.diff(endefX,t),t).expand()
        endefYdd = sm.diff(sm.diff(endefY,t),t).expand()
        endefZdd = sm.diff(sm.diff(endefZ,t),t).expand()

        A11 = endefXdd.coeff(self.q1dd)
        A12 = endefXdd.coeff(self.q2dd)
        A13 = endefXdd.coeff(self.q3dd)

        A21 = endefYdd.coeff(self.q1dd)
        A22 = endefYdd.coeff(self.q2dd)
        A23 = endefYdd.coeff(self.q3dd)

        A31 = endefZdd.coeff(self.q1dd)
        A32 = endefZdd.coeff(self.q2dd)
        A33 = endefZdd.coeff(self.q3dd)

        A = sm.Matrix([[A11, A12, A13], [A21, A22, A23], [A31, A32, A33]])

        b1 = endefXdd - A11*self.q1dd - A12*self.q2dd - A13*self.q3dd
        b2 = endefYdd - A21*self.q1dd - A22*self.q2dd - A23*self.q3dd
        b3 = endefZdd - A31*self.q1dd - A32*self.q2dd - A33*self.q3dd
        b = sm.Matrix([b1.simplify(),b2.simplify(),b3.simplify()])

        self.A_f=lambdify([t,self.l0,self.l1,self.l2, self.q1,self.q2,self.q3, self.q1d, self.q2d, self.q3d],A, "numpy")
        self.b_f=lambdify([t,self.l0,self.l1,self.l2, self.q1,self.q2,self.q3, self.q1d, self.q2d, self.q3d],b, "numpy")

    def compute_A(self,s,t):
        q1v = s[0]
        q2v = s[2]
        q3v = s[4]
        q1_dotv = s[1]
        q2_dotv = s[3]
        q3_dotv = s[5]
        A = self.A_f(t,l0,l1,l2,q1v,q2v,q3v,q1_dotv,q2_dotv,q3_dotv)
        return A

    def compute_b(self,s,t):
        q1v = s[0]
        q2v = s[2]
        q3v = s[4]
        q1_dotv = s[1]
        q2_dotv = s[3]
        q3_dotv = s[5]
        b = self.b_f(t,l0,l1,l2,q1v,q2v,q3v,q1_dotv,q2_dotv,q3_dotv)
        return b