import numpy as np
from params import *

class MDynamics:
    def compute_mass_matrix(self,s):
        q1 = s[0]
        q2 = s[2]
        q3 = s[4]
        m11 = Iy2*(np.sin(q2))**2 + Iy3*(np.sin(q2+q3)**2) + Iz1 
        + Iz2*(np.cos(q2)**2) + Iz3*(np.cos(q2+q3)**2) + m2*(r1**2)*(np.cos(q2)**2) + m3*((l2*np.cos(q2) + r2*np.cos(q2+q3))**2)
        
        m22 = Ix2 + Ix3 + m3*l1**2 + m2*r1**2 + m3*r2**2 + 2*m3*l1*r2*np.cos(q3)
        m33 = Ix3 + m3*r2**2
        m23 = Ix3 + m3*r2**2 + m3*l1*r2*np.cos(q3)
        m32 = Ix3 + m3*r2**2 + m3*l1*r2*np.cos(q3)
        m12 = 0
        m21 = 0
        m13 = 0
        m31 = 0
        return np.array([[m11, m12, m13],[m21, m22, m23],[m31, m32, m33]])
    
    def compute_coriolis_matrix(self,s):
        q1 = s[0]
        q2 = s[2]
        q3 = s[4]
        q1_dot = s[1]
        q2_dot = s[3]
        q3_dot = s[5]



        c11 = (m3*(l1*np.cos(q2) + r2*np.cos(q2+q3))*(l1*np.sin(q2) + r2*np.sin(q2+q3)))*q2_dot - ((Iy2 - Iz2 - m2*r1**2)*np.cos(q2)*np.sin(q2) + (Iy3-Iz3)*np.cos(q2+
            q3)*np.sin(q2+q3))*q2_dot + (m3*r2*np.sin(q2+q3)*(l1*np.cos(q2) + r2*np.cos(q2+q3)))*q3_dot - ((Iy3-Iz3)*np.cos(q2+q3)*np.sin(q2+q3))*q3_dot
        
        c12 = (m3*(l1*np.cos(q2)+r2*np.cos(q2+q3))*(l1*np.sin(q2)+r2*np.sin(q2+q3)))*q1_dot - ((Iy2-Iz2-m2*r1**2)*np.cos(q2)
        *np.sin(q2)+(Iy3-Iz3)*np.cos(q2+q3)*np.sin(q2+q3))*q1_dot

        c13 = (m3*r2*np.sin(q2+q3)*(l1*np.cos(q2)+r2*np.cos(q2+q3)))*q1_dot - ((Iy3-Iz3)*np.cos(q2+q3)*np.sin(q2+q3))*q1_dot
        c21 = (-(Iz2-Iy2+m2*r1**2)*np.cos(q2)*np.sin(q2)-(Iz3-Iy3)*np.cos(q2+q3)*np.sin(q2+q3))*q1_dot - (m3*(l1*np.cos(q2)+r2*np.cos(q2+q3))*(l1*np.sin(q2)+r2*np.sin(q2+q3)))*q1_dot
        c22 = q3_dot*l1*m3*r2*np.sin(q3)
        c23 = q3_dot*l1*m3*r2*np.sin(q3) - q2_dot*l1*m3*r2*np.sin(q3)
        c31 = ((Iz3-Iy3)*np.cos(q2+q3)*np.sin(q2+q3))*q1_dot + (m3*r2*np.sin(q2+q3)*(l1*np.cos(q2)+r2*np.cos(q2+q3)))*q1_dot
        c32 = -l1*m3*r2*q2_dot*np.sin(q3)
        c33 = 0
        return np.array([[c11, c12, c13],[c21, c22, c23],[c31, c32, c33]])


    def compute_gravity_matrix(self,s):
        q1 = s[0]
        q2 = s[2]
        q3 = s[4]
        g11 = 0
        g21 = (m2*g*r1 + m3*g*l1)*np.cos(q3) + m3*r2*np.cos(q2+q3)
        g31 = m3*g*r2*np.cos(q2+q3)
        return np.array([g11,g21,g31])