
from cmath import cos
import numpy as np
from params import *


class Constraints:
    def compute_spatial_eclipsoid_A_b(self,s,t):
        q1 = s[0]
        q2 = s[2]
        q3 = s[4]
        q1_dot = s[1]
        q2_dot = s[3]
        q3_dot = s[5]
        a11 = (l1*np.cos(q2)+l2*np.cos(q2+q3))*np.sin(q1)
        a12 = (l2*np.sin(q2+q3)+l1*np.sin(q2))*np.cos(q1)
        a13 = l2*np.sin(q2+q3)*np.cos(q1)
        a21 = -(l1*np.cos(q2)+l2*np.cos(q2+q1))*np.cos(q1)
        a22 = (l2*np.sin(q2+q3)+l1*np.sin(q2)*np.sin(q1))
        a23 = l2*np.sin(q2+q3)*np.sin(q1)
        a31 = 0
        a32 = l1*np.cos(q2) + l2*np.cos(q2+q3)
        a33 = l2*np.cos(q2+q3)

        b1 = q1_dot*np.sin(q1)*(l2*np.sin(q2+q3)*(q2_dot+q3_dot)+l1
        *q2_dot*np.sin(q2)) - np.cos(q1)*(l2*np.cos(q2+q3)*(q2_dot + q3_dot)**2 + l1*
        (q2_dot**2)*np.cos(q2))+q1_dot*np.sin(q1)*(l2*np.sin(q2+q3)*(q2_dot+q3_dot) 
        + q2_dot*l1*np.sin(q2)) - q1_dot**2*np.cos(q1)*(l1*np.cos(q2)+l2*np.cos(q2+q3)) - 0.25*np.cos(t)

        b2 = -np.sin(q1)*(l2*np.cos(q2+q3)*(q2_dot+q3_dot)**2 + 
        l1*q2_dot**2*np.cos(q2)) - q1_dot*np.cos(q1)*(l2*np.sin(q2+q3)*(q2_dot+q3_dot)+ l1*q2_dot*np.sin(q2)) 
        + q1_dot*np.cos(q1)*(l2*np.sin(q2+q3)*(q2_dot+q3_dot)+ l1*q2_dot*np.sin(q2)) 
        - q1_dot**2*np.sin(q1)*(l1*np.cos(q2)+l2*np.cos(q2+q3))
        + 0.3*np.cos(t) - np.sin(t)

        b3 = l1*q2_dot**2*q2_dot**2*np.sin(q2) + l2*(q2_dot+q3_dot)**2*np.sin(q2+q3) - 0.5*np.sin(t)

        A = np.array([[a11, a12, a13],[a21, a22, a23],[a31, a32, a33]])
        b = np.array([b1, b2, b3])

        return A,b