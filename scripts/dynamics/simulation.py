import numpy as np
from scipy.integrate import odeint, solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from manipulator import MDynamics
from constraints import spatial_eclipsoid_Constraints
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
    

mdy = MDynamics()
ct = spatial_eclipsoid_Constraints()
print('templates generation complete')

def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)

def compute_unconstrained(C,G):
    return -C -G

def compute_constrained(A,b,M,Q):
    #print('computing_constrained_force')
    #print(f'is positive definite ; {is_pos_def(M)}')
    m_sqrt = sqrtm(M)
    m_inv_sqrt = np.linalg.inv(m_sqrt)

    AM=np.dot(A,m_inv_sqrt)
    pinv_AM = np.linalg.pinv(AM)
    w = b - np.dot(A,np.dot(np.linalg.inv(M),Q))
    u = np.dot(pinv_AM, w)
    return np.dot(m_inv_sqrt,u)

def dSdt(t,s):
    print(s)
    M = mdy.compute_mass_matrix(s)
    C = mdy.compute_coriolis_matrix(s)
    G = mdy.compute_gravity_matrix(s)
    Q = compute_unconstrained(C,G)
    A = ct.compute_A(s,t)
    b = ct.compute_b(s,t)
    Qc = compute_constrained(A,b,M,Q)
    q_dot_dot = np.dot(np.linalg.inv(M),Q)+Qc

    return [
        s[1],
        q_dot_dot[0],
        s[3],
        q_dot_dot[1],
        s[5],
        q_dot_dot[2],
    ]


S_0 = [-13.5,0.75,50,-0.24,-100,-0.3] # Initial state of the system

t = np.linspace(0,15,1000)
sol = odeint(dSdt,y0=S_0, t=t, tfirst=True)
q1 = sol.T[0]
q1d = sol.T[1]
q2 = sol.T[2]
q2d = sol.T[3]
q3 = sol.T[4]
q3d = sol.T[5]

plt.close()

# Initialise the subplot function using number of rows and columns
figure, axis = plt.subplots(3, 2)
  

axis[0, 0].plot(t, q1)
axis[0, 0].set_title("q1")
  
axis[0, 1].plot(t, q1d)
axis[0, 1].set_title("q1d")
  
axis[1, 0].plot(t, q2)
axis[1, 0].set_title("q2")
  
axis[1, 1].plot(t, q2d)
axis[1, 1].set_title("q2d")

axis[2, 0].plot(t, q3)
axis[2, 0].set_title("q3")
  
axis[2, 1].plot(t, q3d)
axis[2, 1].set_title("q3d")
  
# Combine all the operations and display
plt.show()