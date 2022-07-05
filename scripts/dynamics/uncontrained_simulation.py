from tensors import Mdtensor
import sympy as smp
import numpy as np
from scipy.integrate import odeint

print('starting')
mdt = Mdtensor()

Ma = mdt.added_mass_matrix()

print(Ma)