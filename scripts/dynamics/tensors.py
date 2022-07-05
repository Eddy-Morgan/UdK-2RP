from dynamics import MDynamics

class Mdtensor:
    def __init__(self):
        g = 9.81
        m = [1,1]
        l = [0.3, 0.25]
        tau = [0,0,0,0]
        r = [0.1,0.1]
        rho = 1023.6
        self.mdys = MDynamics(g,m,l,tau,r,rho)

    def mass_matrix(self):
        return self.mdys.inertia_dynamics_np()
    
    def coriolis_matrix(self):
        return self.mdys.coriolis_dynamics_np()
    
    def gravity_matrix(self):
        return self.mdys.gravity_dynamics_np()

    def added_mass_matrix(self):
        return self.mdys.added_mass()