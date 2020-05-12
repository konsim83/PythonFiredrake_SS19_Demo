from firedrake import *
from firedrake.utility_meshes import UnitIntervalMesh
from firedrake.plot import plot

import numpy as np
from networkx.algorithms.bipartite.basic import color
from mpmath import degree
from firedrake.norms import errornorm


class Helmholtz:
    
    def __init__(self, n):
        self.setup_mesh(n)
        self.setup_space()
        self.setup_data()
        self.setup_form()
        self.setup_bc()
        
    def setup_mesh(self, n):
        self.mesh = UnitIntervalMesh(n)
                
    def setup_space(self):
        self.V = FunctionSpace(self.mesh, "CG", 1)
        
    def setup_data(self):                
        self.x = SpatialCoordinate(self.mesh)
        self.f = Function(self.V)
        self.f.interpolate((1+4*pi*pi)*cos(self.x[0]*pi*2))

    def setup_form(self):
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        
        self.a = (dot(grad(self.v), grad(self.u)) + self.v * self.u) * dx
        self.L = self.f * self.v * dx
        
    def setup_bc(self):
        self.u_exact = Function(self.V)
        self.u_exact.interpolate(cos(self.x[0]*pi*2))
        self.bc = DirichletBC(self.V, self.u_exact, "on_boundary")
        
    def solve_helmholtz(self):
        self.u = Function(self.V)            
        solve(self.a == self.L, self.u, bcs=self.bc, solver_parameters={"ksp_type": "cg"})
        
    def write_solution(self):                
        File("../data/helmholtz_u.pvd").write(self.u)

    def plot_solution(self):
        try:
            import matplotlib.pyplot as plt
        except:
            warning("Matplotlib not imported")
        
        try:
            # Create figure
            fig, axes = plt.subplots(1, 1, figsize=(12,12))
            
            # set alias for axes and create second plot
            ax = axes
            plt_u = plot(self.u, axes=ax, color="b")
            ax.set_xlabel("x")
            ax.set_ylabel("u")
            ax.set_title("u")
        except Exception as e:
            warning("Cannot plot figure. Error msg: '%s'" % e)
        
        try:
            plt.show()
        except Exception as e:
            warning("Cannot show figure. Error msg '%s'" % e)
            
    def plot_error(self):
#         error_l2 = sqrt(assemble(dot(self.u - self.u_exact, self.u - self.u_exact) * dx))         
#         error_h1 = sqrt(error_l2**2 + error_energy**2)

        error_l2 = errornorm(self.u_exact, self.u, "L2")        
        error_h1 = errornorm(self.u_exact, self.u, "H1")
        
        return [error_l2, error_h1]

            
if __name__ == '__main__':
    errors = np.array([])
    rate_l2 = np.array([-1])
    rate_h1 = np.array([-1])
    
    for i in range(2,8):
        problem = Helmholtz(2**i)
        problem.solve_helmholtz()
        problem.write_solution()
        #problem.plot_solution()
        errors = np.append(errors, problem.plot_error())
        
    errors = errors.reshape(-1,2)
    
    for i in range(0,errors.shape[0]-1):
        rate_l2 = np.append(rate_l2, np.log2(errors[i,0]/errors[i+1,0]))
        rate_h1 = np.append(rate_h1, np.log2(errors[i,1]/errors[i+1,1]))
        
    print("errors L2: ", errors[:,0])
    print("rates L2:  ", rate_l2)
    
    print("errors H1: ", errors[:,1])
    print("rates H1:  ", rate_h1)
           
