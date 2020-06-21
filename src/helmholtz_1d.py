from firedrake import *
from firedrake.utility_meshes import UnitIntervalMesh
from firedrake.norms import errornorm
from firedrake.plot import plot

import numpy as np

import os

try:
    import matplotlib.pyplot as plt
except:
    warning("Matplotlib not imported")


class Helmholtz:
    
    def __init__(self, n_cycles):
        self.__fe_degree = 1
        self.__n_cycles = n_cycles
        
        self.__n_cell = np.array([4])
        self.__h = np.array([1/self.__n_cell[-1]])
        
        self.__error_l2 = np.array([])
        self.__error_energy = np.array([])
        self.__rate_l2 = np.array([])
        self.__rate_energy = np.array([])
        
        self.setup_plot()
    
    def reinit(self, n):
        self.setup_mesh(n)
        self.setup_space()
        self.setup_data()
        self.setup_form()
        self.setup_exact_solution()
        self.setup_bc()

        
    def setup_mesh(self, n):
        self.mesh = UnitIntervalMesh(n)

                
    def setup_space(self):
        self.V = FunctionSpace(self.mesh, "CG", self.__fe_degree)

        
    def setup_data(self):                
        self.x = SpatialCoordinate(self.mesh)
        self.f = Function(self.V)
        self.f.interpolate((1 + 4*pi*pi)*cos(self.x[0]*pi*2)*cos(self.y[0]*pi*2))


    def setup_form(self):
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        
        self.a = (dot(grad(self.v), grad(self.u)) + self.v * self.u) * dx
        self.L = self.f * self.v * dx


    def setup_exact_solution(self):
        self.u_exact_expr = cos(self.x[0]*pi*2)*cos(self.y[0]*pi*2)
                
        
    def setup_bc(self):   
        # Take the boundary condition from the exact solution     
        self.bc = DirichletBC(self.V, self.u_exact_expr, "on_boundary")

        
    def setup_plot(self):        
        try:
            # Create figure
            self.__fig, self.__axes = plt.subplots(1, 1, figsize=(12,9))
            
            # set alias for axes and create second plot
            self.__axes.set_xlabel("x")
            self.__axes.set_ylabel("u")
            self.__axes.set_title("solution")
        except Exception as e:
            warning("Cannot plot figure. Error msg: '%s'" % e)

        
    def solve_helmholtz(self):
        self.u = Function(self.V)            
        solve(self.a == self.L, self.u, bcs=self.bc, solver_parameters={"ksp_type": "cg"})

        
    def write_solution(self, cycle):
        output_dir_path = os.path.dirname(os.path.realpath(__file__))
        File(output_dir_path + "/../data/helmholtz_1d_cycle-" + str(cycle) + ".pvd").write(self.u)


    def plot_solution(self, cycle):       
        try:
            #
            # Note that the plots are not optimal for higher order polynomials 
            # since Firedrake's plot returns a PathPatch object and not a Line2D object
            # such as matplotlob.plot 
            #
            plot(self.u, axes=self.__axes, label="h =  " + str(self.__h[cycle]), color=np.random.rand(3,))
            if (cycle == self.__n_cycles-1):
                plt_exact = plot(Function(self.V).interpolate(self.u_exact_expr), axes=self.__axes, label="exact", color="red", linewidth=2)
        except Exception as e:
            warning("Cannot plot figure. Error msg: '%s'" % e)

            
    def compute_error(self, cycle):
        error_l2 = errornorm(self.u_exact_expr, self.u, "L2", degree_rise=self.__fe_degree+1)
        error_energy = errornorm(self.u_exact_expr, self.u, "H1", degree_rise=self.__fe_degree+1)
        
        self.__error_l2 = np.append(self.__error_l2, error_l2)
        self.__error_energy = np.append(self.__error_energy, error_energy)

        if (cycle > 0):    
            self.__rate_l2 = np.append(self.__rate_l2, np.log2(self.__error_l2[-2]/error_l2))
            self.__rate_energy = np.append(self.__rate_energy, np.log2(self.__error_energy[-2]/error_energy))
    
    
    def plot_error(self):
        try:
            self.__error_fig, self.__error_axes = plt.subplots(1, 1, figsize=(8,8))
             
            self.__error_axes.set_xlabel("h")
            self.__error_axes.set_ylabel("error")
            self.__error_axes.set_title("convergence plots")
            
            plt.loglog(self.__h, self.__error_l2, "-*b", label="L2")
            
            # This curve is just to compare the path
            # that the error would take if the order
            # of convergence was exactly the fe_degree*2 
            plt.loglog(self.__h, 
                       0.9 * np.power(self.__h,
                                     self.__fe_degree+1) * self.__error_l2[0]/np.power(self.__h[0],
                                                                                       self.__fe_degree+1),
                                     "--b", label = "order {0}".format(self.__fe_degree+1))
            
            plt.loglog(self.__h, self.__error_energy, "-*r", label="energy norm")
            
            # This curve is just to compare the path
            # that the error would take if the order
            # of convergence was exactly the fe_degree
            plt.loglog(self.__h, 
                       0.9 * np.power(self.__h,
                                   self.__fe_degree) * self.__error_energy[0]/np.power(self.__h[0], 
                                                                                       self.__fe_degree),
                                   "--r", label = "order {0}".format(self.__fe_degree))
            
        except Exception as e:
            warning("Cannot plot figure. Error msg: '%s'" % e)


    def run(self):
        for cycle in range(0,self.__n_cycles):
            
            n_cell = self.__n_cell[-1]
            h = self.__h[-1]
            print("------------------------------------------")
            print("Cycle                  :   ", cycle)
            print("n_cells                :   ", n_cell)
            print("h                      :   ", h)
            
            
            self.reinit(n_cell)
            self.solve_helmholtz()
            self.compute_error(cycle)
            self.write_solution(cycle)
            self.plot_solution(cycle)
            
            print("error L2               :   ", self.__error_l2[-1])
            print("error energy           :   ", self.__error_energy[-1])
            if (cycle > 0):
                print("convergence rate L2    :   ", self.__rate_l2[-1])
                print("convergence rate energy:   ", self.__rate_energy[-1])
            print("------------------------------------------\n")
            
            # The next cycle's n_cell and h
            self.__n_cell = np.append(self.__n_cell, 2*self.__n_cell[-1])
            self.__h = np.append(self.__h, self.__h[-1]/2)
            
        # We added one n_cell and one h too much, so delete
        self.__n_cell = np.delete(self.__n_cell, -1)
        self.__h = np.delete(self.__h, -1)
        
        self.plot_error()
        
        # Now add legends and show figures
        try:
            self.__axes.legend()
            self.__error_axes.legend()
            plt.show()
        except Exception as e:
            warning("Cannot show figure. Error msg '%s'" % e)
        
            
if __name__ == '__main__':
        n_cycles = 5
    
        problem = Helmholtz(n_cycles)
        problem.run()
