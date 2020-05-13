from firedrake import *

import os


## Class to solve mixed Poisson problem.
#
# This class solves the mixed Poisson equation 
# \f$ \nabla\cdot \sigma = f \f$, \f$ \sigma+\nabla u = 0 \f$
# using BDM-DG elements.
class PoissonMixed:
    
    ##Contructor of PoissonMixed class.
    #
    # Setup all necessary stuff to solve the problem.
    #
    # @param[in] n integer grid parameter x-direction
    # @param[in] m integer grid parameter y-direction
    def __init__(self, n, m):
        self.setup_mesh(n, m)
        self.setup_space()
        self.setup_data()
        self.setup_form()
        self.setup_bc()
        
    ## Setup mesh for FEM computation.
    #
    # Setup unit square mesh with nxm elements.
    #
    # @param[in] n integer grid parameter x-direction
    # @param[in] y integer grid parameter y-direction
    def setup_mesh(self, n, m):
        self.mesh = UnitSquareMesh(n, m)
        
    ## Setup stable pair of spaces. 
    #        
    # Setup a stable pair of function spaces for the problem: 
    # a combination of order \f$k\f$ Brezzi-Douglas-Marini (BDM)
    # elements and order \f$k - 1\f$ discontinuous Galerkin
    # elements (DG). We use \f$k = 1\f$ and combine the BDM and
    # DG spaces into a mixed function space ``W``.        
    def setup_space(self):
        self.BDM = FunctionSpace(self.mesh, "BDM", 1)
        self.DG = FunctionSpace(self.mesh, "DG", 0)
        self.W = self.BDM * self.DG
        
    ## Declare source function 
    #
    # Declare ``f`` over the DG space and initialise
    # it with chosen right hand side function value.
    def setup_data(self):                
        self.x, self.y = SpatialCoordinate(self.mesh)
        self.f = Function(self.DG).interpolate(
                10*exp(-(pow(self.x - 0.5, 2) + pow(self.y - 0.5, 2)) / 0.02))

    ## Form and function setup.
    #
    # Define test and trial functions on the subspaces of the mixed function
    # spaces. Then define the variational forms.
    def setup_form(self):
        self.sigma, self.u = TrialFunctions(self.W)
        self.tau, self.v = TestFunctions(self.W)
        
        self.a = (dot(self.sigma, self.tau) + div(self.tau)*self.u + div(self.sigma)*self.v)*dx
        self.L = - self.f*self.v*dx
        
    ## Setup boundary conditions.
    #
    # The strongly enforced boundary conditions on the BDM space on the top and
    # bottom of the domain are declared.
    # Note that it is necessary to apply these boundary conditions to the first
    # subspace of the mixed function space using ``W.sub(0)``. This way the
    # association with the mixed space is preserved. Declaring it on the BDM space
    # directly is *not* the same and would in fact cause the application of the
    # boundary condition during the later solve to fail.
    def setup_bc(self):                
        self.bc0 = DirichletBC(self.W.sub(0), as_vector([0.0, -sin(5*self.x)]), 1)
        self.bc1 = DirichletBC(self.W.sub(0), as_vector([0.0, sin(5*self.y)]), 2)
        
        self.w = Function(self.W)
        
    ## Call the solver.
    #
    # Then we solve the linear variational problem ``a == L`` for ``w`` under the
    # given boundary conditions ``bc0`` and ``bc1``. Afterwards we extract the
    # components ``sigma`` and ``u`` on each of the subspaces with ``split``.
    def solve_Poisson(self):                
        solve(self.a == self.L, self.w, bcs=[self.bc0, self.bc1])
        self.sigma, self.u = self.w.split()
        
    ## Write solution as *.vtk
    #
    # Lastly we write the component of the solution corresponding to the primal
    # variable on the DG space to a file in VTK format for later inspection with a
    # visualisation tool such as `ParaView <http://www.paraview.org/>`.
    def write_solution(self):
        output_dir_path = os.path.dirname(os.path.realpath(__file__))
        File(output_dir_path + "/../data/poisson_mixed_2d_u.pvd").write(self.u)
        File(output_dir_path + "/../data/poisson_mixed_2d_sigma.pvd").write(self.sigma)

    ## Plot solution.
    #        
    # We could use the built in plot function
    # of firedrake by calling :func:`plot <firedrake.plot.plot>`
    # to plot a surface graph. Before that,
    # matplotlib.pyplot should be installed and imported.
    def plot_solution(self):
        try:
            import matplotlib.pyplot as plt
        except:
            warning("Matplotlib not imported")
        
        try:
            # Create figure
            fig, axes = plt.subplots(1, 2, figsize=(16,8))
            
            # set alias for axes and create first plot
            ax = axes[0]
            plt_sigma = quiver(self.sigma, axes=ax, scale=3.0, cmap="inferno")
            fig.colorbar(plt_sigma, ax=axes[0])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("sigma")
            
            # set alias for axes and create second plot
            ax = axes[1]
            plt_u = tripcolor(self.u, axes=ax, cmap="summer")
            fig.colorbar(plt_u, ax=ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("u")
        except Exception as e:
            warning("Cannot plot figure. Error msg: '%s'" % e)
        
        try:
            plt.show()
        except Exception as e:
            warning("Cannot show figure. Error msg '%s'" % e)


if __name__ == '__main__':
    problem = PoissonMixed(32,32)
    problem.solve_Poisson()
    problem.write_solution()
    problem.plot_solution()
