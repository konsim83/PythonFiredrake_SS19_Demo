from firedrake import *

## Class to solve a Helmholtz problem with analytic solution.
#
# This class solves the Helmholtz \f$-\Delta u + u = f\f$ equation using on the unit square
# \f$P_k\f$-finite elements. For the given right hand side we have an exact solution to
# compare with.
class Helmholtz:
    
    ##Contructor of Helmholtz class.
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
    # Setup a conformal function space for the problem. We use simple Lagrange elements. 
    # function space ``V``.        
    def setup_space(self):
        self.V = FunctionSpace(self.mesh, "CG", 1)
        
    ## Declare source function 
    #
    # Declare ``f`` over the space V and initialise
    # it with chosen right hand side function value.
    def setup_data(self):                
        self.x, self.y = SpatialCoordinate(self.mesh)
        self.f = Function(self.V).interpolate((1+8*pi*pi)*cos(self.x*pi*2)*cos(self.y*pi*2))

    ## Form and function setup.
    #
    # Define test and trial functions on the subspace of the function
    # space. Then define the variational forms.
    def setup_form(self):
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        
        self.a = (dot(grad(self.v), grad(self.u)) + self.v * self.u) * dx
        self.L = self.f * self.v * dx
        
    ## Setup boundary conditions.
    #
    # The strongly enforced boundary conditions is enforced on the entire boundary.
    def setup_bc(self):
        self.u_exact = Function(self.V).interpolate(cos(self.x*pi*2)*cos(self.y*pi*2))
        self.bc = DirichletBC(self.V, self.u_exact, "on_boundary")
        
    ## Call the solver.
    #
    # Then we solve the linear variational problem ``a == L``
    # and advice PETSc to use a conjugate gradient method.
    def solve_helmholtz(self):
        self.u = Function(self.V)            
        solve(self.a == self.L, self.u, bcs=self.bc, solver_parameters={"ksp_type": "cg"})
        
    ## Write solution as *.vtk
    #
    # Lastly we write the component of the solution corresponding to the primal
    # variable on the DG space to a file in VTK format for later inspection with a
    # visualisation tool such as `ParaView <http://www.paraview.org/>`.
    def write_solution(self):                
        File("../data/helmholtz_u.pvd").write(self.u)

    ## Plot solution.
    #        
    # We could use the built-in plotting functions of 
    # firedrake by calling tripcolor to make a pseudo-color
    # plot. Before that, matplotlib.pyplot should be 
    # installed and imported:
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
            
    ## Compute error
    #
    # We compute and print the error of the solution in various
    # norms. Note hat the energy norm here is the \f$H_0^1\f$-norm.
    def plot_error(self):
        error_l2 = sqrt(assemble(dot(self.u - self.u_exact, self.u - self.u_exact) * dx))
        print("error L2 = ", error_l2)
        
        error_energy = sqrt(assemble(dot(grad(self.u) - grad(self.u_exact), grad(self.u) - grad(self.u_exact)) * dx))
        print("error energy norm = ", error_energy)
        
        error_h1 = sqrt(error_l2**2 + error_energy**2)
        print("error h1 = ", error_h1)


if __name__ == '__main__':
    problem = Helmholtz(16,16)
    problem.solve_helmholtz()
    problem.write_solution()
    problem.plot_solution()
    problem.plot_error()
