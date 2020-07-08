from firedrake import *
from firedrake.norms import errornorm

import os 


## Class to solve an Elasticity problem of a bending rod.
#
# This class solves the Elasticity problem \f$-\nabla\cdot \sigma = f\f$ for a long 
# rod under gravity load with 
# \f$P_k\f$-finite elements in each component. 
class ElasticRod:
    
    ##Contructor of ElasticRod class.
    #
    # Setup all necessary stuff to solve the problem.
    #
    # @param[in] nx integer grid parameter x-direction
    # @param[in] ny integer grid parameter y-direction
    # @param[in] nz integer grid parameter z-direction
    # @param[in] lx length x-direction
    # @param[in] ly length y-direction
    # @param[in] lz length z-direction
    def __init__(self, nx, ny, nz, lx=10, ly=1, lz=1):
        self.setup_mesh(nx, ny, nz, lx, ly, lz)
        self.setup_space()
        self.setup_data()
        self.setup_form()
        self.setup_bc()
        
    ## Setup mesh for FEM computation.
    #
    # Setup unit square mesh with nx x ny x nz elements.
    #
    # @param[in] nx integer grid parameter x-direction
    # @param[in] ny integer grid parameter y-direction
    # @param[in] nz integer grid parameter z-direction
    # @param[in] lx length x-direction
    # @param[in] ly length y-direction
    # @param[in] lz length z-direction
    def setup_mesh(self, nx, ny, nz, lx, ly, lz):
        self.mesh = BoxMesh(nx, ny, nz, lx, ly, lz)
        
    ## Setup FEf spaces. 
    #        
    # Setup a conformal function space for the problem. We use simple Lagrange elements. 
    # function space ``V``.        
    def setup_space(self):
        self.V = VectorFunctionSpace(self.mesh, "CG", 1)
        
    ## Declare source function 
    #
    # Declare ``f`` over the space V and initialise
    # it with chosen right hand side function value.
    def setup_data(self):
        self.__rho = Constant(2710)
        self.__g = Constant(9.81)
        self.__mu = Constant(2.57e10)
        self.__lambda = Constant(5.46e10)
        
        self.f = as_vector([0, 0, -self.__rho * self.__g])
        self.__Id = Identity(self.mesh.geometric_dimension()) # Identity tensor

    ## Strain tensor
    def epsilon(self, u):
        return 0.5*(grad(u) + grad(u).T)

    ## Hooke's law
    def sigma(self, u):
        return self.__lambda*div(u)*self.__Id + 2*self.__mu*self.epsilon(u)

    ## Form and function setup.
    #
    # Define test and trial functions on the subspace of the function
    # space. Then define the variational forms.
    def setup_form(self):
        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)
        
        self.a = inner(self.sigma(self.u), self.epsilon(self.v)) * dx
        self.L = dot(self.f, self.v) * dx
           
    ## Setup boundary conditions.
    #
    # The strongly enforced boundary conditions is enforced on the entire boundary.
    def setup_bc(self):
        # clamped at x=0
        self.bc = DirichletBC(self.V, Constant([0, 0, 0]), 1)
        #self.bc = DirichletBC(self.V, Constant([0, 0, 0]), 1)
        
    ## Call the solver.
    #
    # Then we solve the linear variational problem ``a == L``
    # and advice PETSc to use a conjugate gradient method.
    def solve(self, options=None, **kwargs):
        # create rigid body modes
#         x, y, z = SpatialCoordinate(self.mesh)
#         b0 = Function(V)
#         b1 = Function(V)
#         b2 = Function(V)
#         b3 = Function(V)
#         b4 = Function(V)
#         b5 = Function(V)
#         b0.interpolate(Constant([1, 0]))
#         b1.interpolate(Constant([0, 1]))
#         b2.interpolate(Constant([0, 1]))
#         b3.interpolate(as_vector([-y, x]))
#         b4.interpolate(as_vector([-y, x]))
#         b5.interpolate(as_vector([-y, x]))
#         nullmodes = VectorSpaceBasis([b0, b1, b2])
#         # Make sure they're orthonormal.
#         nullmodes.orthonormalize()
        
        self.uh = Function(self.V)            
        solve(self.a == self.L,
              self.uh,
              bcs=self.bc, 
              solver_parameters=options,
              **kwargs)
        
    ## Write solution as *.vtk
    #
    # Lastly we write the component of the solution corresponding to the primal
    # variable on the DG space to a file in VTK format for later inspection with a
    # visualisation tool such as `ParaView <http://www.paraview.org/>`.
    def write_solution(self):
        output_dir_path = os.path.dirname(os.path.realpath(__file__))
        File(output_dir_path + "/../data/elastic_rod_3d.pvd").write(self.uh)


if __name__ == '__main__':
    problem = ElasticRod(100, 10, 10)
    problem.solve(options={"ksp_type": "cg", 
                            "ksp_max_it": 100, 
                            "pc_type": "gamg",
                            "mat_type": "aij",
                            "ksp_monitor": None})
    problem.write_solution()
    print("...done.")
