from firedrake import *
import os

n = 32
# mesh = UnitSquareMesh(n, n)
mesh = PeriodicUnitSquareMesh(n,n)

# We choose degree 2 continuous Lagrange polynomials. We also need a
# piecewise linear space for output purposes::

V = VectorFunctionSpace(mesh, "CG", 2)
V_out = VectorFunctionSpace(mesh, "CG", 1)

# We also need solution functions for the current and the next
# timestep. Note that, since this is a nonlinear problem, we don't
# define trial functions::

u_ = Function(V, name="Velocity")
u = Function(V, name="VelocityNext")

v = TestFunction(V)

# For this problem we need an initial condition::

x = SpatialCoordinate(mesh)
scale = 0.1
freq = 3
ic = project(as_vector([scale*sin(freq*pi*x[0]), 0]), V)
bc = DirichletBC(V, as_vector([scale*sin(freq*pi*x[0]),0]), "on_boundary")

# We start with current value of u set to the initial condition, but we
# also use the initial condition as our starting guess for the next
# value of u::

u_.assign(ic)
u.assign(ic)

# :math:`\nu` is set to a (fairly arbitrary) small constant value::

nu = 0.0001

# The timestep is set to produce an advective Courant number of
# around 1. Since we are employing backward Euler, this is stricter than
# is required for stability, but ensures good temporal resolution of the
# system's evolution::

timestep = 1.0/n

# Here we finally get to define the residual of the equation. In the advection
# term we need to contract the test function :math:`v` with 
# :math:`(u\cdot\nabla)u`, which is the derivative of the velocity in the
# direction :math:`u`. This directional derivative can be written as
# ``dot(u,nabla_grad(u))`` since ``nabla_grad(u)[i,j]``:math:`=\partial_i u_j`.
# Note once again that for a nonlinear problem, there are no trial functions in
# the formulation. These will be created automatically when the residual
# is differentiated by the nonlinear solver::

F = (inner((u - u_)/timestep, v)
     + inner(dot(u,nabla_grad(u)), v) + nu*inner(grad(u), grad(v)))*dx


# Finally, we loop over the timesteps solving the equation each time and
# outputting each result::

t = 0.0
end = 1.0

# Output only supports visualisation of linear fields (either P1, or
# P1DG).  In this example we project to a linear space by hand.  Another
# option is to let the :class:`~.File` object manage the decimation.  It
# supports both interpolation to linears (the default) or projection (by
# passing ``project_output=True`` when creating the :class:`~.File`).
# Outputting data is carried out using the :meth:`~.File.write` method
# of :class:`~.File` objects::

output_dir_path = os.path.dirname(os.path.realpath(__file__))
outfile = File(output_dir_path + "/../data/burgers.pvd")
outfile.write(project(u, V_out, name="Velocity"), time=t)

while (t <= end):
#     solve(F == 0, u, bcs=bc)
    solve(F == 0, u)
    u_.assign(u)
    t += timestep
    outfile.write(project(u, V_out, name="Velocity"), time=t)
    print("time step t = ", t, " ....done")
