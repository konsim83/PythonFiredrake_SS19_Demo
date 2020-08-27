from firedrake import *
import os
from firedrake.utility_meshes import PeriodicUnitSquareMesh

n = 32
# mesh = UnitSquareMesh(n, n)
mesh = PeriodicUnitSquareMesh(n,n)

V = VectorFunctionSpace(mesh, "CG", 2)
V_out = VectorFunctionSpace(mesh, "CG", 1)

W = MixedFunctionSpace((V, V))
W_out = MixedFunctionSpace((V_out, V_out))

w_ = Function(W)
u_, v_ = w_.split()

#################################
x, y = SpatialCoordinate(mesh)
scale = 0.1
freq = 3
ic_u = project(as_vector([scale*sin(freq*pi*x), 0]), W.sub(0))
ic_v = project(as_vector([-(freq*pi)**2*scale*sin(freq*pi*x), 0]), W.sub(1))
u_.assign(ic_u)
v_.assign(ic_v)

bc_u = DirichletBC(W.sub(0), as_vector([scale*sin(freq*pi*x),0]), "on_boundary")
bc_v = DirichletBC(W.sub(1), as_vector([-(freq*pi)**2*scale*sin(freq*pi*x),0]), "on_boundary")

w = Function(W)
w.assign(w_)
u, v = split(w)
u_, v_ = split(w_)
#################################

u_test, v_test = TestFunctions(W)


nu = 0.001
mu = 0.001

timestep = 1.0/n

#################################
F = (inner((u - u_)/timestep, u_test)
    + inner(dot(u,nabla_grad(u)), u_test) 
    - nu*inner(v, u_test)
    - mu*inner(grad(v), grad(u_test))
    + inner(v, v_test)
    + inner(grad(u), grad(v_test)))*dx
#################################


# problem = NonlinearVariationalProblem(F, w, bcs=[bc_u, bc_v])
problem = NonlinearVariationalProblem(F, w)

# sp_it = {
#    "ksp_type": "gmres",
#    "pc_type": "fieldsplit",
#    "pc_fieldsplit_type": "schur",
#    "pc_fieldsplit_0_fields": "1",
#    "pc_fieldsplit_1_fields": "0",
#    "pc_fieldsplit_schur_precondition": "selfp",
#    "fieldsplit_0_pc_type": "ilu",
#    "fieldsplit_0_ksp_type": "preonly",
#    "fieldsplit_1_ksp_type": "preonly",
#    "fieldsplit_1_pc_type": "gamg",
#    "ksp_monitor": None,
#    "ksp_max_it": 20,
#    "snes_monitor": None
#    }

sp_it = {
    'mat_type': 'aij',
    'ksp_type': 'preonly',
    'pc_type': 'lu'
    }

solver = NonlinearVariationalSolver(problem,
                                    solver_parameters=sp_it)

u_, v_ = w_.split()
u, v = w.split()

t = 0.0
T_end = 1.0

output_dir_path = os.path.dirname(os.path.realpath(__file__))

outfile_u = File(output_dir_path + "/../data/kse_u.pvd")
outfile_u.write(project(u, V_out, name="Velocity"), time=t)

outfile_v = File(output_dir_path + "/../data/kse_v.pvd")
outfile_v.write(project(v, V_out, name="Velocity_xx"))

while (t <= T_end):
    solver.solve()
    w_.assign(w)
    u, v = w.split()
    t += timestep
    outfile_u.write(project(u, V_out, name="Velocity"), time=t)
    outfile_v.write(project(v, V_out, name="Velocity_xx"), time=t)
    print("time step t = ", t, " ....done")
