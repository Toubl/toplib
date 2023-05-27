import pyfiglet
import numpy
from topopt.boundary_conditions import FixedBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.solvers import TopOptSolver
from topopt.solvers import OCSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI
from voxel import x_to_stl

title = pyfiglet.figlet_format("TopOpt", font="small", width=100)
with open('utils/ascii_ose.txt') as f:
    ose=f.read()

print(title)
print(ose)


nelx, nely, nelz = 20, 10, 10  # Number of elements in the x y and z-direction
volfrac = 0.2  # Volume fraction for constraints
penal = 3.0  # Penalty for SIMP
rmin = 2  # Filter radius
c_critical = 100

# Initial solution
x = volfrac * numpy.ones(nely * nelx * nelz, dtype=float)

# Boundary conditions defining the loads and fixed points
bc = FixedBeamBoundaryConditions(nelx, nely, nelz)

# Problem to optimize given objective and constraints
problem = ComplianceProblem(bc, penal)

# define force vector
problem.f[1] = 1  # 0: F_x, 1: F_y, 2: F_z, 3: M_x, 4: M_y, 5: M_z

gui = GUI(problem, "Topology Optimization Example")
topopt_filter = DensityBasedFilter(nelx, nely, nelz, rmin)
solver = TopOptSolver(problem, volfrac, topopt_filter, gui, c_critical)
x_opt = solver.optimize(x)
with open('x_opt.txt', 'w') as file:
    for item in x_opt:
        file.write(str(item) + '\n')
tol = 0.1
x_to_stl(nelx, nely, nelz, tol, 'output.stl')
problem.compute_reduced_stiffness3(x_opt)
