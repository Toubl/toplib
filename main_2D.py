import pyfiglet
import numpy
from topopt.boundary_conditions import FixedBeamBoundaryConditions
from topopt.problems import ComplianceProblem2
from topopt.solvers import TopOptSolver
from topopt.solvers import OCSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI
from voxel import x_to_stl
import time

title = pyfiglet.figlet_format("TopOpt", font="small", width=100)
with open('utils/ascii_ose.txt') as f:
    ose=f.read()

print(title)
print(ose)


def read_array_from_file(file_path):
    array = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = line.strip()
            array.append(entry)
    return numpy.array(array).astype(numpy.float64)


nelx, nely, nelz = 100, 50, 1  # Number of elements in the x y and z-direction
volfrac = 0.2  # Volume fraction for constraints
penal = 3  # Penalty for SIMP
rmin = 2
# Initial solution
x = volfrac * numpy.ones(nely * nelx * nelz, dtype=float)
file_path = 'x_opt.txt'  # Replace with the actual file path
# x = read_array_from_file(file_path)

# Boundary conditions defining the loads and fixed points
bc = FixedBeamBoundaryConditions(nelx, nely, nelz)

# define force vector
F = numpy.zeros((6, 2))
# 2D-Example
F[0, 0] = 1  # 0: F_y, 1: M_z
F[1, 1] = -0.5

bc.set_forces(F)
bc.set_forces(F)

constraints = []
constraints_f = []

# gui = GUI(bc, "Topology Optimization Example")
topopt_filter = DensityBasedFilter(nelx, nely, nelz, rmin)

# Problem to optimize given objective and constraints
problem = ComplianceProblem2(bc, penal, volfrac, topopt_filter, constraints, constraints_f, 0)
problem.reducedofs = 1
solver = TopOptSolver(problem, len(constraints))
x_opt = solver.optimize(x)
with open(file_path, 'w') as file:
    for item in x_opt:
        file.write(str(item) + '\n')

problem.compute_reduced_stiffness(problem.xPhys)

tol = 0.1
x_to_stl(nelx, nely, nelz, tol, x_opt, 'output.stl')
