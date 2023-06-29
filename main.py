import pyfiglet
import numpy
from topopt.boundary_conditions import FixedBeamBoundaryConditions
from topopt.problems import ComplianceProblem3
from topopt.solvers import TopOptSolver
from topopt.filters import DensityBasedFilter
from voxel import x_to_stl


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


nelx, nely, nelz = 40, 10, 10
volfrac = 0.2  # Volume fraction for constraints
penal = 3  # Penalty for SIMP
rmin = 2
# Initial solution
x = volfrac * numpy.ones(nely * nelx * nelz, dtype=float)                        
file_path = 'x_opt.txt'
# x = read_array_from_file(file_path)

# Boundary conditions defining the loads and fixed points
bc = FixedBeamBoundaryConditions(nelx, nely, nelz)

# define force vector
F = numpy.zeros((6, 1))
F[1, 0] = 1  # 0: F_x, 1: F_y, 2: F_z, 3: M_x, 4: M_y, 5: M_z || 0: F_y, 1: M_z (2D)
# F[5, 0] = -0.5
# F[2, 1] = 1
# F[4, 1] = 0.5
# F[0, 1] = 5

bc.set_forces(F)

# values and load cases of constraints in list
# stays empty for Compliance optimization
constraints = []
constraints_f = []

topopt_filter = DensityBasedFilter(nelx, nely, nelz, rmin)

# Problem to optimize given objective and constraints
problem = ComplianceProblem3(bc, penal, volfrac, topopt_filter, constraints, constraints_f, 0)
problem.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization
solver = TopOptSolver(problem, len(constraints))

x_opt = solver.optimize(x)

# Calculate and display Compliance and stiffness matrix of reduced system
_, C_red = problem.compute_reduced_stiffness(problem.xPhys)
# display optimized topology
x_to_stl(nelx, nely, nelz, 0.1, x_opt, 'output.stl')

# save optimized density values to txt file
with open(file_path, 'w') as file:
    for item in x_opt:
        file.write(str(item) + '\n')
