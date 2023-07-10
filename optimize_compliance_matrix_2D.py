import pyfiglet
import numpy
from topopt.boundary_conditions import FixedBeamBoundaryConditions
from topopt.problems import MinMassProblem2
from topopt.solvers import TopOptSolver
from topopt.solvers import OCSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI
from voxel import x_to_stl

def calculate_strain_energy(cxx, cxy, cyy, M, y):
    F = numpy.zeros((2, 1))
    F[0, 0] = 1
    F[1, 0] = M
    V = cyy * M ** 2 + 2 * cxy * M + cxx
    if isinstance(V, numpy.ndarray):
        V = V[0]
    return V, F


def calculate_minimum_strain_energy(cxx, cxy, cyy, y):
    M_extr = -cxy / cyy
    V, F = calculate_strain_energy(cxx, cxy, cyy, M_extr, y)
    return V, F

def read_array_from_file(file_path):
    array = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = line.strip()
            array.append(entry)
    return numpy.array(array).astype(numpy.float64)

nelx, nely, nelz = 40, 10, 1  # Number of elements in the x y and z-direction
volfrac = 1  # Volume fraction for constraints
penal = 3  # Penalty for SIMP
rmin = 2  # Filter radius

# Initial solution
x = volfrac * numpy.ones(nely * nelx * nelz, dtype=float)
file_path = 'x_opt.txt'
# x = read_array_from_file(file_path)

# Boundary conditions defining the loads and fixed points
bc = FixedBeamBoundaryConditions(nelx, nely, nelz)

# define force vector
F = numpy.zeros((6, 1))
F[1, 0] = 1  # 0: F_x, 1: F_y, 2: F_z, 3: M_x, 4: M_y, 5: M_z
bc.set_forces(F)

# Define desired 4x4 stiffness matrix
# K_11 = ...
# K_22 = ...
# K_44 = ...
# K_33 = K_11
# K_43 = -(K_11 - K_22 + K_44)/2
#
# K = numpy.array([[K_33, K_43], [K_43, K_44]])
#
# C_desired_y = numpy.linalg.inv(K)

# Directly define desired Compliance matrix
c_33 = 6000*32
c_43 = -2000*32
c_44 = 1000*32

C_desired_y = numpy.array([[c_33, c_43], [c_43, c_44]])

# Check if matrix is feasible. Compliance and Stiffness matrices must be positive definite
if numpy.linalg.det(C_desired_y) <= 0:
    print('unfeasible matrix!')
    exit(1)

# values and load cases of constraints in list
constraints = []
constraints_f = []

# calculate constraints to satisfy desired compliance matrix
constraint, constraint_f = calculate_minimum_strain_energy(C_desired_y[0, 0], C_desired_y[1, 0], C_desired_y[1, 1], 1)
constraints.append(constraint)
constraints_f.append(constraint_f)

steps = 40
numbers = numpy.linspace(-5, 5, num=steps)
for i in range(steps):
    constraint, constraint_f = calculate_strain_energy(C_desired_y[0, 0], C_desired_y[1, 0], C_desired_y[1, 1], constraints_f[0][1] + numbers[i], 1)
    # constraint, constraint_f = calculate_strain_energy(C_desired_y[0, 0], C_desired_y[1, 0], C_desired_y[1, 1], numbers[i], 1)
    constraints.append(constraint)
    constraints_f.append(constraint_f)


# Problem to optimize given objective and constraints
topopt_filter = DensityBasedFilter(nelx, nely, nelz, rmin)
gui = GUI(bc, "Topology Optimization Example")
problem = MinMassProblem2(bc, penal, volfrac, topopt_filter, constraints, constraints_f, gui)
problem.C_desired_y = C_desired_y
problem.reducedofs = 0  # delete dofs of elements that are close to zero in density, speeding up optimization
solver = TopOptSolver(problem, len(constraints))

x_opt = solver.optimize(x)

# save optimized density values to txt file
with open(file_path, 'w') as file:
    for item in x_opt:
        file.write(str(item) + '\n')

# Calculate and display Compliance and stiffness matrix of reduced system
_, C_red = problem.compute_reduced_stiffness(problem.xPhys)
# display optimized topology
x_to_stl(nelx, nely, nelz, 0.1, x_opt, 'output.stl')
