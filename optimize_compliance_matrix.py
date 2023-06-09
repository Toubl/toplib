import pyfiglet
import numpy
from topopt.boundary_conditions import FixedBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.solvers import TopOptSolver
from topopt.solvers import OCSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI
from voxel import x_to_stl
import time

def calculate_strain_energy(cxx, cxy, cyy, M, y):
    F = numpy.zeros((6, 1))
    if y == 1:
        F[1, 0] = 1
        F[5, 0] = M
    else:
        F[2, 0] = 1
        F[4, 0] = M
    V = cyy * M ** 2 + 2 * cxy * M + cxx
    if isinstance(V, numpy.ndarray):
        V = V[0]
    return V, F


def calculate_minimum_strain_energy(cxx, cxy, cyy, y):
    M_extr = -cxy / cyy
    V, F = calculate_strain_energy(cxx, cxy, cyy, M_extr, y)
    return V, F
start_time = time.time()


nelx, nely, nelz = 40, 10, 10  # Number of elements in the x y and z-direction
volfrac = 1  # Volume fraction for constraints
penal = 3  # Penalty for SIMP
rmin = 2  # Filter radius

# Initial solution
x = volfrac * numpy.ones(nely * nelx * nelz, dtype=float)

# Boundary conditions defining the loads and fixed points
bc = FixedBeamBoundaryConditions(nelx, nely, nelz)

# define force vector
F = numpy.zeros((6, 1))
F[1, 0] = 1  # 0: F_x, 1: F_y, 2: F_z, 3: M_x, 4: M_y, 5: M_z
bc.set_forces(F)

# Problem to optimize given objective and constraints
problem = ComplianceProblem(bc, penal)
gui = GUI(problem, "Topology Optimization Example")
topopt_filter = DensityBasedFilter(nelx, nely, nelz, rmin)

a = 3000
b = 3000
c = 825

change = 1
change_x = (change - 1) * b ** 2 / (4 * a)
a = a * change
b = b * change
c = c + change_x

C_desired_y = numpy.array([[c, b / 2], [b / 2, a]])
C_desired_z = 2 * numpy.array([[c, -b / 2], [-b / 2, a]])

if numpy.linalg.det(C_desired_y) <= 0 or numpy.linalg.det(C_desired_z) <= 0:
    print('unfeasible matrix!')
    exit(1)
constraints = []
constraints_f = []

problem.C_desired_y = C_desired_y
problem.C_desired_z = C_desired_z

constraint, constraint_f = calculate_minimum_strain_energy(C_desired_y[0, 0], C_desired_y[1, 0], C_desired_y[1, 1], 1)
constraints.append(constraint)
constraints_f.append(constraint_f)

constraint, constraint_f = calculate_minimum_strain_energy(C_desired_z[0, 0], C_desired_z[1, 0], C_desired_z[1, 1], 0)
constraints.append(constraint)
constraints_f.append(constraint_f)

for i in range(10):
    constraint, constraint_f = calculate_strain_energy(C_desired_y[0, 0], C_desired_y[1, 0], C_desired_y[1, 1], constraints_f[0][5] - 2 + 0.4 * i, 1)
    constraints.append(constraint)
    constraints_f.append(constraint_f)

    constraint, constraint_f = calculate_strain_energy(C_desired_z[0, 0], C_desired_z[1, 0], C_desired_z[1, 1], constraints_f[1][4] - 2 + 0.4 * i, 0)
    constraints.append(constraint)
    constraints_f.append(constraint_f)

    constraint, constraint_f = calculate_strain_energy(C_desired_y[0, 0], C_desired_y[1, 0], C_desired_y[1, 1], constraints_f[0][5] + 2 - 0.4 * i, 1)
    constraints.append(constraint)
    constraints_f.append(constraint_f)

    constraint, constraint_f = calculate_strain_energy(C_desired_z[0, 0], C_desired_z[1, 0], C_desired_z[1, 1], constraints_f[1][4] + 2 - 0.4 * i, 0)
    constraints.append(constraint)
    constraints_f.append(constraint_f)

solver = TopOptSolver(problem, volfrac, topopt_filter, gui, constraints, constraints_f)


# for i in range(4):
#     x_opt = solver.optimize(x)
#     _, C_red = problem.compute_reduced_stiffness3(solver.xPhys)
#     x_to_stl(nelx, nely, nelz, 0.1, x_opt, 'output.stl')
#
#     C_red_y = numpy.array([[C_red[i][j] for j in [0, 4]] for i in [0, 4]])
#     C_red_z = numpy.array([[C_red[i][j] for j in [1, 3]] for i in [1, 3]])
#     C_diff_y = C_red_y - C_desired_y
#     C_diff_z = C_red_z - C_desired_z
#
#     constraint, constraint_f = calculate_minimum_strain_energy(C_diff_y[0, 0], C_diff_y[1, 0], C_diff_y[1, 1], 1)
#     constraint, constraint_f = calculate_strain_energy(C_desired_y[0, 0], C_desired_y[1, 0], C_desired_y[1, 1], constraint_f[5], 1)
#     constraints.append(constraint)
#     constraints_f.append(constraint_f)
#
#     constraint, constraint_f = calculate_minimum_strain_energy(C_diff_z[0, 0], C_diff_z[1, 0], C_diff_z[1, 1], 0)
#     constraint, constraint_f = calculate_strain_energy(C_desired_z[0, 0], C_desired_z[1, 0], C_desired_z[1, 1], constraint_f[4], 0)
#     constraints.append(constraint)
#     constraints_f.append(constraint_f)
#
#     solver = TopOptSolver(problem, volfrac, topopt_filter, gui, constraints, constraints_f)

x_opt = solver.optimize(x)
_, C_red = problem.compute_reduced_stiffness3(solver.xPhys)
x_to_stl(nelx, nely, nelz, 0.1, x_opt, 'output.stl')


end_time = time.time()
print('Execution took:')
print(end_time - start_time)