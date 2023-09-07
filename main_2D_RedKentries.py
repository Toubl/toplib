import pyfiglet
import numpy
from topopt.boundary_conditions import RedKentriesBoundaryConditions
from topopt.problems import MinMassRedKentries2
from topopt.solvers import TopOptSolver
from topopt.filters import DensityBasedFilter
from voxel import x_to_stl
from topopt.guis import GUI


title = pyfiglet.figlet_format("TopOpt", font="small", width=100)
with open('utils/ascii_ose.txt') as f:
    ose=f.read()

# print(title)
# print(ose)


def read_array_from_file(file_path):
    array = []
    with open(file_path, 'r') as file:
        for line in file:
            entry = line.strip()
            array.append(entry)
    return numpy.array(array).astype(numpy.float64)


# nelx, nely, nelz = 48, 16, 1
# nelx, nely, nelz = 38, 13, 1 # This seems to work!!!
nelx, nely, nelz = 5, 1, 1 # This seems to work!!!
volfrac = 1.0  # Volume fraction for constraints
penal = 3  # Penalty for SIMP
rmin = 1.2
# Initial solution
x = volfrac * numpy.ones(nely * nelx * nelz, dtype=float)                        
file_path = 'x_opt.txt'
# x = read_array_from_file(file_path)

# Boundary conditions defining the loads and fixed points
# bc = FixedBeamBoundaryConditions(nelx, nely, nelz)
bc = RedKentriesBoundaryConditions(nelx, nely, nelz)

# define force vector
F = numpy.zeros((6, 1))
F[0, 0] = 0  # 0: F_x, 1: F_y, 2: F_z, 3: M_x, 4: M_y, 5: M_z || 0: F_y, 1: M_z (2D)
# F[4, 0] = 1
# F[2, 1] = 1
# F[4, 1] = 0.5
# F[0, 1] = 5

bc.set_forces(F)

# values and load cases of constraints in list
# stays empty for Compliance optimization
# constraints = []
# K00, K11, K33 = 1.9e9/8, 1.5e12/2.6, 1.5e12/2.9 # Gives a solution similar to left component kri2021
K00, K11, K33 = 1.9e9/8, 1.5e12/4, 1.5e12/100 # This seems to work pretty well for a mesh of 38, 13, 1
# K00, K11, K33 = 1.9e9/0.8, 1.5e12/0.8, 1.5e12/0.8
constraints = numpy.array([K00, K11, K33, K00, K11, K33]) # Twice the same to generate the equality constraint from the negative and postivive side
constraints_f = []

topopt_filter = DensityBasedFilter(nelx, nely, nelz, rmin)

# Problem to optimize given objective and constraints
gui = GUI(bc, "Topology Optimization Example")
# domain_lens = [0.3*0.98-0.0, 0.098] # Kri 2021
# domain_lens = [0.3*0.98-0.0, 0.098] # Kri 2021
domain_lens = [2, 1] # Kri 2021
# joint_locs = [3e-3, 0] # Kri 2021
joint_locs = [0, 0] # Kri 2021
problem = MinMassRedKentries2(bc, penal, volfrac, topopt_filter, constraints, constraints_f, gui, domain_lens, joint_locs) # Only pass Kreq if MinMassRedKentries problem
problem.Emin = 10
problem.Emax = 70e9
problem.nu = 0.33
problem.reducedofs = 0  # delete dofs of elements that are close to zero in density, speeding up optimization
solver = TopOptSolver(problem, len(constraints))

x_opt = solver.optimize(x)

print(solver.opt.last_optimize_result())

# Calculate and display Compliance and stiffness matrix of reduced system
# _, C_red = problem.compute_reduced_stiffness(x_opt)
# display optimized topology
x_to_stl(nelx, nely, nelz, 0.1, x_opt, 'output.stl')

# save optimized density values to txt file
with open(file_path, 'w') as file:
    for item in x_opt:
        file.write(str(item) + '\n')
