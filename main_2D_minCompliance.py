import pyfiglet
import numpy
from topopt.boundary_conditions import CantileverBoundaryConditions
from topopt.problems import ComplianceProblem2
from topopt.solvers import TopOptSolver
from topopt.filters import DensityBasedFilter
from voxel import x_to_stl
from topopt.guis import GUI


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


nelx, nely, nelz = 48, 16, 1
volfrac = 0.4  # Volume fraction for constraints
penal = 3  # Penalty for SIMP
rmin = 1.2
# Initial solution
x = volfrac * numpy.ones(nely * nelx * nelz, dtype=float)                        
file_path = 'x_opt.txt'
# x = read_array_from_file(file_path)

# Boundary conditions defining the loads and fixed points
bc = CantileverBoundaryConditions(nelx, nely, nelz)
# bc = RedKentriesBoundaryConditions(nelx, nely, nelz)

topopt_filter = DensityBasedFilter(nelx, nely, nelz, rmin)

# Problem to optimize given objective and constraints
gui = GUI(bc, "Topology Optimization Example")
domain_lens = [0.3*0.98, 0.098] # Kri 2021
joint_locs = [3e-3, 0] # Kri 2021
problem = ComplianceProblem2(bc, penal, volfrac, topopt_filter, constraints=[], constraints_f=[], gui=gui) # Only pass Kreq if MinMassRedKentries problem
problem.Emin = 10
problem.Emax = 70e9
problem.nu = 0.33
problem.reducedofs = 0  # delete dofs of elements that are close to zero in density, speeding up optimization
solver = TopOptSolver(problem, n_constraints=1)

x_opt = solver.optimize(x)


# Calculate and display Compliance and stiffness matrix of reduced system
# _, C_red = problem.compute_reduced_stiffness(x_opt)
# display optimized topology
x_to_stl(nelx, nely, nelz, 0.1, x_opt, 'output.stl')

# save optimized density values to txt file
with open(file_path, 'w') as file:
    for item in x_opt:
        file.write(str(item) + '\n')
