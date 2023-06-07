import pyfiglet
import numpy
from topopt.boundary_conditions import MBBBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.solvers import TopOptSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI, newGUI
from topopt.utils import read_json_file

# this is a test to push my branch

title = pyfiglet.figlet_format("TopOpt", font="small", width=100)
# with open('utils/ascii_ose.txt') as f:
#     ose=f.read()

print(title)
# print(ose)

# Set path to json file and read in parameters
path_to_json_file = "save.json"
nelx, nely, volfrac, penal, rmin = read_json_file(path_to_json_file)

# Print the values
print('Problem Parameters:')
print("nelx:    ", nelx)
print("nely:    ", nely)
print("volfrac: ", volfrac)
print("penal:   ", penal)
print("rmin:    ", rmin)

# Initial solution
x = volfrac * numpy.ones(nely * nelx, dtype=float)

# Boundary conditions defining the loads and fixed points
bc = MBBBeamBoundaryConditions(nelx, nely)

# Problem to optimize given objective and constraints
problem = ComplianceProblem(bc, penal)
# gui = GUI(problem, "Topology Optimization Example")
gui = GUI(problem, "Topology Optimization Example")
topopt_filter = DensityBasedFilter(nelx, nely, rmin)
solver = TopOptSolver(problem, volfrac, topopt_filter, gui)
x_opt = solver.optimize(x)
print(x_opt)

input("Press enter...")
