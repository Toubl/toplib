import pyfiglet
import numpy
from topopt.boundary_conditions import MBBBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.solvers import TopOptSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI
# from topopt.xdmf_writer import XDMFWriter
from topopt.utils import read_json_file
import h5py
from topopt.postprocessor import Postprocessor

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
print(x)

# Boundary conditions defining the loads and fixed points
bc = MBBBeamBoundaryConditions(nelx, nely)

# Problem to optimize given objective and constraints
problem = ComplianceProblem(bc, penal)
# gui = GUI(problem, "Topology Optimization Example")
gui = GUI(problem, "Topology Optimization Example")
topopt_filter = DensityBasedFilter(nelx, nely, rmin)
solver = TopOptSolver(problem, volfrac, topopt_filter, gui)
x_opt = solver.optimize(x)
x_opt = x_opt.reshape((nelx, nely))
print(x_opt)
# converting 2D matrix to 3D tensor
x_opt = numpy.expand_dims(x_opt, axis=2)
x_opt = numpy.repeat(x_opt, 10, axis=2)
x_opt = numpy.flip(numpy.transpose(x_opt, (0, 2, 1)), axis=2)

print(x_opt)


# Save the tensor to a text file, with elements separated by spaces
numpy.savetxt('tensor.txt', x_opt.flatten(), fmt='%d')

postprocessor = Postprocessor(x_opt)
postprocessor.process(2, 3, 1.0, 3, 2, 'initial.stl', 'final.stl')
input("Press enter...")

