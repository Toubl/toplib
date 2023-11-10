import numpy as np
import pyfiglet
import numpy
from topopt.boundary_conditions import FixedBeamBoundaryConditions
from topopt.problems import ComplianceProblem3
from topopt.solvers import TopOptSolver
from topopt.solvers import OCSolver
from topopt.filters import DensityBasedFilter
from topopt.filters import SensitivityBasedFilter
from voxel import x_to_stl
import matlab.engine


fac = 12
nelx, nely, nelz = 8*fac, 4*fac, 1*fac # number of elements in x,y,z-direction
length_x = 1 # Length Beam
volfrac = 0.1 # Volume Fraction
penal = 1  #Penalty for SIMP
rmin = 1.2 # Filter radius
# Initial solution
x = volfrac * numpy.ones(nely * nelx * nelz, dtype=float)
file_path = 'x_opt.txt'

# Boundary conditions defining the loads and fixed points
bc = FixedBeamBoundaryConditions(nelx, nely, nelz)

# define force vector
F = numpy.zeros((6, 1))
F[1, 0] = 1

bc.set_forces(F)

# values and load cases of constraints in list
# stays empty for Compliance optimization
constraints = []
constraints_f = []

#topopt_filter = DensityBasedFilter(nelx, nely, nelz, rmin)
topopt_filter = SensitivityBasedFilter(nelx, nely, nelz, rmin)

# Problem to optimize given objective and constraints
gui = 0
process_number = 0
problem = ComplianceProblem3(bc, penal, volfrac, topopt_filter, constraints, constraints_f, gui, process_number, length_x)
problem.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization

# solver = TopOptSolver(problem, len(constraints))
solver = OCSolver(problem, len(constraints))

x_opt, obj = solver.optimize(x)

# save optimized density values to txt file
with open(file_path, 'w') as file:
    for item in x_opt:
        file.write(str(item) + '\n')

# Calculate and display Compliance and stiffness matrix of reduced system
K_red, C_red = problem.compute_reduced_stiffness(x_opt)

# display optimized topology
density_limit = 0.1
x_to_stl(nelx, nely, nelz, density_limit, x_opt, 'output.stl') # display topology and turn to stl
