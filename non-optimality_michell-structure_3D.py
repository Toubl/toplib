import numpy
from topopt.boundary_conditions import CantileverCenterDistributedLoadBoundaryCondition
from topopt.problems import ComplianceNoRBE2Problem3
from topopt.solvers import TopOptSolver
from topopt.filters import SensitivityBasedFilter
from topopt.continuation_approach import continuation_method
from voxel import x_to_stl
import math

##################################################################################
###################### +++++ user-defined input setup +++++ ######################
##################################################################################
multi = 32
nelx, nely, nelz = 2*multi, multi, math.ceil(multi/4)    # number of element in each direction
volfrac = 0.15               # volume fraction for constraints

# contination approach set up
penalty_increment = 0.2         # The step size of penalty increment.
penalty_max = 4                 # The desired maximum penalty value for SIMP.
loop_per_penalty = numpy.array([50])  # The iteration array to indicate the number of iterations for each penalty value.
numberOfLoops_penalty_max = 50        # The number of iterations at maximum penalty value.

# filter set up
rmin = 1.2      # filter radius for anti-checkerboard filtering
FilterOn = 1       # boolean for filter activation  
loops_deactivateFilter = 25     # number of iterations after filter deactivation
##################################################################################
###################### +++++ user-defined input setup +++++ ######################
##################################################################################

### Initialization ###
x = volfrac * numpy.ones(nely*nelx*nelz, dtype=float)
file_path = 'non-optimality-michell-3D_b.txt'

# define boundary condition
bc = CantileverCenterDistributedLoadBoundaryCondition(nelx, nely, nelz)

# define the magnitude of force and percentage of central distributed-load area 
F = numpy.array( [[-1]])
percentage = 0.2
bc.set_forces(F, percentage)

# values and load cases of constraints in list
# stays empty for compliance optimization
constraints = []
constraints_f = []


# define filter
topopt_filter = SensitivityBasedFilter(nelx, nely, nelz, rmin)

# optimization with continuation approach
penalty_array, loop_array, numberOfPenalty = continuation_method(penalty_increment, penalty_max, loop_per_penalty, numberOfLoops_penalty_max)

current_iter = 0


    # problem to optimize given objectives and constraints
topopt_problem = ComplianceNoRBE2Problem3(bc, penalty_array[0], volfrac, topopt_filter, FilterOn, constraints, constraints_f)
topopt_problem.Emin = 1e-3
topopt_problem.nu = 0.3
    #topopt_problem.iter = current_iter
for i in range(numberOfPenalty):  
    topopt_problem.penalty = penalty_array[i]
    topopt_solver = TopOptSolver(topopt_problem, len(constraints), int(loop_array[i]))
    x_opt = topopt_solver.optimize(x)
    x = x_opt
    #current_iter = topopt_problem.iter
    
# optimization with filter deactivation
topopt_problem.FilterOn = 0
#topopt_problem = ComplianceNoRBE2Problem3(bc, penalty_max, volfrac, topopt_filter, FilterOn, constraints, constraints_f)
#topopt_problem.Emin = 1e-3
#topopt_problem.nu = 0.3
#topopt_problem.iter = current_iter
topopt_solver = TopOptSolver(topopt_problem, len(constraints), loops_deactivateFilter)
x_opt = topopt_solver.optimize(x)


# display optimized tolopogy
x_to_stl(nelx, nely, nelz, 0.1, x_opt, 'non-optimality-michell-3D_b.stl')

# save optimize density values to txt file
with open(file_path,'w') as file:
    for item in x_opt:
        file.write(str(item)+'\n')