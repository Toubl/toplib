import pyfiglet
import numpy
import time
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


nelx, nely, nelz = 16, 16, 1
# nelx, nely, nelz = 34, 16, 1
# nelx = 16 # 82 for max length and 16 for min length
# elem_lenx = 6.125*(82/48) # mm
elem_lenx = 6.125 # mm
print(f'Domain length = {nelx*elem_lenx}')
elem_leny = 6.125 # mm
# nelx, nely, nelz = 38, 13, 1 # This seems to work!!!
# nelx, nely, nelz = 5, 1, 1
volfrac = 0.9 # Volume fraction
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
# K00, K11, K33 = 1.8e4, 3.5e6, 3.5e6 # Max values for shortest length
# K00, K11, K33 = 1e3, 4e7, 4e7 # Values for benchmark
K00, K11, K33 = 2556.12,	5.1035e+07,	4.79732e+07
# K00, K11, K33 = 2.5e2, 2.5e7, 2.5e7 # Values for 82 nelx length
constraints = numpy.array([K00, K11, K33, K00, K11, K33]) # Twice the same to generate the equality constraint from the negative and postivive side
# constraints = numpy.array([K00, K11, K33])
constraints_f = []

topopt_filter = DensityBasedFilter(nelx, nely, nelz, rmin)

# Problem to optimize given objective and constraints
gui = GUI(bc, "Topology Optimization Example")
# gui=0
domain_lens = [nelx*elem_lenx, 98] # Kri 2021
# domain_lens = [2, 1] # 
joint_locs = [3, 0] # Kri 2021
# joint_locs = [0, 0] # 
problem = MinMassRedKentries2(bc, penal, volfrac, topopt_filter, constraints, constraints_f, gui, domain_lens, joint_locs) # Only pass Kreq if MinMassRedKentries problem
problem.Emin = 10 # These guys will only affect compute_young_moduly, which scales lk
problem.Emax = 70e3
problem.nu = 0.33
problem.reducedofs = 0  # delete dofs of elements that are close to zero in density, speeding up optimization
solver = TopOptSolver(problem, len(constraints))

t0 = time.perf_counter()
x_opt = solver.optimize(x)
t1 = time.perf_counter()
print(f'Computation time: {t1-t0} second(s)')

print(f'Exit code: {solver.opt.last_optimize_result()}')

# Calculate final Kgr entries
x_phys = problem.filter_variables(x_opt)
Kfull  = problem.build_K(x_phys,remove_constrained=False).tocsr()
Tg  = problem.StaticCondensationInterfaces(Kfull)
Tr =  problem.Tr
Tgr = Tg @ Tr
Tgr_t = Tgr.transpose()
Kgr = Tgr_t @ Kfull @ Tgr
print([Kgr[0,0], Kgr[1,1], Kgr[3,3]])
K00r, K11r, K33r = problem.Kreq[:]
print([(Kgr[0,0]-K00r)*100/K00r, (Kgr[1,1]-K11r)*100/K11r, (Kgr[3,3]-K33r)*100/K33r])

# Component Mass
rho_material = 2700 # kg/m^3
volume_fraction = solver.opt.last_optimum_value()/(nelx*nely*nelz) # Sum of densities divided by number of elements
e_volume = problem.elen_x*problem.elen_y*1 # Element volume mm^3
total_volume = e_volume*(nelx*nely*nelz)*1e-9 # m^3
mass = total_volume*volume_fraction*rho_material
print(f'Component Mass: {mass}')

# display optimized topology
# x_to_stl(nelx, nely, nelz, 0.1, x_opt, 'output.stl')

# save optimized density values to txt file
# with open(file_path, 'w') as file:
#     for item in x_opt:
#         file.write(str(item) + '\n')
