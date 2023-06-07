import pyfiglet
import numpy
from topopt.boundary_conditions import FixedBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.solvers import TopOptSolver
from topopt.solvers import OCSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI
from voxel import x_to_stl
from multiprocessing import Process, freeze_support

if __name__ == '__main__':
    freeze_support()

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


    nelx, nely, nelz = 80, 20, 20
    # nelx, nely, nelz = 160, 80, 1  # Number of elements in the x y and z-direction
    volfrac = 0.2  # Volume fraction for constraints
    penal = 3  # Penalty for SIMP
    rmin = 2  # Filter radius

    # Initial solution
    x = volfrac * numpy.ones(nely * nelx * nelz, dtype=float)
    file_path = 'x_opt.txt'  # Replace with the actual file path
    # x = read_array_from_file(file_path)

    # Boundary conditions defining the loads and fixed points
    bc = FixedBeamBoundaryConditions(nelx, nely, nelz)

    # define force vector
    F = numpy.zeros((6, 1))
    # 3D-Example
    F[1, 0] = 1  # 0: F_x, 1: F_y, 2: F_z, 3: M_x, 4: M_y, 5: M_z || 0: F_y, 1: M_z (2D)
    F[5, 0] = -0.5
    # 2D-Example
    # F[0, 0] = 1
    # F[1, 0] = -0.5

    bc.set_forces(F)

    constraints = []
    constraints_f = []

    # constraints.append(100)
    # constraints_f.append(F)

    # Problem to optimize given objective and constraints
    problem = ComplianceProblem(bc, penal)

    gui = GUI(problem, "Topology Optimization Example")
    topopt_filter = DensityBasedFilter(nelx, nely, nelz, rmin)
    solver = TopOptSolver(problem, volfrac, topopt_filter, gui, constraints, constraints_f)
    x_opt = solver.optimize(x)
    with open(file_path, 'w') as file:
        for item in x_opt:
            file.write(str(item) + '\n')

    if nelz > 1:
        problem.compute_reduced_stiffness3(solver.xPhys)
        problem.compute_reduced_stiffness3(x_opt)
    else:
        problem.compute_reduced_stiffness(x_opt)

    tol = 0.1
    x_to_stl(nelx, nely, nelz, tol, x_opt, 'output.stl')