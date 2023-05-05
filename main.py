import pyfiglet
import numpy
from utils.import_options import get_options

#TODO replace imports with iterator
from topopt.boundary_conditions import MBBBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.solvers import TopOptSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI

#TODO
#from iterators.iterator import Iterator

import os
import sys
import time

try:
    import simplejson as json
except ImportError:
    import json

def main(args):
    title = pyfiglet.figlet_format("TopOpt", font="small", width=100)
    with open('utils/ascii_ose.txt') as f:
        ose=f.read()

    print(title)
    print(ose)

    #TODO read json input file
    #options = get_options(args)

    #TODO build iterator
    #my_iterator = Iterator.create_iterator(options)

    #TODO run topology optimization
    #my_iterator.optimize()


    nelx, nely = 180, 60  # Number of elements in the x and y
    volfrac = 0.4  # Volume fraction for constraints
    penal = 3.0  # Penalty for SIMP
    rmin = 5.4  # Filter radius

    # Initial solution
    x = volfrac * numpy.ones(nely * nelx, dtype=float)

    # Boundary conditions defining the loads and fixed points
    bc = MBBBeamBoundaryConditions(nelx, nely)

    # Problem to optimize given objective and constraints
    problem = ComplianceProblem(bc, penal)
    gui = GUI(problem, "Topology Optimization Example")
    topopt_filter = DensityBasedFilter(nelx, nely, rmin)
    solver = TopOptSolver(problem, volfrac, topopt_filter, gui)
    x_opt = solver.optimize(x)

    input("Press enter...")

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))