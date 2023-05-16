import plotly.graph_objects as go
import pyfiglet
import numpy
from topopt.boundary_conditions import MBBBeamBoundaryConditions
from topopt.problems import ComplianceProblem
from topopt.solvers import TopOptSolver
from topopt.filters import DensityBasedFilter
from topopt.guis import GUI

def create_2d_heatmap(width, height, values):
    if len(values) != width * height:
        raise ValueError(f"Die Länge von 'values' muss gleich {width * height} sein")

    # Reshape the values to match the width and height
    matrix = numpy.array(values).reshape(height, width)

    # Rotate the matrix 90 degrees clockwise
    matrix = numpy.rot90(matrix, 1)

    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        colorscale='Viridis',  # change to the colorscale you prefer
        zmin=0,
        zmax=1,
    ))

    fig.update_layout(
        autosize=True,
        margin=dict(
            l=50,
            r=50,
            b=100,
            t=100,
            pad=4
        ),
    )

    return fig

nelx, nely = 80, 80  # Number of elements in the x and y
volfrac = 0.36  # Volume fraction for constraints
penal = 4.0  # Penalty for SIMP
rmin = 4.4  # Filter radius

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
print(x_opt)

# nelx, nely = 4, 4
# x_opt = numpy.random.rand(16)
# print(x_opt)

fig = create_2d_heatmap(nelx, nely, x_opt)
fig.show()

input("Press enter...")
