import numpy as np
from topopt.boundary_conditions import FixedBeamBoundaryConditions
from topopt.problems import MinMassProblem3
from topopt.solvers import TopOptSolver_Parallel
from topopt.solvers import OCSolver
from topopt.filters import DensityBasedFilter
from topopt.filters import SensitivityBasedFilter
import matplotlib.pyplot as plt
import chaospy
import keyboard
import scipy
import sys
import multiprocessing

# Attention! Component 1 and 3 are switched compared to thesis!
#
#
#
#
#
#
# Attention! Component 1 and 3 are switched compared to thesis!

def compute_four_by_four(K, L):
    # extract 3 characteristic stiffnesses from 2x2 stiffness matrix
    K33 = K[0, 0]
    K44 = K[1, 1]
    K43 = K[1, 0]

    K11 = K33
    K22 = K11 * L ** 2 + K44 + K43 * 2 * L
    K21 = (K11 * L ** 2 + K22 - K44) / (2 * L)
    K31 = -K11
    K41 = (K11 * L ** 2 - K22 + K44) / (2 * L)
    K32 = -(K11 * L ** 2 + K22 - K44) / (2 * L)
    K42 = (K11 * L ** 2 - K22 - K44) / 2

    # obtain 4x4 stiffness matrix
    K_4x4 = np.array([
        [K11, K21, K31, K41],
        [K21, K22, K32, K42],
        [K31, K32, K33, K43],
        [K41, K42, K43, K44]
    ])
    return K_4x4
def constraint_function(particle, constraint_high):
    # 2x2 stiffness matrices from stiffness entries
    K_1 = np.linalg.inv(np.array([[particle[0], particle[1]], [particle[1], particle[2]]]))
    K_2 = np.linalg.inv(np.array([[particle[3], particle[4]], [particle[4], particle[5]]]))
    K_3 = np.linalg.inv(np.array([[particle[6], particle[7]], [particle[7], particle[8]]]))

    length_x_1 = 0.5
    length_x_2 = 2
    length_x_3 = 1

    # 4x4 stiffness matrices of components
    K_4x4_1 = compute_four_by_four(K_1, length_x_1)
    K_4x4_2 = compute_four_by_four(K_2, length_x_2)
    K_4x4_3 = compute_four_by_four(K_3, length_x_3)

    # Assemling
    K_global = np.zeros((8, 8))
    K_global[:4, :4] = K_global[:4, :4] + K_4x4_3
    K_global[2:6, 2:6] = K_global[2:6, 2:6] + K_4x4_2
    K_global[4:, 4:] = K_global[4:, 4:] + K_4x4_1

    # Applying fixed beam boundary conditions
    K_global = K_global[2:, 2:]

    # solving system for system level load case
    F = np.zeros((6, 1))
    F[4, 0] = 1
    U = np.linalg.solve(K_global, F)

    if np.abs(U[4, 0]) > constraint_high:
        # print('unfeasible design!')
        constraint_value = 1
    else:
        # print('feasible design')
        constraint_value = 0
    return constraint_value





# Define the cost function f(x, y) that you want to minimize
def cost_function(particle, problem, solver, x):
    def calculate_strain_energy(cxx, cxy, cyy, M, y):
        F = np.zeros((6, 1))
        if y == 1:
            F[1, 0] = 1
            F[5, 0] = M
        else:
            F[2, 0] = 1
            F[4, 0] = M
        V = cyy * M ** 2 + 2 * cxy * M + cxx
        if isinstance(V, np.ndarray):
            V = V[0]
        return V, F

    def calculate_minimum_strain_energy(cxx, cxy, cyy, y):
        M_extr = -cxy / cyy
        V, F = calculate_strain_energy(cxx, cxy, cyy, M_extr, y)
        return V, F

    C_desired_y_1 = np.array([[particle[0], particle[1]], [particle[1], particle[2]]])
    C_desired_z_1 = C_desired_y_1

    C_desired_y_2 = np.array([[particle[3], particle[4]], [particle[4], particle[5]]])
    C_desired_z_2 = C_desired_y_2

    C_desired_y_3 = np.array([[particle[6], particle[7]], [particle[7], particle[8]]])
    C_desired_z_3 = C_desired_y_3

    problem.active = np.concatenate((np.arange(64), np.arange(64) + 64 * 15))

    # # Component 1:
    constraints = []
    constraints_f = []

    # calculate constraints to satisfy desired compliance matrix
    constraint, constraint_f = calculate_minimum_strain_energy(C_desired_y_1[0, 0], C_desired_y_1[1, 0],
                                                               C_desired_y_1[1, 1], 1)
    constraints.append(constraint)
    constraints_f.append(constraint_f)

    steps = 10  # number of additional constraints
    numbers = np.linspace(-1, 1, num=steps)   # range of M

    for i in range(steps):
        constraint, constraint_f = calculate_strain_energy(C_desired_y_1[0, 0], C_desired_y_1[1, 0],
                                                           C_desired_y_1[1, 1], constraints_f[0][5] + numbers[i], 1)
        constraints.append(constraint)
        constraints_f.append(constraint_f)

    problem.constraints = constraints
    problem.constraint_f = constraints_f
    problem.C_desired_y = C_desired_y_1
    problem.C_desired_z = C_desired_z_1
    problem.iter = 0
    problem.length_x = 0.5
    problem.build_indices()

    x_opt_1 = solver.optimize2(x, len(constraints))

    # # Component 2:
    constraints = []
    constraints_f = []

    constraint, constraint_f = calculate_minimum_strain_energy(C_desired_y_2[0, 0], C_desired_y_2[1, 0],
                                                               C_desired_y_2[1, 1], 1)
    constraints.append(constraint)
    constraints_f.append(constraint_f)

    steps = 10  # number of additional constraints
    numbers = np.linspace(-1, 1, num=steps)   # range of M
    for i in range(steps):
        constraint, constraint_f = calculate_strain_energy(C_desired_y_2[0, 0], C_desired_y_2[1, 0],
                                                           C_desired_y_2[1, 1], constraints_f[0][5] + numbers[i], 1)
        constraints.append(constraint)
        constraints_f.append(constraint_f)

    problem.constraints = constraints
    problem.constraint_f = constraints_f
    problem.C_desired_y = C_desired_y_2
    problem.C_desired_z = C_desired_z_2
    problem.iter = 0
    problem.length_x = 2
    problem.build_indices()

    x_opt_2 = solver.optimize2(x, len(constraints))


    # # Component 3:
    constraints = []
    constraints_f = []

    constraint, constraint_f = calculate_minimum_strain_energy(C_desired_y_3[0, 0], C_desired_y_3[1, 0],
                                                               C_desired_y_3[1, 1], 1)
    constraints.append(constraint)
    constraints_f.append(constraint_f)

    steps = 10  # number of additional constraints
    numbers = np.linspace(-1, 2.5, num=steps)   # range of M
    for i in range(steps):
        constraint, constraint_f = calculate_strain_energy(C_desired_y_3[0, 0], C_desired_y_3[1, 0],
                                                           C_desired_y_3[1, 1], constraints_f[0][5] + numbers[i], 1)
        constraints.append(constraint)
        constraints_f.append(constraint_f)

    problem.constraints = constraints
    problem.constraint_f = constraints_f
    problem.C_desired_y = C_desired_y_3
    problem.C_desired_z = C_desired_z_3
    problem.iter = 0
    problem.length_x = 1
    problem.build_indices()

    x_opt_3 = solver.optimize2(x, len(constraints))

    # Total Mass:
    m1 = x_opt_1.sum() / len(x_opt_1)
    m2 = x_opt_2.sum() / len(x_opt_2)
    m3 = x_opt_3.sum() / len(x_opt_3)

    # print('Masses:', m1, m2, m3)
    return (m1 * 0.5 + m2 * 2 + m3)/3.5


def parallelizer(particles, problem, solver, x, constraint_high):
    num_particles = len(particles)
    costs = np.zeros(num_particles)

    for i in range(num_particles):
        constraint_value = constraint_function(particles[i, :], constraint_high)
        cost = cost_function(particles[i, :], problem, solver, x)

        if not(constraint_value == 0):  # set very high cost so that cosst is not updated when constraint is not met
            cost = 100
        print(cost)

        costs[i] = cost
    return costs

if __name__ == "__main__":
    np.set_printoptions(precision=4)
    nelx, nely, nelz = 16, 16, 4  # Number of elements in the x y and z-direction
    volfrac = 1  # Volume fraction for constraints
    penal = 3  # Penalty for SIMP
    rmin = 1.732  # Filter radius

    # Initial solution
    x = volfrac * np.ones(nely * nelx * nelz, dtype=float)

    # Boundary conditions defining the loads and fixed points
    bc = FixedBeamBoundaryConditions(nelx, nely, nelz)

    # define force vector
    F = np.zeros((6, 1))
    F[1, 0] = 1  # 0: F_x, 1: F_y, 2: F_z, 3: M_x, 4: M_y, 5: M_z
    bc.set_forces(F)

    # is there a C_desired_y or C_desired_z?
    desired_y = 1
    desired_z = 0

    # values and load cases of constraints in list
    constraints = []
    constraints_f = []

    # Problem to optimize given objective and constraints
    topopt_filter = SensitivityBasedFilter(nelx, nely, nelz, rmin)


    # define 4 problems and solvers for 4 processes
    problem = MinMassProblem3(bc, penal, volfrac, topopt_filter, constraints, constraints_f, 0, 0, 1)
    problem.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization
    problem.desired_y = desired_y
    problem.desired_z = desired_z
    solver = TopOptSolver_Parallel(problem, 11)

    problem1 = MinMassProblem3(bc, penal, volfrac, topopt_filter, constraints, constraints_f, 0, 1, 1)
    problem1.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization
    problem1.desired_y = desired_y
    problem1.desired_z = desired_z
    solver1 = TopOptSolver_Parallel(problem1, 11)

    problem2 = MinMassProblem3(bc, penal, volfrac, topopt_filter, constraints, constraints_f, 0, 2, 1)
    problem2.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization
    problem2.desired_y = desired_y
    problem2.desired_z = desired_z
    solver2 = TopOptSolver_Parallel(problem2, 11)

    problem3 = MinMassProblem3(bc, penal, volfrac, topopt_filter, constraints, constraints_f, 0, 3, 1)
    problem3.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization
    problem3.desired_y = desired_y
    problem3.desired_z = desired_z
    solver3 = TopOptSolver_Parallel(problem3, 11)

    load = 0
    constraint_high = 71.25 # value of system level constraint

    # Compliance values around which the permissible range of compliance values is defined
    cxx1 = 2.5567
    cxy1 = 0.59863
    cyy1 = 94.235
    cxx2 = 23.341
    cxy2 = 1.5949
    cyy2 = 29.1353
    cxx3 = 9.6352
    cxy3 = 0.4671
    cyy3 = 3.9855


    delta = 1   # scaling range of permissible compliance values
    c_values = np.array([cxx1, cxy1, cyy1, cxx2, cxy2, cyy2, cxx3, cxy3, cyy3])
    c_delta_range = np.array([cxx1, np.sqrt(cxx1*cyy1), cyy1, cxx2, np.sqrt(cxx2*cyy2), cyy2, cxx3, np.sqrt(cxx3*cyy3), cyy3])*delta

    # PSO parameters
    num_particles = 8
    num_dimensions = 9
    max_iterations = 50
    w = 0.65  # Inertia weight
    c1 = 1  # Cognitive coefficient
    c2 = 1  # Social coefficient
    particle_history = np.zeros((max_iterations, num_particles, num_dimensions))  # all particle positions for all iterations

    # Initialize particles
    if load:
        mat_data = scipy.io.loadmat('PSO_K.mat')
        particles = mat_data['particles']
        velocity = mat_data['velocity']
        global_best_cost = mat_data['global_best_cost']
        global_best_position = mat_data['global_best_position']
        best_costs = mat_data['best_costs']
        best_positions = mat_data['best_positions']
    else:
        # Generate Halton Sequences
        uniform_cube = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1), chaospy.Uniform(0, 1),
                                 chaospy.Uniform(0, 1), chaospy.Uniform(0, 1), chaospy.Uniform(0, 1),
                                 chaospy.Uniform(0, 1), chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
        halton_samples = uniform_cube.sample(num_particles, rule="halton")
        halton_samples = halton_samples.T

        particles = np.zeros([num_particles, num_dimensions])
        # Scale the samples to your desired range
        particles = c_values - c_delta_range + halton_samples * 2 * c_delta_range
        velocity = (np.random.rand(num_particles, num_dimensions) - 0.5) * c_delta_range * 0.2
        exec(open('Project_Boundary_PSO.py').read())
        best_positions = particles.copy()
        best_costs = np.full(num_particles, np.inf)
        global_best_position = particles[0, :].copy()
        global_best_cost = np.inf

    # Main PSO loop
    for iteration in range(max_iterations):
        particle_history[iteration, :, :] = particles

        # split up particles for 4 seperate processes
        particles0 = particles[:int(num_particles/4), :]
        particles1 = particles[int(num_particles/4):int(num_particles * 2 / 4), :]
        particles2 = particles[int(num_particles * 2 / 4):int(num_particles * 3 / 4), :]
        particles3 = particles[int(num_particles * 3 / 4):, :]

        # define input data for parallelizer function
        input_data = [(particles0, problem, solver, x, constraint_high),
                      (particles1, problem1, solver1, x, constraint_high),
                      (particles2, problem2, solver2, x, constraint_high),
                      (particles3, problem3, solver3, x, constraint_high)]

        # Create a multiprocessing pool with 4 worker processes
        pool = multiprocessing.Pool(processes=4)

        # Use the starmap method to apply the function to each set of input arguments
        results = pool.starmap(parallelizer, input_data)

        # Close the pool to prevent further tasks from being submitted
        pool.close()

        # Wait for all worker processes to complete
        pool.join()
        cost = np.array(results).flatten()


        for i in range(num_particles):
            # Only update personal best and global best if the constraint is satisfied
            # Update personal best if necessary
            if cost[i] < best_costs[i]:
                best_costs[i] = cost[i]
                best_positions[i, :] = particles[i, :].copy()

            # Update global best if necessary
            if cost[i] < global_best_cost:
                global_best_cost = cost[i]
                global_best_position = particles[i, :].copy()
                print('Best Position!:', global_best_position)
                print(global_best_cost, '\n')

        for i in range(num_particles):
            # Update particle velocity and position
            velocity[i, :] = (
                    w * velocity[i, :]
                    + c1 * np.random.rand(num_dimensions) * (best_positions[i, :] - particles[i, :])
                    + c2 * np.random.rand(num_dimensions) * (global_best_position - particles[i, :])
            )
            particles[i, :] = particles[i, :] + velocity[i, :]

            # Make sure particle stays within defined limits
            for j in range(num_dimensions):
                if particles[i, j] < (c_values[j] - c_delta_range[j]):
                    particles[i, j] = (c_values[j] - c_delta_range[j])
                    velocity[i, j] = 0
                if particles[i, j] > (c_values[j] + c_delta_range[j]):
                    particles[i, j] = (c_values[j] + c_delta_range[j])
                    velocity[i, j] = 0

        # Results
        print('Optimal solution found in:', iteration)
        print('x:', global_best_position)
        print('Cost:', global_best_cost)
        print('Cost:', best_costs, '\n')

        exec(open('Project_Boundary_PSO.py').read())

    scipy.io.savemat('PSO.mat', {'particle_history': particle_history})
