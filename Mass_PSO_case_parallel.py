import numpy as np
from topopt.boundary_conditions import FixedBeamBoundaryConditions
from topopt.problems import ComplianceProblem3
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
def constraint_function(v1, v2, v3, constraint, problem, solver):
    # Component 1
    problem.length_x = 0.5
    problem.build_indices()
    problem.volfrac = v1
    x = v1 * np.ones(problem.nel, dtype=float)
    F = np.zeros((6, 1))
    F[1, 0] = 1  # 0: F_x, 1: F_y, 2: F_z, 3: M_x, 4: M_y, 5: M_z || 0: F_y, 1: M_z (2D)
    problem.f[0:6] = F
    problem.iter = 0
    x_opt_1, obj_1 = solver.optimize(x)
    # print('obj_1', obj_1)

    # Component 2
    problem.length_x = 2.0
    problem.build_indices()
    problem.volfrac = v2
    x = v2 * np.ones(problem.nel, dtype=float)
    F = np.zeros((6, 1))
    F[1, 0] = 1
    F[5, 0] = 0.5
    problem.f[0:6] = F
    problem.iter = 0
    x_opt_2, obj_2 = solver.optimize(x)
    # print('obj_2', obj_2)

    # Component 3
    problem.length_x = 1.0
    problem.build_indices()
    problem.volfrac = v3
    x = v3 * np.ones(problem.nel, dtype=float)
    F = np.zeros((6, 1))
    F[1, 0] = 1
    F[5, 0] = 2.5
    problem.f[0:6] = F
    problem.iter = 0
    x_opt_3, obj_3 = solver.optimize(x)
    # print('obj_3', obj_3)

    obj_ges = obj_1 + obj_2 + obj_3

    # print('Displacement:', obj_ges)

    if obj_ges > constraint:
        # print('unfeasible design! \n')
        return 1
    else:
        # print('feasible design \n')
        return 0



# Define the cost function f(x, y) that you want to minimize
def cost_function(x, y, z):
    return (x * 0.5 + 2 * y + z) / 3.5

def parallelizer(particles, problem, solver, best_costs, unfeasible_designs):
    num_particles = len(particles)
    constraint_values = np.ones(num_particles)
    costs = np.ones(num_particles)
    for i in range(num_particles):
        # Evaluate cost function
        costs[i] = cost_function(particles[i, 0], particles[i, 1], particles[i, 2])

        # Evaluate constraint function
        if costs[i] > best_costs[i] or np.any(np.all(np.stack((particles[i, 0] < np.array(unfeasible_designs)[:, 0], particles[i, 1] < np.array(unfeasible_designs)[:, 1], particles[i, 2] < np.array(unfeasible_designs)[:, 2]), axis=-1), axis=1)):
            constraint_values[i] = 2    # save unnecessary evaluation by not evaluating particles with higher cost or obviously unfeasible masses
        else:
            constraint_values[i] = constraint_function(particles[i, 0], particles[i, 1], particles[i, 2], 71.25, problem, solver)
    return constraint_values, costs


if __name__ == "__main__":
    nelx, nely, nelz = 16, 16, 4
    penal = 3  # Penalty for SIMP
    rmin = 1.732

    # Boundary conditions defining the loads and fixed points
    bc = FixedBeamBoundaryConditions(nelx, nely, nelz)

    # define force vector
    F = np.zeros((6, 1))
    bc.set_forces(F)

    constraints = []
    constraints_f = []

    topopt_filter = SensitivityBasedFilter(nelx, nely, nelz, rmin)

    # define 4 problems and solvers for 4 processes
    problem = ComplianceProblem3(bc, penal, 0.2, topopt_filter, constraints, constraints_f, 0, 0, 0.5)
    problem.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization
    problem.active = np.concatenate((np.arange(64), np.arange(64) + 64 * 15))
    solver = OCSolver(problem, len(constraints))

    problem1 = ComplianceProblem3(bc, penal, 0.2, topopt_filter, constraints, constraints_f, 0, 1, 0.5)
    problem1.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization
    problem1.active = np.concatenate((np.arange(64), np.arange(64) + 64 * 15))
    solver1 = OCSolver(problem1, len(constraints))

    problem2 = ComplianceProblem3(bc, penal, 0.2, topopt_filter, constraints, constraints_f, 0, 2, 0.5)
    problem2.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization
    problem2.active = np.concatenate((np.arange(64), np.arange(64) + 64 * 15))
    solver2 = OCSolver(problem2, len(constraints))

    problem3 = ComplianceProblem3(bc, penal, 0.2, topopt_filter, constraints, constraints_f, 0, 3, 0.5)
    problem3.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization
    problem3.active = np.concatenate((np.arange(64), np.arange(64) + 64 * 15))
    solver3 = OCSolver(problem3, len(constraints))

    problem4 = ComplianceProblem3(bc, penal, 0.2, topopt_filter, constraints, constraints_f, 0, 4, 0.5)
    problem4.reducedofs = 1  # delete dofs of elements that are close to zero in density, speeding up optimization
    problem4.active = np.concatenate((np.arange(64), np.arange(64) + 64 * 15))
    solver4 = OCSolver(problem4, len(constraints))

    # PSO parameters
    num_particles = 10
    num_dimensions = 3
    max_iterations = 50
    w = 0.65  # Inertia weight
    c1 = 1  # Cognitive coefficient
    c2 = 1  # Social coefficient

    particle_history = np.zeros((max_iterations, num_particles, num_dimensions))  # all particle positions for all iterations

    load = 0

    # Initialize particles
    # Generate Halton Sequences
    uniform_cube = chaospy.J(chaospy.Uniform(0, 1), chaospy.Uniform(0, 1), chaospy.Uniform(0, 1))
    halton_samples = uniform_cube.sample(num_particles, rule="halton")
    halton_samples = halton_samples.T

    particles = np.zeros([num_particles, num_dimensions])

    # Scale the samples to your desired range
    delta = 0.5
    c_values = np.array([0.5, 0.5, 0.5])
    c_delta_range = c_values * delta

    if load:
        mat_data = scipy.io.loadmat('PSO.mat')
        particles = mat_data['particles']
        velocity = mat_data['velocity']
    else:
        particles = c_values - c_delta_range + halton_samples * 2 * c_delta_range
        velocity = np.random.rand(num_particles, num_dimensions) * delta * 0.2


    best_positions = particles.copy()
    best_costs = np.full(num_particles, np.inf)
    global_best_position = particles[0, :].copy()
    global_best_cost = np.inf
    feasible_designs = []
    unfeasible_designs = []
    unfeasible_designs.append(np.array([0, 0, 0]))

    # Main PSO loop
    for iteration in range(max_iterations):
        particle_history[iteration, :, :] = particles

        # split up particles for 4 seperate processes
        particles0 = particles[:int(num_particles/5), :]
        particles1 = particles[int(num_particles/5):int(num_particles * 2 / 5), :]
        particles2 = particles[int(num_particles * 2 / 5):int(num_particles * 3 / 5), :]
        particles3 = particles[int(num_particles * 3 / 5):int(num_particles * 4 / 5), :]
        particles4 = particles[int(num_particles * 4 / 5):, :]

        best_costs0 = best_costs[:int(num_particles/5)]
        best_costs1 = best_costs[int(num_particles/5):int(num_particles * 2 / 5)]
        best_costs2 = best_costs[int(num_particles * 2 / 5):int(num_particles * 3 / 5)]
        best_costs3 = best_costs[int(num_particles * 3 / 5):int(num_particles * 4 / 5)]
        best_costs4 = best_costs[int(num_particles * 4 / 5):]

        # define input data for parallelizer function
        input_data = [(particles0, problem, solver, best_costs0, unfeasible_designs),
                      (particles1, problem1, solver1, best_costs1, unfeasible_designs),
                      (particles2, problem2, solver2, best_costs2, unfeasible_designs),
                      (particles3, problem3, solver3, best_costs3, unfeasible_designs),
                      (particles4, problem4, solver4, best_costs4, unfeasible_designs)]

        # Create a multiprocessing pool with 4 worker processes
        pool = multiprocessing.Pool(processes=5)

        # Use the starmap method to apply the function to each set of input arguments
        results = pool.starmap(parallelizer, input_data)

        # Close the pool to prevent further tasks from being submitted
        pool.close()

        # Wait for all worker processes to complete
        pool.join()

        constraint_values = np.array(results)[:, 0, :].flatten()
        costs = np.array(results)[:, 1, :].flatten()

        for i in range(num_particles):
            # Only update personal best and global best if the constraint is satisfied
            if constraint_values[i] == 0:
                feasible_designs.append(particles[i, :].copy())
                # Update personal best if necessary
                if costs[i] < best_costs[i]:
                    best_costs[i] = costs[i]
                    best_positions[i, :] = particles[i, :].copy()

                # Update global best if necessary
                if costs[i] < global_best_cost:
                    global_best_cost = costs[i]
                    global_best_position = particles[i, :].copy()
                    print('Best Position!:', global_best_position)
                    print(global_best_cost)
            elif constraint_values[i] == 1:
                unfeasible_designs.append(particles[i, :].copy())

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
        print(global_best_position)
        print('Cost:', global_best_cost)
        print('Cost:', best_costs)

    scipy.io.savemat('PSO.mat', {'particle_history': particle_history})
