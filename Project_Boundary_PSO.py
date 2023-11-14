import numpy as np
import scipy

def constraint_function2(particle, constraint_high):
    K_1 = np.linalg.inv(np.array([[particle[0], particle[1]], [particle[1], particle[2]]]))
    K_2 = np.linalg.inv(np.array([[particle[3], particle[4]], [particle[4], particle[5]]]))
    K_3 = np.linalg.inv(np.array([[particle[6], particle[7]], [particle[7], particle[8]]]))

    length_x_1 = 0.5
    length_x_2 = 2
    length_x_3 = 1

    K_4x4_1 = compute_four_by_four(K_1, length_x_1)
    K_4x4_2 = compute_four_by_four(K_2, length_x_2)
    K_4x4_3 = compute_four_by_four(K_3, length_x_3)

    K_global = np.zeros((8, 8))
    K_global[:4, :4] = K_global[:4, :4] + K_4x4_3
    K_global[2:6, 2:6] = K_global[2:6, 2:6] + K_4x4_2
    K_global[4:, 4:] = K_global[4:, 4:] + K_4x4_1

    K_global = K_global[2:, 2:]

    F = np.zeros((6, 1))
    F[4, 0] = 1
    U = np.linalg.solve(K_global, F)

    if np.abs(U[4, 0]) > constraint_high:
        constraint_value = 1
    else:
        constraint_value = 0

    cost = np.abs(U[4, 0] - constraint_high)
    return constraint_value, cost

constraint_high_orig = constraint_high
constraint_high = constraint_high - 0.05

max_iter = 200

# Initialize particles
particles_orig = particles.copy()
velocity_orig = velocity.copy()
velocity = velocity * 0

best_positions_2 = particles.copy()
best_costs_2 = np.full(num_particles, np.inf)

# Main PSO loop
for iteration_2 in range(max_iter):
    for i in range(num_particles):
        # Evaluate cost function

        constraint_value, cost = constraint_function2(particles[i, :], constraint_high)

        if iteration_2 < max_iter - 1:
            cost = cost + 1 * np.linalg.norm((particles[i, :] - particles_orig[i, :]) / c_delta_range)
        # Update personal best if necessary
        if cost < best_costs_2[i]:
            best_costs_2[i] = cost
            best_positions_2[i, :] = particles[i, :].copy()

    for i in range(num_particles):
        # Update particle velocity and position
        velocity[i, :] = (
                + 1.75 * np.random.rand(num_dimensions) * (best_positions_2[i, :] - particles[i, :])
                + (np.random.rand(num_dimensions) - 0.5) * c_delta_range * (1 - iteration_2/max_iter) * 0.1
        )
        particles[i, :] = particles[i, :] + velocity[i, :]

velocity = velocity_orig.copy()
constraint_high = constraint_high_orig
