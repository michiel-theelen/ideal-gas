import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button

class ParticleSystem:
    def __init__(self, N, box_size, radius=2, mass=1):
        self.N = N
        self.box_size = box_size
        self.radius = radius
        self.mass = mass
        self.positions = np.random.rand(N, 2) * box_size
        self.velocities = np.zeros((N, 2))
        self.velocities[:, 0] = np.random.randn(N) * 10
        self.grid_size = max(radius, box_size / 100)

    def step(self, dt):
        # Update positions
        self.positions += self.velocities * dt

        # Boundary collision
        for i in range(self.N):
            if self.positions[i, 0] < self.radius:
                self.velocities[i, 0] *= -1
                self.positions[i, 0] = self.radius
            elif self.positions[i, 0] > self.box_size - self.radius:
                self.velocities[i, 0] *= -1
                self.positions[i, 0] = self.box_size - self.radius

            if self.positions[i, 1] < self.radius:
                self.velocities[i, 1] *= -1
                self.positions[i, 1] = self.radius
            elif self.positions[i, 1] > self.box_size - self.radius:
                self.velocities[i, 1] *= -1
                self.positions[i, 1] = self.box_size - self.radius

        # Spatial partitioning for collision detection
        grid = {}
        for i in range(self.N):
            key = (int(self.positions[i, 0] / self.grid_size), int(self.positions[i, 1] / self.grid_size))
            grid.setdefault(key, []).append(i)

        # Check for collisions within each grid cell
        for key in grid:
            indices = grid[key] 
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx1, idx2 = indices[i], indices[j]
                    if np.linalg.norm(self.positions[idx1] - self.positions[idx2]) < 2 * self.radius:
                        self.velocities[idx1], self.velocities[idx2] = self.resolve_collision(idx1, idx2)

    def resolve_collision(self, i, j):
        v1, v2 = self.velocities[i], self.velocities[j]
        p1, p2 = self.positions[i], self.positions[j]
        dist = np.linalg.norm(p1 - p2)
        if dist == 0:  # Avoid division by zero
            return v1, v2
        unit_vector = (p2 - p1) / dist
        v1_new = v1 - 2 * self.mass / (self.mass * 2) * np.dot(v1 - v2, unit_vector) * unit_vector
        v2_new = v2 - 2 * self.mass / (self.mass * 2) * np.dot(v2 - v1, -unit_vector) * (-unit_vector)
        return v1_new, v2_new
    

def calculate_speeds(velocities):
    return np.sqrt(np.sum(velocities**2, axis=1))

def calculate_temperature(velocities, mass):
    kinetic_energy_per_particle = 0.5 * mass * np.mean(np.sum(velocities**2, axis=1))
    # Assuming k_B (Boltzmann constant) = 1 for simplicity
    # You may adjust this to use actual physical units if necessary
    return 2 * kinetic_energy_per_particle / 2

def maxwell_boltzmann_distribution(speeds, mass, T):
    # Adjust the scale parameter for 2D
    scale = np.sqrt(T/mass)
    # The 2D Maxwell-Boltzmann distribution in terms of speed is a Rayleigh distribution
    return speeds * np.exp(-speeds**2 / (2 * scale**2)) / (scale**2)



# Parameters
N = 1000  # Number of particles
box_size = 200  # Size of the box
dt = 0.05  #vel Time step
tracked_particle_index = 0  # Index of the particle to track


# Create particle system
system = ParticleSystem(N, box_size)

# Visualization setup
fig, (ax_particles, ax_speed_dist) = plt.subplots(1, 2, figsize=(12, 6))

# Setup particle animation
particles, = ax_particles.plot([], [], 'bo', ms=2)
tracked_particle, = ax_particles.plot([], [], 'ro', ms=2)  # Tracked particle in red
highlight_circle = plt.Circle((0, 0), 10, color='red', alpha=0.4, fill=True)  # Semi-transparent circle
ax_particles.add_patch(highlight_circle)  # Add the circle to the plot


ax_particles.set_xlim(0, box_size)
ax_particles.set_ylim(0, box_size)
ax_particles.set_aspect('equal')
ax_particles.set_title("Particle System")

# Setup speed distribution plot
ax_speed_dist.set_xlim(0, 50)
ax_speed_dist.set_ylim(0, 0.4)
ax_speed_dist.set_xlabel("Speed")
ax_speed_dist.set_ylabel("Frequency")
ax_speed_dist.set_title("Speed Distribution")

# Precompute Maxwell-Boltzmann distribution
T = 1  # Example temperature value
mb_speeds = np.linspace(0, 50, 400)
mb_dist = maxwell_boltzmann_distribution(mb_speeds, system.mass, T)

def init():
    particles.set_data([], [])
    return particles,

def animate(frame):
    system.step(dt)
    particles.set_data(system.positions[:, 0], system.positions[:, 1])
    tracked_particle.set_data(system.positions[0, 0], system.positions[0, 1])
    
    # Update the position of the semi-transparent circle
    highlight_circle.center = (system.positions[tracked_particle_index, 0], system.positions[tracked_particle_index, 1])


    # Update speed distribution data less frequently
    speeds = calculate_speeds(system.velocities)
    T = calculate_temperature(system.velocities, system.mass)
    mb_dist = maxwell_boltzmann_distribution(mb_speeds, system.mass, T)

    # Remove the axis ticks and labels from the particle plot
    ax_particles.set_xticks([])
    ax_particles.set_yticks([])
    ax_particles.set_xticklabels([])
    ax_particles.set_yticklabels([])

    ax_speed_dist.clear()
    ax_speed_dist.hist(speeds, bins=40, density=True, alpha=0.7, color='blue')
    ax_speed_dist.set_ylim(0, max(mb_dist)*1.2)
    ax_speed_dist.set_xlim(0, 40)
    ax_speed_dist.plot(mb_speeds, mb_dist, 'r-', label="Maxwell-Boltzmann")
    ax_speed_dist.legend()
    
    ax_speed_dist.set_title("Speed Distribution")  # Re-set the title here
    ax_speed_dist.set_xlabel("Speed")
    ax_speed_dist.set_ylabel("Frequency")

    # Ensure that the plots are the same size and well positioned
    plt.tight_layout()

    return particles,


# Create animation
anim = FuncAnimation(fig, animate, init_func=init, frames=200, interval=10, blit=False)

plt.show()
