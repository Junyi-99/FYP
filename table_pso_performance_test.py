import numpy as np


def fitness(x) -> float:
    return abs((100.0 - x[0][0]) ** 5 + (50.0 - x[0][1]) ** 5 + (25.0 - x[0][2]) ** 5)


class Particle():
    def __init__(self,
                 position_min, position_max,
                 velocity_min, velocity_max,
                 dimension):
        self.__position = np.random.uniform(position_min, position_max, (1, dimension));
        self.__velocity = np.random.uniform(velocity_min, velocity_max, (1, dimension))
        self.__best_position = np.zeros((1, dimension))
        self.__fitness = fitness(self.__position)

    def get_best_position(self):
        return self.__best_position

    def set_position(self, position):
        self.__position = position

    def get_position(self):
        return self.__position

    def set_velocity(self, velocity):
        self.__velocity = velocity

    def get_velocity(self):
        return self.__velocity

    def set_fitness(self, fitness: float):
        self.__fitness = fitness

    def get_fitness(self):
        return self.__fitness

    def update_fitness(self):
        fit = fitness(self.__position)
        if fit < self.__fitness:
            self.__fitness = fit
            self.__best_position = self.__position

    def update_position(self):
        pos = self.__position + self.__velocity
        self.__position = pos


# Global Best PSO
class PSO():
    def __init__(
            self,

            n_particles: int,  # 粒子数量
            n_dimensions: int,  # 维度

            w=0.5,
            c1=1.4,
            c2=1.4,
    ):
        self.w = w
        self.c1 = c1
        self.c2 = c2

        self.__list_particle = [Particle(
            1, 512, -1, 1, n_dimensions
        ) for i in range(n_particles)]

        self.best_fitness = 99999999999999999.0
        self.best_position = np.zeros(n_dimensions)

    def __update_velocity(self, particle: Particle):
        v = self.w * particle.get_velocity() + \
            self.c1 * np.random.rand() * (particle.get_best_position() - particle.get_position()) + \
            self.c2 * np.random.rand() * (self.best_position - particle.get_position())
        particle.set_velocity(v)

    def __update_position(self, particle: Particle):
        particle.update_position()
        particle.update_fitness()
        fit = particle.get_fitness()
        if fit < self.best_fitness:
            self.best_fitness = fit
            self.best_position = particle.get_position()

    def optimize(self, n_iter_max: int):
        for i in range(n_iter_max):
            for particle in self.__list_particle:
                self.__update_velocity(particle)
                self.__update_position(particle)

        return (self.best_fitness, self.best_position[0])


def rosenbrock_with_args(x):
    particles = x.shape[0]
    j = [abs((100.0 - x[i][0]) ** 5 + (50.0 - x[i][1]) ** 5 + (25.0 - x[i][2]) ** 5) for i in range(particles)]
    return np.array(j)


class NPSO():
    def __init__(self,
                 n_particles: int,
                 n_dimension: int):
        self.n_particles = n_particles
        self.particles_velocity = np.random.uniform(-1, 1, (n_particles, n_dimension))
        self.particles_position = np.random.uniform(1, 512, (n_particles, n_dimension))
        self.particles_position_best = self.particles_position[:]

        self.particles_fitness_best = np.array([99999999999999999 for i in range(n_particles)])
        self.particles_fitness_current = np.array([99999999999999999 for i in range(n_particles)])

        self.global_position_best = [0, 0, 0]
        self.global_fitness_best = 99999999999999999.0

    def update_current_fitness(self):
        self.particles_fitness_current = rosenbrock_with_args(self.particles_position)

    def update_particle_best(self):
        for i in range(self.n_particles):
            if (self.particles_fitness_current[i] < self.particles_fitness_best[i]):
                self.particles_fitness_best[i] = self.particles_fitness_current[i]
                self.particles_position_best[i] = self.particles_position[i]

    def update_global_best(self):
        for i in range(self.n_particles):
            if (self.particles_fitness_best[i] < self.global_fitness_best):
                self.global_fitness_best = self.particles_fitness_best[i]
                self.global_position_best = self.particles_position_best[i]

    def update_position(self):
        pos = self.particles_position + self.particles_velocity
        self.particles_position = pos

    def update_velocity(self):
        c1 = 1.4
        c2 = 1.4
        w = 0.5
        shape = self.particles_position.shape
        cognitive = c1 * np.random.rand() * (
                self.particles_position_best - self.particles_position)

        social = c2 * np.random.rand() * (
                self.global_position_best - self.particles_position)

        velo = w * self.particles_velocity + cognitive + social
        self.particles_velocity = velo

    def optimize(self, n_iters):
        for i in range(n_iters):
            self.update_velocity()
            self.update_position()
            self.update_current_fitness()


            self.update_particle_best()
            self.update_global_best()

        return (self.global_fitness_best, self.global_position_best)


def f(a, b, c):
    return abs((100.0 - a) ** 5 + (50.0 - b) ** 5 + (25.0 - c) ** 5)


if __name__ == '__main__':

    lgr = 0
    suml = 0
    sumr = 0
    
    print("impl1_worse_cases, impl2_worse_cases, impl1_fitness, impl2_fitness")
    for i in range(1000):
        p = PSO(50, 3) # discrete particle
        a = p.optimize(100)

        n = NPSO(50, 3) # batch particle
        b = n.optimize(100)

        l = f(a[1][0], a[1][1], a[1][2]) # discrete particle
        r = f(b[1][0], b[1][1], b[1][2]) # batch particle
        if l > r: # left 不如 right 的结果好
            lgr = lgr + 1
        # 结果越小学好
        suml += l
        sumr += r
        print(lgr,",", i - lgr + 1, ",", l,",", r)
    print(suml/1000, sumr/1000)

# python3 pso_performance_test.py > experiments.csv