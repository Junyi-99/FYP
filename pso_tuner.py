import numpy as np
from .tuner import Tuner

"""
THIS FILE SHOUD BE PLACED AT 'tvm/python/tvm/autotvm/tuner'

AND YOU SHOULD ALSO ADD A LINE 'from .pso_tuner import PSOTuner'
IN 'tvm/python/tvm/autotvm/tuner/__init__.py'

"""

def point2knob(p, dims):
    """convert point form (single integer) to knob form (vector)"""
    knob = []
    for dim in dims:
        knob.append(p % dim)
        p //= dim
    return knob

def knob2point(knob, dims):
    """convert knob form (vector) to point form (single integer)"""
    
    p = 0
    for j, k in enumerate(knob):
        p += int(np.prod(dims[:j])) * k
    #print("knob2point", knob, p)
    return p

class Particle():
    def __randint(self, low, high):
        if low >= high:
            return low
        return np.random.randint(low, high)
    def __fun_fitness(self, x) -> float:
        tile_f = x[0]
        tile_y = x[1]
        tile_x = x[2]
        tile_rc = x[3]
        tile_ry = x[4]
        tile_rx = x[5]
        auto_unroll_max_step = x[6]
        unroll_explicit = x[7]
        """
         tile_f 220
         tile_y 4
         tile_x 4
         tile_rc 10
         tile_ry 2
         tile_rx 2
         auto_unroll_max_step 3
         unroll_explicit 2
         """
        # Best 128, 4, 4, 8, 2, 1, 3, 1
        # Seco 64, 2, 4, 8, 2, 1, 3, 1
        fit = (128 - tile_f) ** 2 + (4 - tile_y) ** 2 + (4 - tile_x) ** 2 + (8 - tile_rc) ** 2 + (2 - tile_ry) ** 2 + \
              (1 - tile_rx) ** 2 + (3 - auto_unroll_max_step) ** 2 + (1 - unroll_explicit) ** 2

        return fit

    def __init__(self,
                 position_range,
                 velocity_range,
                 dimension,
                 w,
                 c1,
                 c2,
                 ):
        self.__w = w
        self.__c1 = c1
        self.__c2 = c2
        self.__position_range = position_range[:]
        self.__velocity_range = velocity_range[:]
        self.__dimension = dimension
        self.__position = np.array([self.__randint(low, high) for low, high in position_range])
        self.__velocity = np.array([self.__randint(low, high) for low, high in velocity_range])
        self.__best_position = np.random.randint((dimension,))
        self.__fitness = 0.0

    def get_best_position(self):
        return self.__best_position

    def set_position(self, pos):
        for index in range(self.__dimension):
            if pos[index] < self.__position_range[index][0]:
                self.__position[index] = self.__position_range[index][0]
            elif pos[index] > self.__position_range[index][1]:
                self.__position[index] = self.__position_range[index][1]
            else:
                self.__position[index] = pos[index]
    
    def random_position(self):
        self.__position = np.array([self.__randint(low, high) for low, high in self.__position_range])
        self.__best_position = np.random.randint((self.__dimension,))
        self.__fitness = 0.0

    def get_position(self):
        return self.__position[:]

    def set_velocity(self, vel):
        for index in range(self.__dimension):
            if vel[index] < self.__velocity_range[index][0]:
                self.__velocity[index] = self.__velocity_range[index][0]
            elif vel[index] > self.__velocity_range[index][1]:
                self.__velocity[index] = self.__velocity_range[index][1]
            else:
                self.__velocity[index] = vel[index]

    def get_velocity(self):
        return self.__velocity[:]

    def get_fitness(self) -> float:
        return self.__fitness

    def update_velocity(self, global_best_position):
        v = self.__w * self.__velocity + \
            self.__c1 * np.random.rand() * (self.__best_position - self.__position) + \
            self.__c2 * np.random.rand() * (global_best_position - self.__position)
        v = np.floor(v)
        self.set_velocity(v)

    # fit is flops
    def update_fitness(self, fit_flops = None) -> float:
        if fit_flops is None:
            fit = self.__fun_fitness(self.__position)
        else:
            fit = fit_flops
        
        if fit > self.__fitness:
            self.__fitness = fit
            self.__best_position = self.__position
        return self.__fitness

    def update_position(self):
        pos = self.__position + self.__velocity
        self.set_position(pos)



class PSOTuner(Tuner):
    def __init__(self, task,
                 n_particles: int = 50,  # 粒子数量
                 n_iter_max: int = 20,
                 w: float = 0.5,
                 c1: float = 1.4,
                 c2: float = 1.4):
        super(PSOTuner, self).__init__(task)

        self.kv = [(k, v) for k, v in self.task.config_space.space_map.items()]
        self.position_range = []
        self.velocity_range = []
        self.n_dimensions = len(self.kv)  # 获取 ConfigSpace 的参数数量
        self.dims = [] # 用来记录每一个 knob 的参数数量（维度长度）
        self.xs = []
        self.ys = []
        self.flops_max = 0

        #print("PSOTuner kv: ", self.kv)
        for key, value in self.kv:
            entities = value.entities  # type() == list
            self.position_range.append((0, len(entities)-1))
            self.velocity_range.append((-100, 100))
            self.dims.append(len(value))
        print("Dimensions:", self.dims)
        self.__list_particle = [Particle(
            self.position_range,
            self.velocity_range,
            self.n_dimensions,
            w, c1, c2
        ) for _ in range(n_particles)]

        self.n_particles = n_particles
        self.n_iter_max = n_iter_max
        self.n_iter_cur = 0

        self.best_fitness = 0.0
        self.best_position = np.zeros(self.n_dimensions)
        self.visited = []
    # batch_size should equals n_particle
    def next_batch(self, batch_size):
        ret = []
        for i in range(self.n_particles):
            particle = self.__list_particle[i]
            index = knob2point(particle.get_position(), self.dims)

            while index in self.visited: # 为了解决重复访问的问题
                particle.random_position()
                index = knob2point(particle.get_position(), self.dims)

            ret.append(self.task.config_space.get(index))
            self.visited.append(index)
        self.n_iter_cur = self.n_iter_cur + 1
        print("\n\nIteration: % 4d, BatchSize: % 4d"%(self.n_iter_cur, len(ret)))
        return ret

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            index = inp.config.index
            if res.error_no == 0:
                flops = inp.task.flop / np.mean(res.costs)
            else:
                flops = 0.0
            self.xs.append(index)
            self.ys.append(flops)
            self.flops_max = max(self.flops_max, flops)
            
            for i in range(self.n_particles):
                particle = self.__list_particle[i]
                pIndex = knob2point(self.__list_particle[i].get_position(), self.dims)
                
                # 找到是哪个 particle 的 index 是当前的 (input,result)
                if pIndex == index:
                    
                    fit = particle.update_fitness(flops)
                    print("Iteration: % 4d, Particle: % 4d, current flops: % 15.4f, best flops: % 15.4f, config: " % (self.n_iter_cur, i+1, flops, fit), particle.get_position())

                    if fit > self.best_fitness:
                        self.best_fitness = fit
                        self.best_position = np.copy(particle.get_position())
                        print("Iteration: % 4d, Particle: % 4d, fitness(flops): % 15.4f" % (self.n_iter_cur, i + 1, self.best_fitness,), self.best_position)
                    particle.update_velocity(self.best_position)
                    particle.update_position()
                    break
            
    def has_next(self):
        return self.n_iter_cur < self.n_iter_max

    def load_history(self, data_set):
        pass


if __name__ == '__main__':
    b = PSO(100, 8, 0.9, 0.5, 0.3)
    fitness, position = b.optimize(500)
    print("%.4f" % (fitness,), position)


