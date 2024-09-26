from functools import total_ordering
from .dist_eval import eval_net_id
from math import exp
from copy import copy
from random import uniform as rand, randrange
import ray

@total_ordering
class Particle:
    def __init__(self, dimension:int) -> None:
        self._dimension = dimension
        self.x = [0.0 for _ in range(self._dimension)]
        self.v = [0.0 for _ in range(self._dimension)]
        self.fitness = float('inf')
        self.cost = float('inf')
        self.local_fitness = float('inf')
    
    def __lt__(self, __o: "Particle") -> bool:
        return self.fitness < __o.fitness
    
    def __eq__(self, __o: "Particle") -> bool:
        return self.fitness == __o.fitness

class LLSO:
    def __init__(
        self,
        net_id:int,
        x_upper_bound:list[float],
        x_lower_bound:list[float],
        swarm_size:int,
        mode:str
    ) -> None:

        self.net_id = net_id

        self.swarm_size = swarm_size
        self.x_upper_bound = copy(x_upper_bound)
        self.x_lower_bound = copy(x_lower_bound)
        self._dimension = len(self.x_upper_bound)

        self.phi = 0.5

        self.rand_level_set = None
        self.level_size_performance = None
        self.__select_level_idx = None
        self.__number_of_level = None
        self.__level_size = None

        self.best_fitness = float('inf')
        self.selected_particle = None
        self.selected_p_idx = None
        self.mode = mode

        self.__init()
    

    def __init(self) -> None:
        if self.swarm_size >= 300:
            self.rand_level_set = [4, 6, 8, 10, 20, 50]
        elif self.swarm_size >= 20:
            self.rand_level_set = [4, 6, 8, 10]
        else:
            self.rand_level_set = [2, 3]
        self.level_size_performance = [1.0 for _ in range(len(self.rand_level_set))]

        self.swarm = [Particle(self._dimension) for _ in range(self.swarm_size)]
        for p in self.swarm:
            for d in range(self._dimension):
                p.x[d] = rand(self.x_lower_bound[d], self.x_upper_bound[d])
        futures = [eval_net_id.remote(self.net_id, p.x) for p in self.swarm]
        result = ray.get(futures)
        for idx, t in enumerate(result):
            _, cost, fit = t
            self.swarm[idx].cost = cost
            self.swarm[idx].local_fitness = fit
            self.swarm[idx].fitness = fit
        
        self.swarm.sort()
        self.best_fitness = self.swarm[0].fitness

    def update_selected_particle(self) -> None:
        match self.mode:
            case 'global_best':
                self.selected_p_idx = 0
                self.selected_particle = self.swarm[0]
            case 'top_level_random':
                self.selected_p_idx = randrange(0, self.__level_size)
                self.selected_particle = self.swarm[self.selected_p_idx]
            case _:
                self.selected_particle = None
    
    def select_level_size(self):
        total = 0.0
        for i in range(len(self.rand_level_set)):
            total += exp(7 * self.level_size_performance[i])
        p = [0.0 for _ in range(len(self.rand_level_set) + 1)]
        for i in range(len(self.rand_level_set)):
            p[i + 1] = p[i] + exp(7 * self.level_size_performance[i]) / total
        tmp = rand(0.0, 1.0)
        selected = -1
        for i in range(len(self.rand_level_set)):
            if tmp <= p[i + 1]:
                selected = i
                break
        if selected == -1:
            raise ValueError()
        return selected
    
    def level(self):
        self.__select_level_idx = self.select_level_size()
        self.__number_of_level = self.rand_level_set[self.__select_level_idx]
        self.__level_size = self.swarm_size // self.__number_of_level
    
    def evolve_particle(self):
        self.level()
        self.update_selected_particle()
        for level_index in range(self.__number_of_level - 1, 0, -1):
            number_of_particle = self.__level_size
            if level_index == self.__number_of_level - 1:
                number_of_particle += self.swarm_size % self.__number_of_level
        
            for p_index in range(number_of_particle):
                p_cur = level_index * self.__level_size + p_index
                p_1:int = None
                p_2:int = None
                if level_index >= 2:
                    rl1 = randrange(level_index)
                    rl2 = randrange(level_index)
                    while(rl1 == rl2):
                        rl2 = randrange(level_index)
                    if rl1 > rl2:
                        rl1, rl2 = rl2, rl1
                    p_1 = self.__level_size * rl1 + randrange(self.__level_size)
                    p_2 = self.__level_size * rl2 + randrange(self.__level_size)
                elif level_index == 1:
                    p_1 = randrange(self.__level_size)
                    p_2 = randrange(self.__level_size)
                    while p_1 == p_2:
                        p_2 = randrange(self.__level_size)
                    if self.swarm[p_2].fitness < self.swarm[p_1].fitness:
                        p_1, p_2 = p_2, p_1
                
                for d in range(self._dimension):
                    r1 = rand(0.0, 1.0)
                    r2 = rand(0.0, 1.0)
                    r3 = rand(0.0, 1.0)
                    self.swarm[p_cur].v[d] = r1 * self.swarm[p_cur].v[d] \
                        + r2 * (self.swarm[p_1].x[d] - self.swarm[p_cur].x[d]) \
                            + r3 * self.phi * (self.swarm[p_2].x[d] - self.swarm[p_cur].x[d])
                    self.swarm[p_cur].x[d] += self.swarm[p_cur].v[d]
                    self.swarm[p_cur].x[d] = max(self.x_lower_bound[d], min(self.swarm[p_cur].x[d], self.x_upper_bound[d]))

    def update(self) -> None:
        self.swarm.sort()
        # update level_size_performance
        if self.best_fitness > self.swarm[0].fitness:
            self.level_size_performance[self.__select_level_idx] = abs(self.best_fitness - self.swarm[0].fitness) / abs(self.best_fitness)
        else:
            self.level_size_performance[self.__select_level_idx] = 0
        self.best_fitness = self.swarm[0].fitness

    def cur_level_size(self) -> int:
        return self.__level_size