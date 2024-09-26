from pandapower import converter, pandapowerNet
import pandapower as pp
import numpy as np
from abc import ABC, abstractmethod
from math import sin
import pickle

# def load_net_9(file_path:str="./power_case/case9_1.m") -> pandapowerNet:
#     net = converter.from_mpc(file_path)
#     net.bus.max_vm_pu = np.array([1.10 for _ in range(len(net.bus))])
#     net.bus.min_vm_pu = np.array([0.9  for _ in range(len(net.bus))])
#     return net

# def load_net_30(file_path:str="./power_case/case_ieee30_1.m") -> pandapowerNet:
#     net = converter.from_mpc(file_path)
#     net.gen.max_p_mw = np.array([80, 50, 35, 30, 40])
#     net.gen.min_p_mw = np.array([20, 15, 10, 10, 12])
#     net.ext_grid.max_p_mw = np.array([250])
#     net.ext_grid.min_p_mw = np.array([50])
#     net.gen.max_q_mvar = np.array([60, 62.45, 48.73, 40, 44.72])
#     net.gen.min_q_mvar = np.array([-20, -15, -15, -10, -15])
#     net.ext_grid.max_q_mvar = np.array([150])
#     net.ext_grid.min_q_mvar = np.array([-20])
#     return net

def load_net_9(file_path:str="./power_case/case9") -> pandapowerNet:
    with open(file_path, 'rb') as f:
        net = pickle.load(f)
    return net

def load_net_30(file_path:str="./power_case/case30") -> pandapowerNet:
    with open(file_path, 'rb') as f:
        net = pickle.load(f)
    return net

# def load_net_118_a(file_path:str="./power_case/case118_1.m") -> pandapowerNet:
#     net = converter.from_mpc(file_path)
#     return net

# def load_net_118_b(file_path:str="./power_case/case1_118.m") -> pandapowerNet:
#     net = converter.from_mpc(file_path)
#     return net

def load_net_118_a(file_path:str="./power_case/case118a") -> pandapowerNet:
    with open(file_path, 'rb') as f:
        net = pickle.load(f)
    return net

def load_net_118_b(file_path:str="./power_case/case118b") -> pandapowerNet:
    with open(file_path, 'rb') as f:
        net = pickle.load(f)
    return net

class Network(ABC):
    def __init__(self, net:pandapowerNet) -> None:
        self.net = net
        self.x_upper_bound = []
        self.x_lower_bound = []

        # Control Variable
        # PV bus active power
        self.num_of_generator = len(net.gen)
        self.x_upper_bound.extend(list(net.gen.max_p_mw.values))
        self.x_lower_bound.extend(list(net.gen.min_p_mw.values))
        # PV bus and Slack bus voltage magnitude
        mask = list(net.gen.bus.values)
        mask.extend(list(net.ext_grid.bus.values))
        self.x_upper_bound.extend(list(net.bus.max_vm_pu.loc[mask].values))
        self.x_lower_bound.extend(list(net.bus.min_vm_pu.loc[mask].values))
        self.PQ_bus = [i for i in range(len(self.net.bus)) if i not in mask]

        # Power demand
        self.total_loads_p_mw = sum(net.load.p_mw.values)
        self.total_loads_q_mvar = sum(net.load.q_mvar.values)

    def get_upper_bound(self) -> list[float]:
        return self.x_upper_bound
    
    def get_lower_bound(self) -> list[float]:
        return self.x_lower_bound

    def set_control_variable(self, x:list[float]) -> None:
        self.net.gen.p_mw = np.array(x[0 : self.num_of_generator])
        self.net.gen.vm_pu = np.array(x[self.num_of_generator : 2 * self.num_of_generator])
        self.net.ext_grid.vm_pu = np.array(x[2 * self.num_of_generator])

    def run_powerflow(self) -> bool:
        EPSILON_IN = 1 * 10 ** (-5)
        converge = True
        try:
            pp.runpp(self.net, tolerance_mva=EPSILON_IN, numba=True)
        except:
            converge = False
        return converge

    @abstractmethod
    def total_cost(self) -> float:
        ...

    def penalty(self, penalty_g=5, penalty_v=50) -> float:
        penalty = 0.0
        # Slack Bus P
        slack_bus_p_mw = self.net.res_ext_grid.p_mw.at[0]
        if slack_bus_p_mw > self.net.ext_grid.max_p_mw.at[0]:
            penalty += (slack_bus_p_mw - self.net.ext_grid.max_p_mw.at[0]) * penalty_g
        elif slack_bus_p_mw < self.net.ext_grid.min_p_mw.at[0]:
            penalty += (self.net.ext_grid.min_p_mw.at[0] - slack_bus_p_mw) * penalty_g
        # PV bus Q
        for i in range(self.num_of_generator):
            gen_q_mvar = self.net.res_gen.q_mvar.at[i]
            if gen_q_mvar > self.net.gen.max_q_mvar.at[i]:
                penalty += (gen_q_mvar - self.net.gen.max_q_mvar.at[i]) * penalty_g
            elif gen_q_mvar < self.net.gen.min_q_mvar.at[i]:
                penalty += (self.net.gen.min_q_mvar.at[i] - gen_q_mvar) * penalty_g
        # PQ bus V
        for idx in self.PQ_bus:
            pq_bus_v = self.net.res_bus.vm_pu.at[idx]
            if pq_bus_v > self.net.bus.max_vm_pu.at[idx]:
                penalty += (pq_bus_v - self.net.bus.max_vm_pu.at[idx]) * penalty_v
            elif pq_bus_v < self.net.bus.min_vm_pu.at[idx]:
                penalty += (self.net.bus.min_vm_pu.at[idx] - pq_bus_v) * penalty_v
        
        return penalty

class Network9(Network):
    def __init__(self) -> None:
        net = load_net_9()
        super().__init__(net)

    def total_cost(self) -> float:
        cost = 0.0
        p_mw = []
        p_mw.append(self.net.res_ext_grid.p_mw.at[0])
        p_mw.extend(list(self.net.res_gen.p_mw.values))
        for i in range(len(p_mw)):
            if p_mw[i] > 0:
                cost += self.net.poly_cost.cp0_eur.at[i] + \
                    self.net.poly_cost.cp1_eur_per_mw.at[i] * p_mw[i] + \
                        self.net.poly_cost.cp2_eur_per_mw2.at[i] * (p_mw[i] ** 2)
        return cost

class Network30(Network):
    def __init__(self) -> None:
        net = load_net_30()
        super().__init__(net)

    def total_cost(self) -> float:
        # generator cost function parameter
        a = [0., 0., 0., 0., 0., 0.]
        b = [2, 1.75, 1., 3.25, 3, 3]
        c = [0.00375, 0.0175, 0.0625, 0.00834, 0.025, 0.025]
        e = [18, 16, 14, 12, 13, 13.5]
        f = [0.037, 0.038, 0.04, 0.045, 0.042, 0.041]  
        cost = 0.0
        p_mw = []
        p_mw.append(self.net.res_ext_grid.p_mw.at[0])
        p_mw.extend(list(self.net.res_gen.p_mw.values))
        min_p_mw = []
        min_p_mw.append(self.net.ext_grid.min_p_mw.at[0])
        min_p_mw.extend(self.net.gen.min_p_mw.values)
        for i in range(len(p_mw)):
            if p_mw[i] > 0:
                cost += a[i] + b[i] * p_mw[i] + c[i] * (p_mw[i] ** 2) + abs(e[i] * sin(f[i] * (min_p_mw[i] - p_mw[i])))
        return cost

class _Network118(Network):
    def __init__(self, net: pandapowerNet) -> None:
        super().__init__(net)
    
    def total_cost(self) -> float:
        # generator cost function parameter
        a = [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0
        ]
        b = [
            20, 40, 40, 40, 40, 20, 20, 40, 40, 40,
            40, 20, 20, 40, 20, 40, 40, 40, 40, 40,
            20, 20, 20, 40, 20, 20, 40, 20, 20, 20,
            40, 40, 40, 40, 40, 40, 20, 40, 20, 20,
            40, 40, 40, 40, 20, 20, 40, 40, 40, 40,
            20, 40, 40, 40
        ]
        c = [
            0.0193648335, 0.01, 0.01, 0.01, 0.01, 0.0222222222, 0.117647059, 0.01, 0.01, 0.01, 
            0.01, 0.0454545455, 0.0318471338, 0.01, 1.42857143, 0.01, 0.01, 0.01, 0.01, 0.01, 
            0.526315789, 0.0490196078, 0.208333333, 0.01, 0.01, 0.064516129, 0.0625, 0.01, 0.0255754476, 0.0255102041,
            0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.0209643606, 0.01, 2.5, 0.0164744646,
            0.01, 0.01, 0.01, 0.01, 0.0396825397, 0.25, 0.01, 0.01, 0.01, 0.01,
            0.277777778, 0.01, 0.01, 0.01
        ]
        d = [
            13.5, 18, 16, 14, 12, 13, 13.5, 18, 16, 14, 
            12, 13, 13.5, 18, 16, 14, 12, 13, 13.5, 18, 
            16, 14, 12, 13, 13.5, 18, 16, 14, 12, 13,
            18, 16, 14, 12, 13, 13.5, 18, 16, 14, 12, 
            13, 13.5, 18, 16, 14, 12, 13, 13.5, 18, 16,
            14, 12, 13, 13.5
        ]
        e = [
            0.041, 0.037, 0.038, 0.04, 0.045, 0.042, 0.041, 0.037, 0.038, 0.04, 
            0.045, 0.042, 0.041, 0.037, 0.038, 0.04, 0.045, 0.042, 0.041, 0.037,
            0.038, 0.04, 0.045, 0.042, 0.041, 0.037, 0.038, 0.04, 0.045, 0.042, 
            0.037, 0.038, 0.04, 0.045, 0.042, 0.041, 0.037, 0.038, 0.04, 0.045, 
            0.042, 0.041, 0.037, 0.038, 0.04, 0.045, 0.042, 0.041, 0.037, 0.038,
            0.04, 0.045, 0.042, 0.041
        ]
        
        cost = 0.0
        p_mw = []
        p_mw.append(self.net.res_ext_grid.p_mw.at[0])
        p_mw.extend(list(self.net.res_gen.p_mw.values))
        min_p_mw = []
        min_p_mw.append(self.net.ext_grid.min_p_mw.at[0])
        min_p_mw.extend(self.net.gen.min_p_mw.values)
        for i in range(len(p_mw)):
            if p_mw[i] > 0:
                cost += a[i] + b[i] * p_mw[i] + c[i] * (p_mw[i] ** 2) + abs(d[i] * sin(e[i] * (min_p_mw[i] - p_mw[i])))
        return cost

    def penalty(self, penalty_g=50, penalty_v=300) -> float:
        return super().penalty(penalty_g, penalty_v)

class Network118a(_Network118):
    def __init__(self) -> None:
        net = load_net_118_a()
        super().__init__(net)

class Network118b(_Network118):
    def __init__(self) -> None:
        net = load_net_118_b()
        super().__init__(net)