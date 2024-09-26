from .networks import Network9, Network30
import pandapower as pp
import numpy as np
import ray

MAX_VALUE = 1000000

def cosd(degree):
    return np.cos(np.radians(degree))

def sind(degree):
    return np.sin(np.radians(degree))

@ray.remote
def eval_net_id(id:int, x:list[float]) -> tuple[bool, float, float]:
    # load network
    net = Network9() if id == 9 else Network30()
    net.set_control_variable(x)
    converge = net.run_powerflow()
    cost = MAX_VALUE
    fit = MAX_VALUE
    if converge:
        cost = net.total_cost()
        fit = cost + net.penalty()
    return (converge, cost, fit)

@ray.remote
def dist_eval(x_9:list[float], x_30:list[float]) -> tuple[bool, float, float, float, float]:
    # load network
    net9 = Network9()
    net30 = Network30()

    # parameter
    MAX_ITER = 12
    LAMBDA_1 = 0.7922
    ETA_1 = 0.7318
    LAMBDA_2 = 0.2078
    ETA_2 = 0.2682
    EPSILON_IN = 1 * 10 ** (-5)
    EPSILON_OUT = 1 * 10 ** (-4)
    ii = 0

    # return values
    converge = False
    cost_9 = MAX_VALUE
    fit_9 = MAX_VALUE
    cost_30 = MAX_VALUE
    fit_30 = MAX_VALUE

    pf_fail = False

    # virtual power input
    net9_cb = 3
    net9_vb = 9
    net9_sgen = pp.create_sgen(net9.net, net9_vb, 0)
    net9_load = pp.create_load(net9.net, net9_vb, 0)

    net30_cb = 2
    net30_vb = 30
    net30_sgen = pp.create_sgen(net30.net, net30_vb, 0)
    net30_load = pp.create_load(net30.net, net30_vb, 0)

    # impedance of tie-line
    z_1 = 0.053 + 0.057j
    s1_2 = []
    s2_1 = []

    # initial injection power
    s1_s2 = 0 + 0 * 1j
    s2_s1 = 0 + 0 * 1j

    s1_2.append(s1_s2)
    s2_1.append(s2_s1)

    if s2_s1.real < 0:
        net9.net.load.p_mw.at[net9_load] = - s2_s1.real
    else:
        net9.net.sgen.p_mw.at[net9_sgen] = s2_s1.real
    net9.net.sgen.q_mvar.at[net9_sgen] = s2_s1.imag

    if s1_s2.real < 0:
        net30.net.load.p_mw.at[net30_load] = - s1_s2.real
    else:
        net30.net.sgen.p_mw.at[net30_sgen] = s1_s2.real
    net30.net.sgen.q_mvar.at[net30_sgen] = s1_s2.imag

    # set control variable
    net9.set_control_variable(x_9)
    net30.set_control_variable(x_30)

    # first round
    converge_9 = net9.run_powerflow()
    converge_30 = net30.run_powerflow()
    if not(converge_9 and converge_30):
        pf_fail = True

    v_b1 = []
    theta_b1 = []
    v_b2 = []
    theta_b2 = []

    v_b1.append([net9.net.res_bus.vm_pu.at[net9_cb], net9.net.res_bus.vm_pu.at[net9_vb]])
    theta_b1.append([net9.net.res_bus.va_degree.at[net9_cb], net9.net.res_bus.va_degree.at[net9_vb]])
    v_b2.append([net30.net.res_bus.vm_pu.at[net30_vb], net30.net.res_bus.vm_pu.at[net30_cb]])
    theta_b2.append([net30.net.res_bus.va_degree.at[net30_vb], net30.net.res_bus.va_degree.at[net30_cb]])    

    # main loop
    v_update = []
    theta_update = []
    delta_v = []
    delta_theta = []
    while True and (not pf_fail):
        ii += 1
        v_update.append([
            v_b1[-1][0] * LAMBDA_1 + v_b2[-1][0] * (1 - LAMBDA_1),
            v_b1[-1][1] * LAMBDA_2 + v_b2[-1][1] * (1 - LAMBDA_2)
        ])
        mean = ((theta_b1[-1][0] - theta_b2[-1][0]) + (theta_b1[-1][1] - theta_b2[-1][1])) / 2
        theta_update.append([
            theta_b1[-1][0] * ETA_1 + (theta_b2[-1][0] + mean) * (1 - ETA_1),
            theta_b1[-1][1] * ETA_2 + (theta_b2[-1][1] + mean) * (1 - ETA_2)
        ])

        com_vb1 = v_update[-1][0] * cosd(theta_update[-1][0]) + (v_update[-1][0] * sind(theta_update[-1][0])) * 1j
        com_vb2 = v_update[-1][1] * cosd(theta_update[-1][1]) + (v_update[-1][1] * sind(theta_update[-1][1])) * 1j

        s1_2.append((com_vb1 * (com_vb1 - com_vb2) / z_1) * 100)
        s2_1.append((com_vb2 * (com_vb2 - com_vb1) / z_1) * 100)

        # reset
        net9.net.load.p_mw.at[net9_load] = 0
        net9.net.sgen.p_mw.at[net9_sgen] = 0
        net30.net.load.p_mw.at[net30_load] = 0
        net30.net.sgen.p_mw.at[net30_sgen] = 0
        # new injection
        s1_s2 = s1_2[-1]
        s2_s1 = s2_1[-1]
        if s2_s1.real < 0:
            net9.net.load.p_mw.at[net9_load] = - s2_s1.real
        else:
            net9.net.sgen.p_mw.at[net9_sgen] = s2_s1.real
        net9.net.sgen.q_mvar.at[net9_sgen] = s2_s1.imag

        if s1_s2.real < 0:
            net30.net.load.p_mw.at[net30_load] = - s1_s2.real
        else:
            net30.net.sgen.p_mw.at[net30_sgen] = s1_s2.real
        net30.net.sgen.q_mvar.at[net30_sgen] = s1_s2.imag

        # run power flow
        converge_9 = net9.run_powerflow()
        converge_30 = net30.run_powerflow()
        if not (converge_9 and converge_30):
            pf_fail = True
            break

        v_b1.append([net9.net.res_bus.vm_pu.at[net9_cb], net9.net.res_bus.vm_pu.at[net9_vb]])
        theta_b1.append([net9.net.res_bus.va_degree.at[net9_cb], net9.net.res_bus.va_degree.at[net9_vb]])
        v_b2.append([net30.net.res_bus.vm_pu.at[net30_vb], net30.net.res_bus.vm_pu.at[net30_cb]])
        theta_b2.append([net30.net.res_bus.va_degree.at[net30_vb], net30.net.res_bus.va_degree.at[net30_cb]])

        delta_v.append([abs(v_b1[-1][0] - v_b2[-1][0]), abs(v_b1[-1][1] - v_b2[-1][1])])
        mean = ((theta_b1[-1][0] - theta_b2[-1][0]) + (theta_b1[-1][1] - theta_b2[-1][1])) / 2
        delta_theta.append([
            abs(theta_b1[-1][0] - (theta_b2[-1][0] + mean)),
            abs(theta_b1[-1][1] - (theta_b2[-1][1] + mean))
        ])

        if(max(delta_v[-1]) < EPSILON_OUT and max(delta_theta[-1]) < EPSILON_OUT):
            converge = True
            break
        elif ii >= MAX_ITER:
            converge = False
            break
    
    # result
    if converge:
        cost_9 = net9.total_cost()
        fit_9 = cost_9 + net9.penalty()
        cost_30 = net30.total_cost()
        fit_30 = cost_30 + net30.penalty()
    return (converge, cost_9, fit_9, cost_30, fit_30)
