import ray
import pandas as pd
from distopf.networks import Network118a, Network118b
from copf.c_optimizer_2 import LLSO_C, Particle_C
from distopf.dist_eval import dist_eval_236
from copy import copy

def run(test_name:str):
    ray.init()
    # parameter
    LOOP = 2000
    SWARM_SIZE = 30

    neta = Network118a()
    netb = Network118b()
    upper_bound = neta.get_upper_bound()
    lower_bound = neta.get_lower_bound()
    mid = len(upper_bound)
    upper_bound.extend(netb.get_upper_bound())
    lower_bound.extend(netb.get_lower_bound())

    optim = LLSO_C(upper_bound, lower_bound, SWARM_SIZE)
    # evaluate
    futures = [dist_eval_236.remote(p.x[:mid], p.x[mid:]) for p in optim.swarm]
    result = ray.get(futures)
    for idx, t in enumerate(result):
        _, cost_a, fit_a, cost_b, fit_b = t
        optim.swarm[idx].cost_a = cost_a
        optim.swarm[idx].cost_b = cost_b
        optim.swarm[idx].cost = cost_a + cost_b
        optim.swarm[idx].fit_a = fit_a
        optim.swarm[idx].fit_b = fit_b
        optim.swarm[idx].fitness = fit_a + fit_b
    optim.swarm.sort()
    optim.best_fitness = optim.swarm[0].fitness

    print(f"{'#' * 15} EXP  {test_name} {'#' * 15}")
    print(f"Iter: {LOOP}\nSwarm_size: {SWARM_SIZE}")
    print(f"{'#' * 15} LOOP {test_name} {'#' * 15}")
    record_content = ["Iteration", "cost_a", "cost_b", "total_cost", "fit_a", "fit_b", "pen_a", "pen_b", "total_fitness", "history_best"]
    history_best = float('inf')
    records = []

    for iter in range(LOOP):
        # evolve
        optim.evolve_particle()
        # evaluate level 2 - level NL [cur_level_size : end]
        cur_level_size = optim.cur_level_size()
        futures = [dist_eval_236.remote(optim.swarm[i].x[:mid], optim.swarm[i].x[mid:]) for i in range(cur_level_size, len(optim.swarm))]
        result = ray.get(futures)
        for idx, t in enumerate(result):
            _, cost_a, fit_a, cost_b, fit_b = t
            p = cur_level_size + idx
            optim.swarm[p].cost_a = cost_a
            optim.swarm[p].cost_b = cost_b
            optim.swarm[p].cost = cost_a + cost_b
            optim.swarm[p].fit_a = fit_a
            optim.swarm[p].fit_b = fit_b
            optim.swarm[p].fitness = fit_a + fit_b
        
        # update
        optim.update()

        if optim.swarm[0].fitness < history_best:
            history_best = optim.swarm[0].fitness
        
        best_p = optim.swarm[0]

        log = f"""\
Iter {iter} {test_name}
    cost_a:     {best_p.cost_a}
    cost_b:    {best_p.cost_b}
    cost_total: {best_p.cost}
    fit_a:      {best_p.fit_a}
    fit_b:     {best_p.fit_b}
    pen_a:      {best_p.fit_a - best_p.cost_a}
    pen_b:     {best_p.fit_b - best_p.cost_b}
    fit_total:  {best_p.fitness}
    h_best:     {history_best}
"""
        print(log)

        records.append([
            iter,
            best_p.cost_a, best_p.cost_b, best_p.cost,
            best_p.fit_a, best_p.fit_b,
            best_p.fit_a - best_p.cost_a, best_p.fit_b - best_p.cost_b,
            best_p.fitness,
            history_best
        ])
    
    # record
    output = pd.DataFrame(data=records, columns=record_content)
    output.to_csv(f"result/{test_name}.csv", index=False)

    ray.shutdown()

if __name__ == "__main__":
    for test_idx in range(20):
        run(f"case236_central_{test_idx}")