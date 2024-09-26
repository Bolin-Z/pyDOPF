import ray
import pandas as pd
from distopf.networks import Network9, Network30
from copf.c_optimizer import LLSO_C, Particle_C
from distopf.dist_eval import dist_eval
from copy import copy

def run(test_name:str):
    ray.init()
    # parameter
    LOOP = 150
    SWARM_SIZE = 30

    net9 = Network9()
    net30 = Network30()
    upper_bound = net9.get_upper_bound()
    lower_bound = net9.get_lower_bound()
    mid = len(upper_bound)
    upper_bound.extend(net30.get_upper_bound())
    lower_bound.extend(net30.get_lower_bound())

    optim = LLSO_C(upper_bound, lower_bound, SWARM_SIZE)
    # evaluate
    futures = [dist_eval.remote(p.x[:mid], p.x[mid:]) for p in optim.swarm]
    result = ray.get(futures)
    for idx, t in enumerate(result):
        _, cost_9, fit_9, cost_30, fit_30 = t
        optim.swarm[idx].cost_9 = cost_9
        optim.swarm[idx].cost_30 = cost_30
        optim.swarm[idx].cost = cost_9 + cost_30
        optim.swarm[idx].fit_9 = fit_9
        optim.swarm[idx].fit_30 = fit_30
        optim.swarm[idx].fitness = fit_9 + fit_30
    optim.swarm.sort()
    optim.best_fitness = optim.swarm[0].fitness

    print(f"{'#' * 15} EXP  {test_name} {'#' * 15}")
    print(f"Iter: {LOOP}\nSwarm_size: {SWARM_SIZE}")
    print(f"{'#' * 15} LOOP {test_name} {'#' * 15}")
    record_content = ["Iteration", "cost_9", "cost_30", "total_cost", "fit_9", "fit_30", "pen_9", "pen_30", "total_fitness", "history_best"]
    history_best = float('inf')
    records = []

    for iter in range(LOOP):
        # evolve
        optim.evolve_particle()
        # evaluate level 2 - level NL [cur_level_size : end]
        cur_level_size = optim.cur_level_size()
        futures = [dist_eval.remote(optim.swarm[i].x[:mid], optim.swarm[i].x[mid:]) for i in range(cur_level_size, len(optim.swarm))]
        result = ray.get(futures)
        for idx, t in enumerate(result):
            _, cost_9, fit_9, cost_30, fit_30 = t
            p = cur_level_size + idx
            optim.swarm[p].cost_9 = cost_9
            optim.swarm[p].cost_30 = cost_30
            optim.swarm[p].cost = cost_9 + cost_30
            optim.swarm[p].fit_9 = fit_9
            optim.swarm[p].fit_30 = fit_30
            optim.swarm[p].fitness = fit_9 + fit_30
        
        # update
        optim.update()

        if optim.swarm[0].fitness < history_best:
            history_best = optim.swarm[0].fitness
        
        best_p = optim.swarm[0]

        log = f"""\
Iter {iter} {test_name}
    cost_9:     {best_p.cost_9}
    cost_30:    {best_p.cost_30}
    cost_total: {best_p.cost}
    fit_9:      {best_p.fit_9}
    fit_30:     {best_p.fit_30}
    pen_9:      {best_p.fit_9 - best_p.cost_9}
    pen_30:     {best_p.fit_30 - best_p.cost_30}
    fit_total:  {best_p.fitness}
    h_best:     {history_best}
"""
        print(log)

        records.append([
            iter,
            best_p.cost_9, best_p.cost_30, best_p.cost,
            best_p.fit_9, best_p.fit_30,
            best_p.fit_9 - best_p.cost_9, best_p.fit_30 - best_p.cost_30,
            best_p.fitness,
            history_best
        ])
    
    # record
    output = pd.DataFrame(data=records, columns=record_content)
    output.to_csv(f"result/{test_name}.csv", index=False)

    ray.shutdown()

if __name__ == "__main__":
    for test_idx in range(20):
        run(f"case39_central_{test_idx}")