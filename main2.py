import ray
import pandas as pd
from distopf.networks import Network9, Network30
from distopf.dist_optimizer import LLSO, Particle
from distopf.dist_eval import dist_eval
from copy import copy

MODES = ['global_best', 'top_level_random']

def run(test_name:str, mode:str):
    ray.init()
    # parameter
    LOOP = 150
    SWARM_SIZE = 30

    net9 = Network9()
    net30 = Network30()
    optim_9 = LLSO(9, net9.get_upper_bound(), net9.get_lower_bound(), SWARM_SIZE, mode)
    optim_30 = LLSO(30, net30.get_upper_bound(), net30.get_lower_bound(), SWARM_SIZE, mode)
    # evaluate
    optim_9.level()
    optim_9.update_selected_particle()
    optim_30.level()
    optim_30.update_selected_particle()
    select_9:Particle = optim_9.selected_particle
    select_30:Particle = optim_30.selected_particle
    
    future_9 = [dist_eval.remote(p.x, select_30.x) for p in optim_9.swarm]
    future_30 = [dist_eval.remote(select_9.x, p.x) for p in optim_30.swarm]
    result_9 = ray.get(future_9)
    result_30 = ray.get(future_30)

    select_9_res = []
    select_30_res = []
    for idx, t in enumerate(result_9):
        _, cost_9, fit_9, cost_30, fit_30 = t
        optim_9.swarm[idx].cost = cost_9
        optim_9.swarm[idx].local_fitness = fit_9
        optim_9.swarm[idx].fitness = fit_9 + fit_30
        select_30_res.append([cost_30, fit_30, fit_9 + fit_30])
    for idx, t in enumerate(result_30):
        _, cost_9, fit_9, cost_30, fit_30 = t
        optim_30.swarm[idx].cost = cost_30
        optim_30.swarm[idx].local_fitness = fit_30
        optim_30.swarm[idx].fitness = fit_9 + fit_30
        select_9_res.append([cost_9, fit_9, fit_9 + fit_30])

    for i in range(len(select_9_res)):
        cost_9, fit_9, total_fit = select_9_res[i]
        if total_fit < optim_9.selected_particle.fitness:
            optim_9.selected_particle.cost = cost_9
            optim_9.selected_particle.local_fitness = fit_9
            optim_9.selected_particle.fitness = total_fit
    for i in range(len(select_30_res)):
        cost_30, fit_30, total_fit = select_30_res[i]
        if total_fit < optim_30.selected_particle.fitness:
            optim_30.selected_particle.cost = cost_30
            optim_30.selected_particle.local_fitness = fit_30
            optim_30.selected_particle.fitness = total_fit

    optim_9.swarm.sort()
    optim_30.swarm.sort()
    optim_9.best_fitness = optim_9.swarm[0].fitness
    optim_30.best_fitness = optim_30.swarm[0].fitness  

    print(f"{'#' * 15} EXP  {test_name} {'#' * 15}")
    print(f"Iter: {LOOP}\nSwarm_size: {SWARM_SIZE}\nmode: {mode}")
    print(f"{'#' * 15} LOOP {test_name} {'#' * 15}")
    record_content = ["Iteration", "cost_9", "cost_30", "total_cost", "fit_9", "fit_30", "pen_9", "pen_30", "total_fitness", "history_best"]
    history_best = float('inf')
    records = []

    for iter in range(LOOP):
        # evolve
        optim_9.evolve_particle()
        optim_30.evolve_particle()
        # evaluate
        level_size_9 = optim_9.cur_level_size()
        level_size_30 = optim_30.cur_level_size()
        select_9:Particle = optim_9.selected_particle
        select_30:Particle = optim_30.selected_particle

        future_9 = [dist_eval.remote(p.x, select_30.x) for p in optim_9.swarm[level_size_9:]]
        future_30 = [dist_eval.remote(select_9.x, p.x) for p in optim_30.swarm[level_size_30:]]

        result_9 = ray.get(future_9)
        result_30 = ray.get(future_30)
            
        for idx, t in enumerate(result_9):
            p = level_size_9 + idx
            _, cost_9, fit_9, cost_30, fit_30 = t
            optim_9.swarm[p].cost = cost_9
            optim_9.swarm[p].local_fitness = fit_9
            total_fit = fit_9 + fit_30
            optim_9.swarm[p].fitness = total_fit
            if total_fit < select_30.fitness:
                select_30.cost = cost_30
                select_30.local_fitness = fit_30
                select_30.fitness = total_fit
        for idx, t in enumerate(result_30):
            p = level_size_30 + idx
            _, cost_9, fit_9, cost_30, fit_30 = t
            optim_30.swarm[idx].cost = cost_30
            optim_30.swarm[idx].local_fitness = fit_30
            total_fit = fit_9 + fit_30
            optim_30.swarm[idx].fitness = total_fit
            if total_fit < select_9.fitness:
                select_9.cost = cost_9
                select_9.local_fitness = fit_9
                select_9.fitness = total_fit
    
        # update
        optim_9.update()
        optim_30.update()

        if optim_9.swarm[0].fitness < history_best:
            history_best = optim_9.swarm[0].fitness
        if optim_30.swarm[0].fitness < history_best:
            history_best = optim_30.swarm[0].fitness

        best_9 = optim_9.swarm[0]
        best_30 = optim_30.swarm[0]
        log = f"""\
Iter {iter} {test_name}
    cost_9:     {best_9.cost}
    cost_30:    {best_30.cost}
    cost_total: {best_9.cost + best_30.cost}
    fit_9:      {best_9.local_fitness}
    fit_30:     {best_30.local_fitness}
    pen_9:      {best_9.local_fitness - best_9.cost}
    pen_30:     {best_30.local_fitness -  best_30.cost}
    fit_total:  {best_9.fitness} ({best_30.fitness})
    h_best:     {history_best}
"""
        print(log)

        records.append([
            iter,
            best_9.cost, best_30.cost, best_9.cost + best_30.cost,
            best_9.local_fitness, best_30.local_fitness,
            best_9.local_fitness - best_9.cost, best_30.local_fitness -  best_30.cost,
            best_9.fitness,
            history_best
        ])
    
    # record
    output = pd.DataFrame(data=records, columns=record_content)
    output.to_csv(f"result/{test_name}.csv", index=False)

    ray.shutdown()

if __name__ == "__main__":
    for test_idx in range(10):
        run(f"v_2_case39_global_best_{test_idx}", 'global_best')
    for test_idx in range(10):
        run(f"v_2_case39_'top_level_random'_{test_idx}", 'top_level_random')
