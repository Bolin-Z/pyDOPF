import ray
import pandas as pd
from distopf.networks import Network118a, Network118b
from distopf.dist_optimizer import LLSO, Particle
from distopf.dist_eval import dist_eval_236
import warnings

MODES = ['global_best', 'top_level_random']

def run(test_name:str) -> None:
    ray.init()
    # parameter
    LOOP = 2000
    SWARM_SIZE = 30

    mode = MODES[0]

    neta = Network118a()
    netb = Network118b()
    optim_a = LLSO('net118a', neta.get_upper_bound(), neta.get_lower_bound(), SWARM_SIZE, mode)
    optim_b = LLSO('net118b', netb.get_upper_bound(), netb.get_lower_bound(), SWARM_SIZE, mode)

    print(f"{'#' * 15} EXP  {test_name} {'#' * 15}")
    print(f"Iter: {LOOP}\nSwarm_size: {SWARM_SIZE}\nmode: {mode}")
    print(f"{'#' * 15} LOOP {test_name} {'#' * 15}")
    record_content = ["Iteration", "cost_a", "cost_b", "total_cost", "fit_a", "fit_b", "pen_a", "pen_b", "total_fitness", "history_best"]
    history_best = float('inf')
    records = []

    for iter in range(LOOP):
        # evolve
        optim_a.evolve_particle()
        optim_b.evolve_particle()
        # evaluate
        select_a:Particle = optim_a.selected_particle
        select_b:Particle = optim_b.selected_particle

        future_a = [dist_eval_236.remote(p.x, select_b.x) for p in optim_a.swarm]
        future_b = [dist_eval_236.remote(select_a.x, p.x) for p in optim_b.swarm]

        result_a = ray.get(future_a)
        result_b = ray.get(future_b)

        select_a_res = []
        select_b_res = []
        for idx, t in enumerate(result_a):
            _, cost_a, fit_a, cost_b, fit_b = t
            optim_a.swarm[idx].cost = cost_a
            optim_a.swarm[idx].local_fitness = fit_a
            optim_a.swarm[idx].fitness = fit_a + fit_b
            select_b_res.append([cost_b, fit_b, fit_a + fit_b])
        for idx, t in enumerate(result_b):
            _, cost_a, fit_a, cost_b, fit_b = t
            optim_b.swarm[idx].cost = cost_b
            optim_b.swarm[idx].local_fitness = fit_b
            optim_b.swarm[idx].fitness = fit_a + fit_b
            select_a_res.append([cost_a, fit_a, fit_a + fit_b])

        for i in range(len(select_a_res)):
            cost_a, fit_a, total_fit = select_a_res[i]
            if total_fit < optim_a.selected_particle.fitness:
                optim_a.selected_particle.cost = cost_a
                optim_a.selected_particle.local_fitness = fit_a
                optim_a.selected_particle.fitness = total_fit
        for i in range(len(select_b_res)):
            cost_b, fit_b, total_fit = select_b_res[i]
            if total_fit < optim_b.selected_particle.fitness:
                optim_b.selected_particle.cost = cost_b
                optim_b.selected_particle.local_fitness = fit_b
                optim_b.selected_particle.fitness = total_fit
        
        # update
        optim_a.update()
        optim_b.update()

        if optim_a.swarm[0].fitness < history_best:
            history_best = optim_a.swarm[0].fitness
        if optim_b.swarm[0].fitness < history_best:
            history_best = optim_b.swarm[0].fitness
        
        best_a = optim_a.swarm[0]
        best_b = optim_b.swarm[0]
        log = f"""\
Iter {iter} {test_name}
    cost_a:     {best_a.cost}
    cost_b:    {best_b.cost}
    cost_total: {best_a.cost + best_b.cost}
    fit_a:      {best_a.local_fitness}
    fit_b:     {best_b.local_fitness}
    pen_a:      {best_a.local_fitness - best_a.cost}
    pen_b:     {best_b.local_fitness -  best_b.cost}
    fit_total:  {best_a.fitness} ({best_b.fitness})
    h_best:     {history_best}
"""
        print(log)

        records.append([
            iter,
            best_a.cost, best_b.cost, best_a.cost + best_b.cost,
            best_a.local_fitness, best_b.local_fitness,
            best_a.local_fitness - best_a.cost, best_b.local_fitness -  best_b.cost,
            best_a.fitness,
            history_best
        ])

    # record
    output = pd.DataFrame(data=records, columns=record_content)
    output.to_csv(f"result/{test_name}.csv", index=False)

    ray.shutdown()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    for test_idx in range(20):
        run(f"case236_global_best_{test_idx}")