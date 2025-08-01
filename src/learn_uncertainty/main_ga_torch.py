import pygad
import numpy as np
import random
import time
from learn_uncertainty import distance, fit_traj
from learn_uncertainty.fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict,plot,recplot,scatter_with_number,read_config,Alignment,donothing,NoAlignedAfter,NoAlignedBefore
import torch
#*** Algo Gen parameters ***#

random.seed(41)
num_gen =1
prob_no_crossover=0.55
current_population = None
fitness_cache = None
last_generation = -1
last_population = None
param_names = ['dangleI', 'dangleS', 'dt0I', 'dt0S', 'dt1I', 'dt1S','dspeedI', 'dspeedS', 'ldspeedI', 'ldspeedS','vspeedI','vspeedS']
to_meters = 1852
#*** Learn uncertainty parameters ***#

DEVICE = torch.device("cuda")
fname = "/disk2/jsonKimdebugBeacons/all_800_10_1800_2.dsituation"
#fname = "/disk2/jsonKimdebugBeacons/situations_800_120_120_10_1800/2201/34330127_1643280923_1643281399.situation"
DSITUATION = fit_traj.load_situation(fname)
print(list(DSITUATION.keys()))
DSITUATION = {10:DSITUATION[10][:400]}
modelDistance = distance.GenerateDistance.from_dsituation_step(DSITUATION,step=5)
modelDistance = modelDistance.to(DEVICE)


params_names = ['dangleI','dangleS',
           'dt0I','dt0S',
           'dt1I','dt1S',
           'dspeedI','dspeedS',
           'ldspeedI','ldspeedS',
           'vspeedI','vspeedS']

param_bounds = {
    'dangleI': (np.radians(-10), 0),
    'dangleS': (0,np.radians(10)),
    'dt0I': (-30, 0),
    'dt0S': (0, 60),
    'dt1I': (-30, 0),
    'dt1S': (0, 60),
    'dspeedI': (0.8, 1.),
    'dspeedS': (1., 1.2),
    'ldspeedI': (0.8, 1.),
    'ldspeedS': (1., 1.2),
    'vspeedI': (0.8, 1),
    'vspeedS': (1, 1.2),
}

gene_space = [
    {'low':param_bounds[k][0],'high':param_bounds[k][1]}
    for k in params_names
    # {'low': param_bounds['dangleI'][0], 'high': param_bounds['dangleI'][1]},
    # {'low': param_bounds['dangleS'][0], 'high': param_bounds['dangleS'][1]},
    # {'low': param_bounds['dt0I'][0], 'high': param_bounds['dt0I'][1]},
    # {'low': param_bounds['dt0S'][0], 'high': param_bounds['dt0S'][1]},
    # {'low': param_bounds['dt1I'][0], 'high': param_bounds['dt1I'][1]},
    # {'low': param_bounds['dt1S'][0], 'high': param_bounds['dt1S'][1]},
    # {'low': param_bounds['dspeedI'][0], 'high': param_bounds['dspeedI'][1]},
    # {'low': param_bounds['dspeedS'][0], 'high': param_bounds['dspeedS'][1]},
    # {'low': param_bounds['ldspeedI'][0], 'high': param_bounds['ldspeedI'][1]},
    # {'low': param_bounds['ldspeedS'][0], 'high': param_bounds['ldspeedS'][1]},
    # {'low': param_bounds['vspeedI'][0], 'high': param_bounds['vspeedI'][1]},
    # {'low': param_bounds['vspeedS'][0], 'high': param_bounds['vspeedS'][1]}
]

def decode_solution(solution):
    return {name: solution[i] for i, name in enumerate(param_names)}

def batch_fitness_evaluation(population):
    #global alldmin
    sols = [decode_solution(s) for s in population]
    dangle = [[params['dangleI'],params['dangleS']] for params in sols]
    dt0 = [[params['dt0I'],params['dt0S']] for params in sols]
    dt1 = [[params['dt1I'],params['dt1S']] for params in sols]
    dspeed = [[params['dspeedI'],params['dspeedS']] for params in sols]
    ldspeed = [[params['ldspeedI'],params['ldspeedS']] for params in sols]
    vspeed = [[params['vspeedI'],params['vspeedS']] for params in sols]
    duparams = {
        "dangle": torch.tensor(dangle),
        "dt0": torch.tensor(dt0),
        "dt1": torch.tensor(dt1),
        "dspeed": torch.tensor(dspeed),
        "ldspeed": torch.tensor(ldspeed),
        "vspeed": torch.tensor(vspeed),
    }
    duparams = {k:v.to(DEVICE) for k,v in duparams.items()}
    st = time.perf_counter()
    d,lid,ltzero = modelDistance(duparams)
    d = d.cpu()
    nbsituations = d.shape[1]
    # print(d.shape)
    # raise Exception
    print(time.perf_counter()-st)
    # print(d.shape,d.names)
    #scores = ((d-5)**2).mean(axis=-1)
    scores = (d-5).abs().mean(axis=-1)
    i=scores.rename(None).argmin()
    print(i)
    print(d[i].min(),d[i].mean(),d[i].max(),np.var(d[i].numpy()))
    print(np.mean((d[i]-5).abs().numpy()),np.var((d[i]-5).abs().numpy()))
    fits = 1 / (1 + scores)
    # print(fits)
    # raise Exception
    # for i in range(len(population)):
    #     dmin = d[i].tolist()
    #     alldmin.append(dmin)
    #     fits.append(nbsituations / (nbsituations + sum([ (d-5)**2 for d in dmin])))
    return fits.tolist()


def fitness_func(ga_instance, solution, solution_idx):
    global fitness_cache, last_population

    current_population = ga_instance.population


    if last_population is None or not np.array_equal(current_population, last_population):
        fitness_cache = batch_fitness_evaluation(current_population)
        last_population = current_population.copy()

    return fitness_cache[solution_idx]

def scale_old(child,ga_instance):
    child = np.copy(child)
    # print(child)
    # raise Exception
    for i in range(num_genes):
        low = ga_instance.gene_space[i]['low']
        high = ga_instance.gene_space[i]['high']
        child[i] = np.clip(child[i], low, high)
    for i in [6,8,10]:
        low = ga_instance.gene_space[i]['low']
        child[i] = np.clip(child[i], low, child[i+1])
    return child

def scale(child,ga_instance):
    child = np.copy(child)
    # print(child.shape)
    # raise Exception
    low = np.array([v['low'] for v in ga_instance.gene_space])
    high = np.array([v['high'] for v in ga_instance.gene_space])
    vhigh = np.clip(child-high,min=0.)
    vlow = np.clip(low-child,min=0.)
    child = child + 2 * (vlow-vhigh)
    for i in range(num_genes):
        assert(low[i]<=child[i]<=high[i])
    # for i in [6,8,10]:
    #     low = ga_instance.gene_space[i]['low']
    #     child[i] = np.clip(child[i], low, child[i+1])
    return child

def custom_crossover(parents, offspring_size, ga_instance):
    offspring = []
    num_parents = parents.shape[0]
    # print(num_parents,offspring_size)
    # raise Exception
    num_genes = parents.shape[1]
    while len(offspring) < offspring_size[0]:
        if random.random() < prob_no_crossover:
            idx = random.randint(0, num_parents - 1)
            child = parents[idx].copy()
            offspring.append(child)
        else:
            p1_idx, p2_idx = random.sample(range(num_parents), 2)
            parent1 = parents[p1_idx]
            parent2 = parents[p2_idx]
            #alphas = np.random.rand(num_genes)
            alphas = np.random.uniform(low=-0.5, high=1.5, size=num_genes)
            child1 = alphas * parent1 + (1 - alphas) * parent2
            child2 = (1 - alphas) * parent1 + alphas * parent2
            child1 = scale(child1,ga_instance)
            child2 = scale(child2,ga_instance)
            offspring.append(child1)
            if len(offspring) < offspring_size[0]:
                offspring.append(child2)

    return np.array(offspring)

def custom_mutation(offspring, ga_instance):
    for i in range(offspring.shape[0]):
        gene_idx = random.randint(0, len(offspring[i]) - 1)
        low = ga_instance.gene_space[gene_idx]['low']
        high = ga_instance.gene_space[gene_idx]['high']
        step_size = (high - low) / 4.0
        delta = random.uniform(-step_size, step_size)
        new_value = offspring[i][gene_idx] + delta
        offspring[i][gene_idx] = new_value
        offspring[i] = scale(offspring[i],ga_instance)
    return offspring

def on_generation(ga_instance):
    global num_gen
    best_sol, best_fitness, best_indice = ga_instance.best_solution()
    print(f"Génération {ga_instance.generations_completed} : Meilleure fitness = {best_fitness}")
    print(f" Meilleur individu : {best_sol}")
    # print(f" Din moy:{np.mean(dsol)}, ecart type:{np.std(dsol)}")
    with open("save_gens500.txt","a") as file:
        file.write(f"Numgen : {num_gen}\n")
        file.write(f"Sol: {best_sol}\n")
        file.write(f"Fit: {best_fitness}\n")
        #file.write(f" Din moy:{np.mean(dsol)}, ecart type:{np.std(dsol)}\n")
        #file.write(f" Din moy diff:{np.mean([abs(d-5) for d in dsol])}, ecart type nu:{np.std([abs(d-5) for d in dsol])}\n\n")
    num_gen += 1
def on_start(ga_instance):
    for sol in ga_instance.population:
        sol = scale(sol,ga_instance)

# num_generations = 10
# num_parents_mating = 40

sol_per_pop = 100
num_genes = len(param_names)


parent_selection_type = "rws"

ga_instance = pygad.GA(
    num_generations=10,
    on_start=on_start,
    random_seed=42,
    num_parents_mating=int(sol_per_pop*0.8),
    fitness_func=fitness_func,
    sol_per_pop=sol_per_pop,
    parent_selection_type="tournament",
    K_tournament = 2,
    keep_elitism=1,
    num_genes=num_genes,
    on_generation=on_generation,
    gene_space=gene_space,
    crossover_type = custom_crossover,#"scattered",
    mutation_type=custom_mutation,
    crossover_probability=0.7,
    mutation_probability=0.2,
#    crossover_type=custom_crossover,

)

ga_instance.run()
ga_instance.plot_fitness()
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("Best solution:", decode_solution(solution))
print("Fitness:", solution_fitness)
