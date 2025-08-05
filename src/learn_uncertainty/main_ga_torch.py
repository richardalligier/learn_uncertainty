import pygad
import numpy as np
import random
import time
from learn_uncertainty import distance, fit_traj
from learn_uncertainty.fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict,plot,recplot,scatter_with_number,read_config,Alignment,donothing,NoAlignedAfter,NoAlignedBefore,read_config
import torch
import os
#*** Algo Gen parameters ***#

params_names = [
    'dangleI', #'dangleS',
    'dt0I', 'dt0S', 'dt1I', 'dt1S','dspeedI', 'dspeedS', 'ldspeedI', 'ldspeedS','vspeedI','vspeedS']
to_meters = 1852
#*** Learn uncertainty parameters ***#

DEVICE = torch.device("cuda")
DTYPE = torch.float32
config = read_config()
print(config)
fname = os.path.join(config.FOLDER,"all_800_10_1800_2.dsituation")
#fname = "/disk2/jsonKimdebugBeacons/situations_800_120_120_10_1800/2201/34330127_1643280923_1643281399.situation"
DSITUATION = fit_traj.load_situation(fname)
print(list(DSITUATION.keys()))
#DSITUATION = {10:DSITUATION[10][:2]}
DSITUATION = {10:DSITUATION[10][:400]}
FID = np.concat([vi["deviated"].fid.numpy() for v in DSITUATION.values() for vi in v])
TDEVIATION = np.concat([vi["deviated"].tzero.numpy()+vi["deviated"].tdeviation.numpy().astype(np.int64) for v in DSITUATION.values() for vi in v])
DIST_MIN_ACTUAL = np.concat([vi["deviated"].actual_min_dist.numpy() for v in DSITUATION.values() for vi in v])

modelDistance = distance.GenerateDistance.from_dsituation_step(DSITUATION,step=5)
modelDistance = modelDistance.to(DEVICE)


# params_names = [
# 	'dangleI',
# 	'dangleS',
#          'dt0I','dt0S',
#            'dt1I','dt1S',
#            'dspeedI','dspeedS',
#            'ldspeedI','ldspeedS',
#            'vspeedI','vspeedS']

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
    return {name: solution[i] for i, name in enumerate(params_names)}

def compute_scores(metric,tau,clip_dist,d):
    assert(0<=tau<=1)
    if metric == "square":
        return ((d-5)**2).mean(axis=-1)
    elif metric == "abs":
        if clip_dist is not None:
            d = torch.clip(d,max=clip_dist)
        e = d-5
        epos = torch.clip(e,min=0)
        eneg = torch.clip(e,max=0)
        return (tau * epos - (1-tau) * eneg).mean(axis=-1)
    else:
        raise Exception

def prepare_dparams(sols):
    dangle = [[params['dangleI'],-params['dangleI']] for params in sols]
    dt0 = [[params['dt0I'],params['dt0S']] for params in sols]
    dt1 = [[params['dt1I'],params['dt1S']] for params in sols]
    dspeed = [[params['dspeedI'],params['dspeedS']] for params in sols]
    ldspeed = [[params['ldspeedI'],params['ldspeedS']] for params in sols]
    vspeed = [[params['vspeedI'],params['vspeedS']] for params in sols]
    duparams = {
        "dangle": torch.tensor(dangle,dtype=DTYPE),
        "dt0": torch.tensor(dt0,dtype=DTYPE),
        "dt1": torch.tensor(dt1,dtype=DTYPE),
        "dspeed": torch.tensor(dspeed,dtype=DTYPE),
        "ldspeed": torch.tensor(ldspeed,dtype=DTYPE),
        "vspeed": torch.tensor(vspeed,dtype=DTYPE),
    }
    return duparams
def fitness_func(ga_instance,population,population_indices):
    print(f"{len(population)=}")
    print(f"{min(population_indices)=}")
    #print(population)
    # if ga_instance.mylastpop is not None:
    #     print(ga_instance.mylastpop[population_indices]==population)
    ga_instance.mylastpop[population_indices]=population
    sols = [decode_solution(s) for s in population]
    # print(sols)
    duparams = prepare_dparams(sols)
    # for k in ["dangle", "dt1"]:
    #     duparams[k]=torch.zeros_like(duparams["dt0"])
    # for k in ["dspeed","ldspeed", "vspeed"]:
    #     duparams[k]=torch.ones_like(duparams["dt0"])
    # print(duparams)
    duparams = {k:v.to(DEVICE) for k,v in duparams.items()}
    # st = time.perf_counter()
    d,lid,ltzero = modelDistance(duparams)
    d = d.cpu()
    ga_instance.metric_dist[population_indices] = d
    # nbsituations = d.shape[1]
    # print(d.shape)
    # raise Exception
    # print(time.perf_counter()-st)
    # print(d.shape,d.names)
    # if ga_instance.metric == "square":
    #     scores = ((d-5)**2).mean(axis=-1)
    # elif ga_instance.metric == "abs":
    #     scores = (d-5).abs().mean(axis=-1)
    # else:
    #     raise Exception
    scores = compute_scores(ga_instance.metric,ga_instance.my_args.tau,ga_instance.my_args.clip_dist,d)
    # ga_instance.metric_ibest = i
    # print(i)
    # print(d[i].min(),d[i].mean(),d[i].max(),np.var(d[i].numpy()))
    # print(np.mean((d[i]-5).abs().numpy()),np.var((d[i]-5).abs().numpy()))
    fits = 1 / (1 + scores)
    if ga_instance.mybest_fitness is None or ga_instance.mybest_fitness<fits.rename(None).max().item():
        ibest = fits.rename(None).argmax()
        ga_instance.mybest_sol= population[ibest].copy()
        ga_instance.mybest_fitness = fits[ibest].item()
        ga_instance.mybest_dist = d[ibest].numpy().copy()
    ga_instance.metric_fitness[population_indices] = fits
    # print(fits)
    # raise Exception
    # for i in range(len(population)):
    #     dmin = d[i].tolist()
    #     alldmin.append(dmin)
    #     fits.append(nbsituations / (nbsituations + sum([ (d-5)**2 for d in dmin])))
    return fits.tolist()


# def fitness_func(ga_instance, solutions, solution_indices):
#     return batch_fitness_evaluation(ga_instance.metric,solutions)


def scale(child,ga_instance):
    child = np.copy(child)
    # print(child.shape)
    # raise Exception
    low = np.array([v['low'] for v in ga_instance.gene_space])
    high = np.array([v['high'] for v in ga_instance.gene_space])
    vhigh = np.clip(child-high,min=0.)
    vlow = np.clip(low-child,min=0.)
    child = child + 2 * (vlow-vhigh)
    for i in range(len(child)):
        assert(low[i]<=child[i]<=high[i])
    # for i in [6,8,10]:
    #     low = ga_instance.gene_space[i]['low']
    #     child[i] = np.clip(child[i], low, child[i+1])
    return child

def custom_crossover(parents, offspring_size, ga_instance):
    offspring = []
    num_parents = parents.shape[0]
    print(f"{num_parents=} {offspring_size=}")
    # raise Exception
    num_genes = parents.shape[1]
    while len(offspring) < offspring_size[0]:
        if random.random() > ga_instance.metric_prob_crossover:
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

# def custom_mutation_one(offspring, ga_instance):
#     print(f"mutation {offspring.shape=}")
#     #return offspring
#     for i in range(offspring.shape[0]):
#         gene_idx = random.randint(0, len(offspring[i]) - 1)
#         low = ga_instance.gene_space[gene_idx]['low']
#         high = ga_instance.gene_space[gene_idx]['high']
#         step_size = (high - low) / 4.0
#         delta = random.uniform(-step_size, step_size)
#         new_value = offspring[i,gene_idx] + delta
#         offspring[i,gene_idx] = new_value
#         offspring[i] = scale(offspring[i],ga_instance)
#     print(offinit==offspring)
#     #print(offspring)
#     return offspring.copy()

def custom_mutation(offspring, ga_instance):
    low = np.array([v['low'] for v in ga_instance.gene_space])
    high = np.array([v['high'] for v in ga_instance.gene_space])
    for i in range(offspring.shape[0]):
        if random.random() < ga_instance.metric_prob_mutation:
            step_size = (high - low) * ga_instance.my_args.scale_mutation
            if ga_instance.my_args.mutation_type == "uniform":
                delta = step_size * np.random.uniform(low=-1, high=1,size=len(step_size))
            elif ga_instance.my_args.mutation_type == "gaussian":
                delta = step_size * np.random.normal(size=len(step_size))
            else:
                raise Exception
            offspring[i]=offspring[i]+delta
            offspring[i] = scale(offspring[i],ga_instance)
    return offspring.copy()


def on_start(ga_instance):
    pass
    # for sol in ga_instance.population:
    #     sol = scale(sol,ga_instance)
def on_mutation(ga_instance,offspring_mutation):
    print("on_mutation()")
    # for sol in ga_instance.population:
    #     sol = scale(sol,ga_instance)
def on_fitness(ga_instance, population_fitness):
    print("on_fitness()")

def on_parents(ga_instance, selected_parents):
    print("on_parents()")

def on_crossover(ga_instance, offspring_crossover):
    print("on_crossover()")

def on_generation(ga_instance):
    print("on_generation")
    print(f"Génération {ga_instance.generations_completed} : Meilleure fitness = {ga_instance.mybest_fitness}")
    print(f" Meilleur individu : {ga_instance.mybest_sol}")
    print(ga_instance.generations_completed)
    sep = ","
    d = vars(ga_instance.my_args)
    res = {
        **d,
        "generations_completed":ga_instance.generations_completed,
        "mybest_fitness":ga_instance.mybest_fitness,
    }
    sol = decode_solution(ga_instance.mybest_sol)
    for k,v in sol.items():
        res[k]=v.item()
    imin = np.argmin(ga_instance.mybest_dist)
    imax = np.argmax(ga_instance.mybest_dist)
    for (what,i) in [("min",imin),("max",imax)]:
        res[f"best{what}_dist"]=ga_instance.mybest_dist[i]
        res[f"best{what}_actual_dist"]=DIST_MIN_ACTUAL[i]
        res[f"best{what}_fid"]=FID[i]
        res[f"best{what}_tdeviation"]=TDEVIATION[i]
    if ga_instance.generations_completed==1:
        with open(ga_instance.metric_csv,"w") as file:
            file.write(f"{sep.join(res.keys())}")
            file.write(f"\n{sep.join([str(x) for x in res.values()])}")
    else:
        with open(ga_instance.metric_csv,"a") as file:
            file.write(f"\n{sep.join([str(x) for x in res.values()])}")
    #     file.write(f"Sol: {best_sol}\n")
    #     file.write(f"Fit: {best_fitness}\n")
        #file.write(f" Din moy:{np.mean(dsol)}, ecart type:{np.std(dsol)}\n")
        #file.write(f" Din moy diff:{np.mean([abs(d-5) for d in dsol])}, ecart type nu:{np.std([abs(d-5) for d in dsol])}\n\n")
# num_generations = 10
# num_parents_mating = 40

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='EA',
    )
    parser.add_argument("-csv",default= "stats.csv")
    parser.add_argument("-num_generations",type=int,default= 50)
    parser.add_argument("-random_seed",type=int,default= 42)
    parser.add_argument("-sol_per_pop",type=int,default= 100)
    parser.add_argument("-ratio_num_parents_mating",type=float,default= 0.9)
    parser.add_argument("-K_tournament",type=int,default= 2)
    parser.add_argument("-crossover_probability",type=float,default= 1.)
    parser.add_argument("-mutation_probability",type=float,default= 1.)
    parser.add_argument("-mutation_type",type=str,default= "uniform")
    parser.add_argument("-scale_mutation",type=float,default= 0.1)
    parser.add_argument("-parent_selection_type",type=str,default="tournament")
    parser.add_argument("-metric",type=str,default="abs")
    parser.add_argument("-tau",type=float,default=0.2)
    parser.add_argument("-clip_dist",type=float,default=20)
    # parser.add_argument("-prob_crossover",type=float,default=0)#0.55)
    args = parser.parse_args()
    assert(args.metric in ["square","abs"])
    num_genes = len(params_names)
    ga_instance = pygad.GA(
        num_generations=args.num_generations,
        on_start=on_start,
        on_fitness=on_fitness,
        on_parents=on_parents,
        on_crossover=on_crossover,
        on_mutation=on_mutation,
        on_generation=on_generation,
        random_seed=args.random_seed,
        num_parents_mating=int(args.sol_per_pop*args.ratio_num_parents_mating),
        fitness_func=fitness_func,
        sol_per_pop=args.sol_per_pop,
        parent_selection_type=args.parent_selection_type,
        K_tournament = args.K_tournament,
        fitness_batch_size=min(args.sol_per_pop,100),
        keep_elitism=1,
        num_genes=num_genes,
        gene_space=gene_space,
        crossover_type = custom_crossover,#"scattered",
        mutation_type=custom_mutation,
        #crossover_probability=args.crossover_probability,
        #mutation_probability=args.mutation_probability,
        #    crossover_type=custom_crossover,
    )
    ga_instance.my_args = args
    ga_instance.metric = args.metric
    ga_instance.metric_csv = args.csv
    ga_instance.metric_prob_crossover = args.crossover_probability
    ga_instance.metric_prob_mutation = args.mutation_probability
    ga_instance.metric_dist = np.zeros((args.sol_per_pop,len(FID)),dtype=float)
    ga_instance.metric_fitness = np.zeros((args.sol_per_pop,),dtype=float)
    ga_instance.mylastpop = np.zeros((args.sol_per_pop,num_genes),dtype=float)
    ga_instance.mybest_sol = None
    ga_instance.mybest_fitness = None
    ga_instance.run()
    # ga_instance.plot_fitness()
    # solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Best solution:", decode_solution(ga_instance.mybest_sol))
    print("Fitness:", ga_instance.mybest_fitness)


if __name__ == '__main__':
    main()
