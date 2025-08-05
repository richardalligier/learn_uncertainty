import numpy as np

FOLDER = "stats"
lcmd = [f"mkdir -p {FOLDER}"]

np.random.seed(0)
n = 10000
d ={
    "num_generations": 300,
    "sol_per_pop": 100,
    "tau":0.2,
    "metric":"abs",
    "clip_dist":20,
}

parent_selection=["rws","tournament","rank"]
mutation_type = ["uniform","gaussian"]


dtotest = {
    "scale_mutation": np.random.uniform(low=0.01,high=0.2,size=n),
    "K_tournament": np.random.uniform(low=0.01,high=0.2,size=n),
    "parent_selection_type":[parent_selection[i] for i in np.random.randint(0,len(parent_selection),size=n)],
    "ratio_num_parents_mating":np.random.uniform(low=0.8,high=0.95,size=n),
    "mutation_probability":np.random.uniform(low=0.5,high=1,size=n),
    "mutation_type":[mutation_type[i] for i in np.random.randint(0,len(mutation_type),size=n)],
    "crossover_probability":np.random.uniform(low=0.5,high=1,size=n),
}
print(dtotest)
for i in range(10):
    lcmd.append("python3 main_ga_torch.py -csv {FOLDER}/stats{i}.csv")
