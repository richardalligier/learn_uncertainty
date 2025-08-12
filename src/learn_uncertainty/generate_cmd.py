import numpy as np

FOLDER = "stats"
lcmd = [f"mkdir -p {FOLDER}"]

np.random.seed(0)
n = 10000
d ={
    "num_generations": 150,
    "sol_per_pop": 200,
    "tau":0.5,
    "metric":"square",
    "clip_dist":10,
}

parent_selection=["rws","tournament","rank"]
mutation_type = ["uniform","gaussian"]


dtotest = {
    "scale_mutation": np.random.uniform(low=0.01,high=0.2,size=n),
    "K_tournament": np.random.randint(1,5,size=n),
    "parent_selection_type":[parent_selection[i] for i in np.random.randint(0,len(parent_selection),size=n)],
    "ratio_num_parents_mating":np.random.uniform(low=0.8,high=0.95,size=n),
    "mutation_probability":np.random.uniform(low=0.5,high=1,size=n),
    "mutation_type":[mutation_type[i] for i in np.random.randint(0,len(mutation_type),size=n)],
    "crossover_probability":np.random.uniform(low=0.5,high=1,size=n),
}
print(dtotest)
def convert(x):
    if isinstance(x,str) or isinstance(x,int):
        return x
    else:
        return x.item()
for i in range(10,11):
    for k,v in dtotest.items():
        d[k]=convert(v[i])
    print(d)
    options = " ".join([f"-{k} {v}" for k,v in d.items()])
    lcmd.append(f"python3 main_ga_torch.py -csv {FOLDER}/stats{i}.csv {options}")

print(" && ".join(lcmd))
