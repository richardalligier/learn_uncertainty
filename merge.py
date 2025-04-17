import os
from fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict


def merge(lsit):
    d = {k:[s[k] for s in lsit] for k in lsit[0]}
    for k,v in d.items():
        d[k] = type(v[0]).cat(v,dimname=SITUATION)
    return d

def partition(criteria,l):
    d = {}
    for x in l:
        k = criteria(x)
        if k in d:
            d[k].append(x)
        else:
            d[k]=[x]
    return d

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-situationsfolder')
    parser.add_argument('-dsituation')
    args = parser.parse_args()
    lfname = [f"{args.situationsfolder}/{fname}" for fname in os.listdir(args.situationsfolder)]
    lsit = [load_situation(fname) for fname in lfname]
    print(f"{len(lsit)=}")
    dlsit = partition(lambda sit: sit["others"].t.shape[sit["others"].t.names.index(OTHERS)],lsit)
    for k,v in dlsit.items():
        dlsit[k]=merge(v)
    print(dlsit)
    save_situation(dlsit,args.dsituation)


if __name__ == "__main__":
    main()
