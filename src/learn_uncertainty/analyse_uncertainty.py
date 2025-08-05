import argparse
import os
import numpy as np
import torch
import pandas as pd
from traffic.core import Traffic
from traffic.core import Flight as TrafficFlight
import matplotlib.pyplot as plt
from learn_uncertainty.fit_traj import save_situation, load_situation, SituationDeviated, SituationOthers, SITUATION, OTHERS, deserialize_dict,plot,recplot,scatter_with_number,read_config,Alignment,donothing,NoAlignedAfter,NoAlignedBefore
from learn_uncertainty import read_json
from torchtraj.utils import T, XY,WPTS, apply_mask
from learn_uncertainty.add_uncertainty import VALUESTOTEST, Add_uncertainty
import matplotlib.lines as mlines
import main_ga_torch

def get_original_uparams(device):
    uparams = {
        "dangle": torch.tensor([[0.,0.]],device=device),
        "dt0": torch.tensor([[0,0.]],device=device),
        "dt1": torch.tensor([[0,0.]],device=device),
        "dspeed": torch.tensor([[1.,1]],device=device),
        "ldspeed": torch.tensor([[1.,1]],device=device),
        "vspeed": torch.tensor([[1.,1]],device=device),
    }
    return uparams

def hist_distance(d,fname=None):
    # print(d)
    # print(main_ga_torch.DIST_MIN_ACTUAL)
    fig = plt.figure()
    _,_,patches = plt.hist([d,main_ga_torch.DIST_MIN_ACTUAL],bins=200)
    for pi,li in zip(patches,["With uncertainty","Without uncertainty"]):
        pi.set_label(li)
    plt.xlabel("Minimal distance between deviated trajectory and others [NM]")
    plt.ylabel("Count [-]")
    plt.legend(ncol=1,
               loc='upper right',
               frameon=False,
               columnspacing=0.1,
               fontsize=8,
               )
    if fname is not None:
        fig.set_tight_layout({'pad':0})
        fig.set_figwidth(6)
        plt.savefig(fname, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-statscsv')
    args = parser.parse_args()
    metric ='abs'
    tau = 0.2
    clip_dist=20
    df = pd.read_csv(args.statscsv)
    df = df.query("metric==@metric").query("tau==@tau").query("clip_dist==@clip_dist")
    imax = df.mybest_fitness.idxmax()
    print(imax)
    print(df)
    bestline =df.iloc[imax]
    duparams = main_ga_torch.prepare_dparams([{name: bestline[name] for name in main_ga_torch.params_names}])
    duparams_without =  get_original_uparams("cpu")
    duparams = {k:torch.cat([duparams[k],duparams_without[k]],dim=0) for k in duparams}
    duparams = {k:v.to(main_ga_torch.DEVICE) for k,v in duparams.items()}
    print(duparams)
    d,lid,ltzero = main_ga_torch.modelDistance(duparams)
    scores = main_ga_torch.compute_scores(metric,tau,clip_dist,d)
    d = d.cpu().numpy()
    scores = scores.cpu().numpy()
    print(scores)
    hist_distance(d[0],fname="figures/histdist.pdf")
    print(np.abs(d[1]-main_ga_torch.DIST_MIN_ACTUAL).max())
    def export(i):
        what=["with","without"]
        iduparams = {k:v.cpu().numpy()[i] for k,v in duparams.items()}
        dico={
            "$\epsilon_{\\alpha}$ [^{\\circ}]":np.degrees(iduparams["dangle"][1].item()),
            "$\delta_{t_0}^-$ [s]":iduparams["dt0"][0].item(),
            "$\delta_{t_0}^+$ [s]":iduparams["dt0"][1].item(),
            "$\delta_{t_1}^-$ [s]":iduparams["dt1"][0].item(),
            "$\delta_{t_1}^+$ [s]":iduparams["dt1"][1].item(),
            "$\epsilon_{d}^-$ [-]":iduparams["dspeed"][0].item(),
            "$\epsilon_{d}^+$ [-]":iduparams["dspeed"][1].item(),
            "$\epsilon_{o}^-$ [-]":iduparams["ldspeed"][0].item(),
            "$\epsilon_{o}^+$ [-]":iduparams["ldspeed"][1].item(),
            "$\epsilon_{v}^-$ [-]":iduparams["vspeed"][0].item(),
            "$\epsilon_{v}^+$ [-]":iduparams["vspeed"][1].item(),
            "$D_{\Omega}$ [NM]":scores[i].item(),
            "uncertainty":what[i],
        }
        return pd.DataFrame({k:[v] for k,v in dico.items()})
    pf = pd.concat([export(0),export(1)]).set_index("uncertainty")
    names = list(pf)
    print(names)
    with open("./figures/uparams.tex",'w') as f:
        f.write(pf[names[:-1]].to_latex(float_format="{:.2f}".format))
    with open("./figures/uparamsperfo.tex",'w') as f:
        f.write(pf[names[-1:]].to_latex(float_format="{:.2f}".format))
    # print(d)
    # print(main_ga_torch.DIST_MIN_ACTUAL)

main()
