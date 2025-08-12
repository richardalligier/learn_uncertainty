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
import glob

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
    _,_,patches = plt.hist([di for di in d],bins=200)
    for pi,li in zip(patches,["with uncertainty","without uncertainty"]):#,"no uncertainty/buffer"]):
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
        plt.clf()
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='fit trajectories and save them in folders',
    )
    parser.add_argument('-statscsv')
    args = parser.parse_args()
    metric ='square'
    tau = 0.5
    clip_dist=10
    argsnd = False
    all_files = glob.glob(os.path.join(args.statscsv, "*.csv"))
    # df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
    df = pd.read_csv(args.statscsv)
    df = df.query("metric==@metric").query("tau==@tau").query("clip_dist==@clip_dist").query("nd==@argsnd")
    imax = df.mybest_fitness.idxmax()
    # tau = 0.1
    # clip_dist=10
    print(imax)
    print(df)
    bestline =df.iloc[imax]
    print(bestline)
    if True:
        duparams = main_ga_torch.prepare_dparams([{name: bestline[name] for name in main_ga_torch.params_names}])
        duparams_without =  get_original_uparams(main_ga_torch.DEVICE)
        duparams = {k:torch.cat([duparams[k],duparams_without[k]],dim=0) for k in duparams}
        duparams = {k:v.to(main_ga_torch.DEVICE) for k,v in duparams.items()}
        print(duparams)
        d,lid,ltzero = main_ga_torch.modelDistance(duparams)
        # mask = (d[-1]<10).rename(None)
        # indices = torch.arange(0,mask.shape[0],device=mask.device)[mask].cpu()
        # print(indices)
        d = d.cpu()
        print(main_ga_torch.DIST_MIN_ACTUAL,d[1])
        print(main_ga_torch.DIST_MIN_ACTUAL-d[1].numpy())
        diff = d[1].numpy()- main_ga_torch.DIST_MIN_ACTUAL
        def printi(i):
            print(main_ga_torch.FID[i],main_ga_torch.TDEVIATION[i])
        printi(diff.argmax())
        printi(diff.argmin())
        printi(d[1].numpy().argmin())
        printi(main_ga_torch.DIST_MIN_ACTUAL.argmin())
        plt.scatter(main_ga_torch.DIST_MIN_ACTUAL,d[1].numpy())
        # plt.show()
        plt.savefig("distxdist.pdf", dpi=300, bbox_inches='tight')
        plt.clf()
        # print(mask.shape)
        print(d.shape)
        def select(v,ind):
            return torch.tensor(torch.index_select(v.rename(None),dim=1,index=ind).cpu().numpy(),device=v.device,dtype=v.dtype).rename(*v.names)
        # d=select(d,indices)#torch.tensor(torch.index_select(d.rename(None),1,indices).cpu().numpy()).rename(*d.names)
        #nd=select(nd,indices)#torch.tensor(torch.index_select(nd.rename(None),1,indices).cpu().numpy()).rename(*nd.names)
        scores = main_ga_torch.compute_scores(metric,tau,clip_dist,argsnd,d)
        scores = scores.cpu().numpy()
        if argsnd:
            nd = torch.mean(d,dim=-1,keepdim=True).numpy()
        else:
            nd = 5
        d = d.numpy()
        mu = np.mean(np.abs(d-nd),axis=-1)
        std = np.std(np.abs(d-nd),axis=-1)
        print(scores)
        print(mu)
        print(std)
        #check en filtrant sur d
        hist_distance(d,fname="figures/histdist.pdf")
        #print(np.abs(d[1]-main_ga_torch.DIST_MIN_ACTUAL).max())
        def export(i):
            what=["with","without"]
            iduparams = {k:v.cpu().numpy()[i] for k,v in duparams.items()}
            dico={
                "$\epss{-}$ [-]":iduparams["dspeed"][0].item(),
                "$\epss{+}$ [-]":iduparams["dspeed"][1].item(),
                "$\epsv{-}$ [-]":iduparams["vspeed"][0].item(),
                "$\epsv{+}$ [-]":iduparams["vspeed"][1].item(),
                "$\dtzero{-}$ [s]":iduparams["dt0"][0].item(),
                "$\dtzero{+}$ [s]":iduparams["dt0"][1].item(),
                "$\dtone{-}$ [s]":iduparams["dt1"][0].item(),
                "$\dtone{+}$ [s]":iduparams["dt1"][1].item(),
                "$\epsa{+} [^{\\circ}]$":np.degrees(iduparams["dangle"][1].item()),
                # "$\epsilon_{o}^-$ [-]":iduparams["ldspeed"][0].item(),
                # "$\epsilon_{o}^+$ [-]":iduparams["ldspeed"][1].item(),
                "$D_{\Omega} [NM^2]$":scores[i].item(),
                "$\mu$ [NM]":mu[i].item(),
                "$\sigma$ [NM]":std[i].item(),
                "uncertainty":what[i],
            }
            return pd.DataFrame({k:[v] for k,v in dico.items()})
        pf = pd.concat([export(0),export(1)]).set_index("uncertainty")
        names = list(pf)
        print(names)
        with open("./figures/uparams.tex",'w') as f:
            f.write(pf[names[:-3]].to_latex(float_format="{:.3f}".format))
        with open("./figures/uparamsperfo.tex",'w') as f:
            f.write(pf[names[-3:]].to_latex(float_format="{:.2f}".format))
    if True:
        sol = {name: bestline[name] for name in main_ga_torch.params_names}
        # sol = {name:0. for name in main_ga_torch.params_names}
        # sol["nd"]=5.
        # sol["dspeedI"]=1.
        # sol["dspeedS"]=1.
        # sol["vspeedI"]=1.
        # sol["vspeedS"]=1.
        n = 100
        for k in main_ga_torch.params_names:#["nd"]:
            print(k)
            sols = []
            # main_ga_torch.param_bounds["dt0S"]=(0,150)
            totest= np.linspace(*main_ga_torch.param_bounds[k],n)
            for i in range(n):
                soltotest=sol.copy()
                soltotest[k]=totest[i]
                sols.append(soltotest)
            duparams = main_ga_torch.prepare_dparams(sols)
            duparams = {k:duparams[k] for k in duparams}
            duparams = {k:v.to(main_ga_torch.DEVICE) for k,v in duparams.items()}
            #print(duparams)
            d,lid,ltzero = main_ga_torch.modelDistance(duparams)
            scores = main_ga_torch.compute_scores(metric,tau,clip_dist,argsnd,d).cpu().numpy()
            sign = -1 if k=="dangleI" else 1
            plt.plot(sign*totest,scores)
            plt.axvline(x=sign*(sol[k] if isinstance(sol[k],(int,float)) else sol[k].item()))
            plt.savefig(f"figures/optim/{k}.pdf", dpi=300, bbox_inches='tight')
            plt.clf()
            print(scores)
            # raise Exception
    # duparams = main_ga_torch.prepare_dparams([])
    # print(d)
    # print(main_ga_torch.DIST_MIN_ACTUAL)

main()
