#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:19:01 2024

@author: nefeltellioglu
"""

import polars as pl
import numpy as np
import random
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import os, sys
import json

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(100)

import time

repo_path = "." # __file__#
repo_path = os.path.dirname(os.path.abspath(os.path.join(repo_path)))

if os.getcwd() != repo_path:
    os.chdir(os.path.join(repo_path))
    print(sys.path)


#####plot transmission
def plot_SEIR(transmission_df) -> None:
    
    
    fig, ax = plt.subplots(figsize=(7,5))
    unique_runs = transmission_df["run_no"].unique().to_list()

    for r in unique_runs:
        cur_df = transmission_df.filter(pl.col("run_no") == r).sort("t")

        for state, color, label in zip(["Susceptible", "Infectious", "Exposed", "Recovered"],
                                       ["tab:green", "tab:red", "tab:orange", "tab:blue"],
                                       ["Susceptible", "Infectious", "Exposed", "Recovered"]):
            state_df = cur_df.filter(pl.col("state") == state)
            plt.plot(state_df["t"].to_numpy(), state_df["count"].to_numpy(), color=color, label=label)# if r == unique_runs[0] else "")

    plt.xlabel("Time (days)")
    plt.ylabel("Number of Individuals")
    #plt.title("Disease Spread Dynamics")
    plt.grid()

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc = "upper right")

    plt.show()
    # fig.savefig('plot_SEIR.tiff', bbox_inches="tight", dpi=300)
    

@dataclass
class Params:
    no_runs: int  
    hh_size: int 
    hh_size_distribution: str
    inf_duration: float
    exposed_duration:float 
    random_seed: int 
    #R0: float
    transmission_rate: float
    time_horizon: float
    time_step: float
    record_transmission: bool
    record_all_new_cases: bool

if __name__ == "__main__":
    output_dir = "output/covid"
    #load params
    with open(f"{output_dir}/params.txt", "rb") as fout:
        #params = json.loads(fout.read())
        params = json.load(fout)
    
    #read outputs
    results = {}
    results["exposed_by_seed"] = pl.read_csv(os.path.join(f'{output_dir}/exposed_by_seed.csv'))
    results["all_exposed_cases"] = pl.read_csv(os.path.join(f'{output_dir}/all_exposed_cases.csv'))
    
    if os.path.isfile(os.path.join(output_dir, 'transmission.csv')): 
        results["all_transmission"] = pl.read_csv(os.path.join(output_dir, 
                                                      'transmission.csv'))
    if os.path.isfile(os.path.join(output_dir, 'sar.csv')): 
        results["sar"] = pl.read_csv(os.path.join(output_dir, 'sar.csv'))
    
        agg_sar = results["sar"].select([
            pl.col("SAR").mean().alias("mean_SAR"),
            pl.col("SAR").median().alias("median_SAR"),
            pl.col("SAR").quantile(0.0).alias("quantile_0.0_SAR"),
            pl.col("SAR").quantile(0.1).alias("quantile_0.1_SAR"),
            pl.col("SAR").quantile(0.25).alias("quantile_0.25_SAR"),
            pl.col("SAR").quantile(0.5).alias("quantile_0.5_SAR"),
            pl.col("SAR").quantile(0.75).alias("quantile_0.75_SAR"),
            pl.col("SAR").quantile(0.9).alias("quantile_0.9_SAR"),
            pl.col("SAR").quantile(1).alias("quantile_1.0_SAR"),
        ])
        print(agg_sar)
        agg_sar.write_csv(os.path.join(f'{output_dir}/sar_summary.csv'))     
        
    
    #ALL no of transmissions in HHs
    #not just secondary cases, all cases
    no_transmissions = results["all_exposed_cases"].filter(
        pl.col("exposed_from") != -1).group_by("hh_size", "run_no").agg(
            pl.count() )
    
    no_transmissions = results["all_exposed_cases"].group_by("hh_size", "run_no").agg(
            (pl.col("exposed_from").ne(-1)).sum().alias("no_transmissions"))
    
    #transmissions occurred
    print(no_transmissions.select((pl.col("no_transmissions").ne(0)).sum().alias("transmissions"),
                            (pl.col("no_transmissions").eq(0)).sum().alias("zero_transmissions")))
    
    #transmissions occurred by hh size
    transmissions_hh = no_transmissions.group_by("hh_size").agg(
        (pl.col("no_transmissions").ne(0)).sum().alias("transmissions"),
         (pl.col("no_transmissions").eq(0)).sum().alias("zero_transmissions")
        ).sort("transmissions", "hh_size")
    print(transmissions_hh)
    #no of transmissions by hh size
    no_transmissions_hh = no_transmissions.group_by("hh_size", 
                             "no_transmissions").agg(pl.count()).sort("hh_size", 
                                                            "no_transmissions")
    print(no_transmissions_hh)
    
    #plot transmissions   
    if "all_transmission" in results:
        plot_SEIR(results["all_transmission"])

