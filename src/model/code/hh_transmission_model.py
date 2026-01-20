#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 11:19:01 2024

@author: nefeltellioglu
"""

import polars as pl
import numpy as np
import random
from dataclasses import dataclass, asdict, fields

import matplotlib.pyplot as plt
import os, sys
import json
from pathlib import Path
import logging

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(100)

import time


repo_path = "." #__file__#
repo_path = os.path.dirname(os.path.abspath(os.path.join(repo_path)))

if os.getcwd() != repo_path:
    os.chdir(os.path.join(repo_path))
    print(sys.path)


@dataclass
class Params:
    no_runs: int  
    hh_size: int 
    hh_size_distribution: str
    inf_duration_gamma_mean: float
    inf_duration_gamma_shape: int
    exposed_duration:float 
    random_seed: int 
    #R0: float
    transmission_rate: float
    time_horizon: float
    time_step: float
    record_transmission: bool
    record_all_new_cases: bool


def run_SEIR_model(p: Params):
    
    st = time.time()
    
    secondary_infections_from_seed_infection_list = []
    exposed_by_seed_df = pl.DataFrame()
    all_exposed_cases = pl.DataFrame()
    rng = np.random.RandomState(p.random_seed)
    ts = np.arange(p.time_step, p.time_horizon, p.time_step)    
    transmission_rate = 1 - ((1 - p.transmission_rate)**(p.time_step))
    inf_duration_gamma_scale = (p.inf_duration_gamma_mean/
                                  p.inf_duration_gamma_shape)
    
    
    1 - ((p.transmission_rate)**(p.time_step))
    
    
    if p.hh_size_distribution != "constant":
        hh_sizes = np.loadtxt("data/hh_sizes.csv", delimiter=",")
        # Separate columns
        hh_sizes_loaded = hh_sizes[:, 0].astype(int)       # as integer NumPy array
        hh_size_prop_loaded = hh_sizes[:, 1]               # as float NumPy array
        
        # Optionally convert to Python lists
        hh_sizes = hh_sizes_loaded.tolist()
        hh_size_prop = hh_size_prop_loaded.tolist()

        all_hh_sizes = rng.choice(hh_sizes, size = p.no_runs, 
                                  p = hh_size_prop)
    else:
        all_hh_sizes = [p.hh_size] * p.no_runs
    
    if p.record_transmission:
        possible_states = pl.DataFrame(
           { "state": ["Susceptible", "Exposed", "Infectious", "Recovered"],
            "count":  [0,0,0,0]}
            )
        all_records = pl.DataFrame()
    
    for run in range(p.no_runs):
        secondary_infections_from_seed_infection = 0
        cur_hh_size = all_hh_sizes[run]
        # Initialize household
        household = pl.DataFrame(
        {
        "id": range(cur_hh_size),
        "state": ["Susceptible"] * cur_hh_size,
        "s_time_exposed": [0.0] * cur_hh_size,
        "s_time_infectious": [0.0] * cur_hh_size,
        "s_time_recovery": [0.0] * cur_hh_size,
        "exposed_from": [-1] * cur_hh_size,
        "hh_size": cur_hh_size,
        "hh_id": run,
        "run_no": run,
        }
        )
        #if p.record_transmission:
        #    cur_records = pl.DataFrame()
        
        # Initialize household
        cur_exposed_by_seed_df = pl.DataFrame()
        #cur_all_exposed_cases = pl.DataFrame()
        
        # Infect one individual in the household (seed infection)
        
        # Calculate time until seed infection goes into the Recovered state
        t_seed_recovery = rng.gamma(p.inf_duration_gamma_shape, 
                                    inf_duration_gamma_scale)
        
        # Infect one individual in the household (seed infection)
        seed_infection_index = 0#rng.randint(0, cur_hh_size - 1)
        household = household.with_columns(
        (pl.when(pl.col("id") == seed_infection_index)
        .then(pl.lit("Infectious"))
        .otherwise(pl.col("state"))).alias("state"),
        
        (pl.when(pl.col("id") == seed_infection_index)
        .then(rng.gamma(p.inf_duration_gamma_shape, 
                        inf_duration_gamma_scale))
        .otherwise(pl.col("s_time_recovery"))).alias("s_time_recovery"),
        )
        
        if p.record_transmission:
            cur_records = possible_states.update(
                household.group_by(pl.col("state")).agg(
                pl.count()), on = ["state"], how = "left").with_columns(
                    pl.lit(ts[0] - p.time_step).alias("t"))
         
        if p.record_all_new_cases:
            cur_all_exposed_cases = household.filter(
                            (pl.col("state") == "Infectious"))
        
        cur_exposed_by_seed_df = household.filter(
                        (pl.col("state") == "Infectious"))
        
        
        # Simulate transmission in the household
        for t in ts:
            infected_ids = household.filter(
                pl.col("state") == "Infectious")["id"]
            if not household.filter(
                pl.col("state").is_in(["Infectious", "Exposed"])).height:
                break
            susceptible_individuals = household.filter(
                pl.col("state") == "Susceptible")
            will_infected_individuals = susceptible_individuals.with_columns(
                 pl.Series(rng.rand(susceptible_individuals.height) 
                     < (transmission_rate * len(infected_ids)))
                 .alias("will_infected")
                    ).filter(pl.col("will_infected")).drop("will_infected")
            s_time_infectious = pl.Series("s_time_infectious", 
                                t + rng.gamma(
                                    p.inf_duration_gamma_shape, 
                                    inf_duration_gamma_scale,
                                    will_infected_individuals.height) ) 
                                
            will_infected_individuals = will_infected_individuals.with_columns(
                pl.lit(t).alias("s_time_exposed"),
                pl.lit(s_time_infectious).alias("s_time_infectious"),
                pl.Series("s_time_recovery", 
                        s_time_infectious + rng.gamma(
                            p.inf_duration_gamma_shape, 
                            inf_duration_gamma_scale,
                            will_infected_individuals.height)),
                pl.Series("exposed_from", 
                         rng.choice(infected_ids, 
                                    size = will_infected_individuals.height)),
                pl.lit("Exposed").alias("state"),
                )
            
            household = household.update(will_infected_individuals, 
                                         on = "id", how= "left")
            
            #I -> R state transition
            household = household.with_columns(
                pl.when((pl.col("state") == "Infectious" ) & 
                        ( pl.col("s_time_recovery") < t))
                .then(pl.lit("Recovered")).otherwise(pl.col("state"))
                .alias("state"),
                )
            
            #E -> I state transition
            household = household.with_columns(
               pl.when((pl.col("state") == "Exposed" ) & 
                       (pl.col("s_time_infectious") < t))
                .then(pl.lit("Infectious")).otherwise(pl.col("state"))
                .alias("state"),
                )
            
            
            new_infs_from_seed = household.filter(
                            (pl.col("s_time_exposed") == t) &
                             (pl.col("exposed_from") == seed_infection_index)
                             )
            if p.record_all_new_cases:
                new_exposed_cases = household.filter(
                                (pl.col("s_time_exposed") == t))
                
            cur_exposed_by_seed_df = cur_exposed_by_seed_df.vstack(new_infs_from_seed)
            secondary_infections_from_seed_infection += new_infs_from_seed.height
            
            if p.record_all_new_cases:
                #record all new exposed cases
                cur_all_exposed_cases = cur_all_exposed_cases.vstack(new_exposed_cases)
            #record transmissions
            if p.record_transmission:
                cur_records =  cur_records.vstack(possible_states.update(
                    household.group_by(pl.col("state")).agg(
                    pl.count()), on = ["state"], how = "left").with_columns(
                        pl.lit(t).alias("t")))
            
        secondary_infections_from_seed_infection_list.append(secondary_infections_from_seed_infection)
        if cur_exposed_by_seed_df.height:
            #cur_exposed_by_seed_df = cur_exposed_by_seed_df.with_columns(
            #    pl.Series("run_no", [run] * cur_exposed_by_seed_df.height))
            exposed_by_seed_df = exposed_by_seed_df.vstack(cur_exposed_by_seed_df)
        if p.record_all_new_cases and cur_all_exposed_cases.height:
            #cur_all_exposed_cases = cur_all_exposed_cases.with_columns(
            #    pl.Series("run_no", [run] * cur_all_exposed_cases.height))
            all_exposed_cases = all_exposed_cases.vstack(cur_all_exposed_cases)
        
        
        
        if p.record_transmission:
            all_records =  all_records.vstack(cur_records.with_columns(
                pl.Series("run_no", [run] * cur_records.height),
                ))
    if p.record_all_new_cases:    
        #calculate SAR
        sar = all_exposed_cases.filter(pl.col("exposed_from") == 0).group_by(
                    "run_no", "hh_id").agg(pl.count().alias("secondary_cases"), 
                                           pl.col("hh_size").first(),
                                SAR = pl.count()/(pl.col("hh_size").first() - 1))
        all_sar = (all_exposed_cases.select(["run_no", "hh_id", "hh_size"])
                   .unique().with_columns(pl.lit(0).alias("secondary_cases"),
                                          pl.lit(0).alias("SAR"))
                   .update(sar, on = ["run_no", "hh_id", "hh_size"], how = "left"))
    
    results = {"no_secondary_cases": secondary_infections_from_seed_infection_list, 
               "hh_sizes": all_hh_sizes,
                  "exposed_by_seed": exposed_by_seed_df,
                  }
    if p.record_transmission:
        results["all_transmission"] = all_records

    if p.record_all_new_cases:
        results["all_exposed_cases"] = all_exposed_cases
        results["sar"] = all_sar
    
    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')   
        
    return results

def save_results(results, params, output_dir = "output"):
    """
    Save results of a run in multiple csv files as well as the params set
    
    inspect the results:
        
    np.mean(results['no_secondary_cases'])
    results["exposed_by_seed"] 
    if params.record_transmission:
        results["all_transmission"] 
    print(f"No of secondary infections from seed in each run:
          {results['no_secondary_cases']}")
    """
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    if params.record_all_new_cases:
        results["all_exposed_cases"] 
        results["sar"]
        agg_sar = results["sar"].select([
            pl.col("SAR").mean().alias("mean_SAR"),
            pl.col("SAR").median().alias("median_SAR"),
            pl.col("SAR").quantile(0.025).alias("quantile_0.025_SAR"),
            pl.col("SAR").quantile(0.1).alias("quantile_0.1_SAR"),
            pl.col("SAR").quantile(0.25).alias("quantile_0.25_SAR"),
            pl.col("SAR").quantile(0.75).alias("quantile_0.75_SAR"),
            pl.col("SAR").quantile(0.9).alias("quantile_0.9_SAR"),
            pl.col("SAR").quantile(0.975).alias("quantile_0.975_SAR"),
        ])
        print(agg_sar)
        results["sar"].write_csv(os.path.join(f'{output_dir}/sar.csv'))
        agg_sar.write_csv(os.path.join(f'{output_dir}/sar_summary.csv'))     
        (results["all_exposed_cases"]
         .write_csv(os.path.join(f'{output_dir}/all_exposed_cases.csv')))
    
    (results["exposed_by_seed"]
     .write_csv(os.path.join(f'{output_dir}/exposed_by_seed.csv')))
    if params.record_transmission:
        (results["all_transmission"]
         .write_csv(os.path.join(f'{output_dir}/transmission.csv')))
    
    #save params
    with open(f'{output_dir}/params.txt', 'w') as fout:
        json_dumps_str = json.dumps(asdict(params), indent=4)
        print(json_dumps_str, file=fout)


def load_params(path: str | Path) -> Params:
    """
    Load parameters from a JSON file and return a Params instance.
    If 'inf_duration_gamma_shape' is missing, default_shape is used.
    
    Example sets:
        
    #covid-like
    params = Params(no_runs = 5000,
                    hh_size = 5,
                    #hh_size_distribution = "constant",
                    hh_size_distribution = "distribution",
                    inf_duration_gamma_mean = 7.68, 
                    inf_duration_gamma_shape = 3,
                    exposed_duration = 5.2,
                    random_seed = 10,
                    transmission_rate = 0.0165,#0.0164,#0.0276, 0.0268
                    time_horizon = 80,
                    time_step = 0.01,
                    record_transmission= False,
                    record_all_new_cases = True)
    
    #another example
    params = Params(no_runs = 20,
                    hh_size = 500,
                    hh_size_distribution = "constant",
                    #hh_size_distribution = "distribution",
                    inf_duration_gamma_mean = 3, 
                    inf_duration_gamma_shape = 3,
                    exposed_duration = 2,
                    random_seed = 10,
                    transmission_rate = 0.001,#0.0164,#0.0276, 0.0268
                    time_horizon = 300,
                    time_step = 1,
                    record_transmission= True,
                    record_all_new_cases = False)

    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Parameter file not found: {p}")

    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    # Validate required fields (all dataclass fields)
    required = {f.name for f in fields(Params)}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Missing required parameter keys in {p}: {missing}")

    # Construct Params (dataclass will cast types in __post_init__)
    params = Params(**{k: data[k] for k in required})
    return params


if __name__ == "__main__":
    
    
    #covid-like
    params = Params(no_runs = 5000,
                    hh_size = 5,
                    #hh_size_distribution = "constant",
                    hh_size_distribution = "distribution",
                    inf_duration_gamma_mean = 7.68, 
                    inf_duration_gamma_shape = 3,
                    exposed_duration = 5.2,
                    random_seed = 10,
                    transmission_rate = 0.0165,#0.0164,#0.0276, 0.0268
                    time_horizon = 80,
                    time_step = 0.01,
                    record_transmission= False,
                    record_all_new_cases = True)
    
    
    #another example
    params = Params(no_runs = 20,
                    hh_size = 500,
                    hh_size_distribution = "constant",
                    #hh_size_distribution = "distribution",
                    inf_duration_gamma_mean = 3, 
                    inf_duration_gamma_shape = 3,
                    exposed_duration = 2,
                    random_seed = 10,
                    transmission_rate = 0.001,#0.0164,#0.0276, 0.0268
                    time_horizon = 300,
                    time_step = 1,
                    record_transmission= True,
                    record_all_new_cases = False)
    
    params = load_params("data/covid_params.txt")
    print(params)
    
    ####RUN THE SIMULATION
    results = run_SEIR_model(params)
    
    save_results(results, params, output_dir = "output/covid")
    
    #secondary_infections_from_seed_infection_list, exposed_by_seed_df, 
    #all_records = run_SEIR_model(params)
