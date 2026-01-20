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
from pathlib import Path

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(100)

import time
import model


from model.code.hh_transmission_model import (Params,
                                     run_SEIR_model, 
                                     save_results,
                                     load_params)
from model.code.analysis_outputs import plot_SEIR

if __name__ == "__main__":

    repo_path = __file__#
    repo_path = os.path.dirname(os.path.abspath(os.path.join(repo_path)))
    if os.getcwd() != str(Path(repo_path).parent):
        os.chdir(os.path.join(str(Path(repo_path).parent)))
        #rint(sys.path)
    sys.path.append(repo_path)
    sys.path.append(os.path.join(repo_path, 'model/code'))
    
    params = load_params("configs/exemplar_covid_params.txt")
    #print(params)
    
    ####RUN THE SIMULATION
    results = run_SEIR_model(params)
    
    save_results(results, params, output_dir = "output/covid")
    
    output_dir = "output/covid"
    #load params
    with open(f"{output_dir}/params.txt", "rb") as fout:
        #params = json.loads(fout.read())
        params = json.load(fout)
    
    #read outputs
    results = {}
    
    if os.path.isfile(os.path.join(output_dir, 'transmission.csv')): 
        results["all_transmission"] = pl.read_csv(os.path.join(output_dir, 
                                                      'transmission.csv'))
        plot_SEIR(results["all_transmission"])
    
    #secondary_infections_from_seed_infection_list, exposed_by_seed_df, 
    #all_records = run_SEIR_model(params)
