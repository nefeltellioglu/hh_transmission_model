#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 08:55:46 2026

@author: tellioglun
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom
import polars as pl
import random
from dataclasses import dataclass, asdict, fields

import matplotlib.pyplot as plt
import os, sys
from scipy.stats import lognorm

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(100)

import time


repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
code_path = os.path.abspath(os.path.join(repo_path, "src/model"))

if code_path not in sys.path:
    sys.path.append(code_path)

os.chdir(os.path.join(code_path))
    
if __name__ == "__main__":
    
    """   
    pathogen_params = pl.read_csv("data/generation_interval_means_and_sds_for_pathogens.csv") 
    pathogen_params = pathogen_params.rename({"V1": "disease",
                                              "V2": "mean",
                                              "V3": "var"})
    
    for i in range(pathogen_params.height):
        meanlog = pathogen_params["mean"][i]
        sdlog = np.sqrt(pathogen_params["var"][i])
        mean_X, sd_X = mean_sd_to_log_params(mean, sdlog)
        print("Mean:", mean_X)
        print("SD:", sd_X)
    
    """
    
    pathogen_params = pl.read_csv("data/meanlog_sdlogs_for_pathogens.csv") 
    
    def mean_sd_to_log_params(mean, sd):
        """
        Convert mean and sd of X to meanlog and sdlog of log(X)
        """
        sdlog = np.sqrt(np.log(1 + (sd / mean)**2))
        meanlog = np.log(mean) - 0.5 * sdlog**2
        return meanlog, sdlog

    def log_params_to_mean_sd(meanlog, sdlog):
        mean_X = np.exp(meanlog + 0.5 * sdlog**2)
        sd_X   = np.sqrt((np.exp(sdlog**2) - 1) * np.exp(2*meanlog + sdlog**2))
        return mean_X, sd_X
    
    # Plot
    plt.figure(figsize=(6, 4))
    #for i, (label, (meanlog, sdlog)) in enumerate(pars.items()):
    for i in range(pathogen_params.height):
        meanlog = pathogen_params["meanlog"][i]
        sdlog = pathogen_params["sdlog"][i]
        mean_X, sd_X = log_params_to_mean_sd(meanlog, sdlog)
        print("Mean:", mean_X)
        print("SD:", sd_X)
        label = pathogen_params["name"][i]
        # x range (positive support)
        x = np.linspace(0.001, 30, 1000)
        #meanlog, sdlog = mean_sd_to_log_params(mean, sd)
        
        # scipy parameterization:
        # shape = sdlog, scale = exp(meanlog)
        dist = lognorm(s=sdlog, scale=np.exp(meanlog), loc = 0)
        pdf = dist.pdf(x)
        plt.plot(x, pdf, lw=2, label = label)
    plt.xlabel('x')
    plt.ylabel('PDF')
    plt.legend()
    plt.tight_layout()
    plt.show()
        
    try_others = False
    
    if try_others:
        # Support
        k = np.arange(0, 50)
        
        # ---- First distribution ----
        r1, p1 = 5, 0.4
        dist1 = nbinom(r1, p1)
        pmf1 = dist1.pmf(k)
        peak = pmf1.max()
        
        # ---- Search for second distribution with same peak ----
        r_vals = np.linspace(1, 30, 300)
        p_vals = np.linspace(0.9, 1.5, 300)
        
        best_diff = np.inf
        best_params = None
        
        for r in r_vals:
            for p in p_vals:
                pmf = nbinom(r, p).pmf(k)
                diff = abs(pmf.max() - peak)
                if diff < best_diff and abs(r - r1) > 0.5:
                    best_diff = diff
                    best_params = (r, p)
                    best_pmf = pmf
        
        r2, p2 = best_params
        dist2 = nbinom(r2, p2)
        
        # ---- Plot ----
        plt.figure(figsize=(7, 4))
        plt.plot(k, pmf1, 'o-', label=f'Dist 1: r={r1}, p={p1}, mean={dist1.mean():.2f}')
        plt.plot(k, best_pmf, 's-', label=f'Dist 2: r={r2:.2f}, p={p2:.2f}, mean={dist2.mean():.2f}')
        plt.xlabel('k')
        plt.ylabel('PMF')
        plt.title('Negative Binomial Distributions\nSame Peak Height, Different Means')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        print("Peak height:", peak)
        print("Mean 1:", dist1.mean())
        print("Mean 2:", dist2.mean())
        
    
    
    
   