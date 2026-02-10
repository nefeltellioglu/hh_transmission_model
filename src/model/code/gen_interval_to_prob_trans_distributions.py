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
from matplotlib.lines import Line2D

pl.Config.set_tbl_rows(50)
pl.Config.set_tbl_cols(100)

import time


repo_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

if repo_path not in sys.path:
    sys.path.append(repo_path)

os.chdir(os.path.join(repo_path))
    
if __name__ == "__main__":
    
    
    
    # ----- PARAMETERS -----
    mu = 2               # mean of log(t)
    sigma = 0.3            # std dev of log(t)
    R0 = 2.5               # basic reproduction number (optional)
    contacts_per_day = 10  # average number of susceptible contacts per day
    max_days = 20          # max days to consider
    
    # ----- CONTINUOUS GENERATION INTERVAL PDF -----
    t_continuous = np.linspace(0.001, max_days, 1000)  # avoid t=0
    g_t = lognorm.pdf(t_continuous, s=sigma, scale=np.exp(mu))
    
    # ----- DISCRETIZE INTO DAILY PROBABILITIES -----
    daily_prob = np.zeros(max_days)
    for day in range(max_days):
        mask = (t_continuous >= day) & (t_continuous < day + 1)
        daily_prob[day] = np.trapz(g_t[mask], t_continuous[mask])
    
    # Normalize to sum to 1 (conditional probability)
    daily_prob /= daily_prob.sum()
    
    # ----- CONVERT TO TRANSMISSION PROBABILITY PER CONTACT -----
    # Given R0 and average contacts per day, compute per-contact transmission probability per day
    # R0 ≈ contacts_per_day * sum_over_days(p_day_per_contact)
    # So per-day absolute probability of transmission per contact:
    daily_prob_per_contact = R0 * daily_prob / contacts_per_day
    
    # Expected secondary infections per day (if you want)
    daily_expected_secondary = daily_prob_per_contact * contacts_per_day
    
    # ----- PLOTTING -----
    plt.figure(figsize=(8,4))
    
    # Continuous generation interval PDF
    line1 = plt.plot(t_continuous, g_t, label='Continuous Generation Interval PDF', color='blue')
    
    # Daily transmission probability per contact
    bar = plt.bar(np.arange(max_days), daily_prob_per_contact, alpha=0.6, color='orange', edgecolor='k',
            label=f'Daily Transmission Probability per contact')
    
    # Daily expected secondary infections
    line2 = plt.plot(np.arange(max_days), daily_expected_secondary, 'g-o', label='Expected Secondary Infections per day')
    # These handles don't appear in the plot but will be in the legend
    dummy1 = plt.plot([0], [0], color='none', label=f'No of contacts per day = {contacts_per_day}')
    dummy2 = plt.plot([0], [0], color='none', label=f'R0 = {R0}')

    plt.xlabel('Days since infection')
    plt.ylabel('Probability / Expected Number')
    plt.title('Generation Interval to Daily Transmission Probability')
    plt.legend(handles=[line1[0], bar, line2[0], dummy1[0], dummy2[0]], loc='upper right')

    plt.grid(alpha=0.3)
    plt.show()
    
                
               