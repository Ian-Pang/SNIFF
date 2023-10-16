import numpy as np
import random
import time
import torch

################################################ SCORE FUNCTIONS ##########################################################

def likelihood0(inx, mu, sigma):
    """
    Computation of true likelihood given the observed value, and the mean and uncertainty of the experimental value.

    Args:
        inx: Observed value
        mu: Mean of experimental value
        sigma: Uncertainty of the experimental value
    """
    return np.exp(-((inx-mu)**2)/(2*sigma**2))


def likelihood_u(inx, unc, mu, sigma, num):
    """
    Computation of true likelihood given the observed value, and the mean and uncertainty of the experimental value.

    Args:
        inx: Observed value
        unc: Uncertainty of predicted value
        mu: Mean of experimental value
        sigma: Uncertainty of the experimental value
        num: Iteration number (used to reweight the score)
    """
    full_unc = 100*np.exp(-10*num/29)*unc+sigma**2 # modify line as appropriate
    return np.exp(-((inx-mu)**2)/(2*full_unc))

################################################ RECOMMENDATION FUNCTIONS ##########################################################

def recommend_basic(newdat, model, obs_calc, mu, sigma, scale):
    n = 0
    num_good_2sigma = 0
    num_good_1sigma = 0
    start_time = time.time()
    while n < 100:
        a = random.random()*scale
        b = random.random()*scale
        xy= [a,b]
        for i in range(10):
            model.eval()
            outs = model(torch.tensor(xy).float()/scale).detach().numpy()
            if i == 0: full_output = outs
            else: full_output = np.append(full_output,outs,axis=-1)
        mean = full_output.mean()
        var = full_output.var()
        if  random.random() < likelihood0(mean,var, mu, sigma):
            newdat = np.row_stack((newdat,[a,b,obs_calc(a,b)]))
            if obs_calc(a,b) <= 120 and obs_calc(a,b) >= 80: num_good_2sigma += 1
            if obs_calc(a,b) <= 110 and obs_calc(a,b) >= 90: num_good_1sigma += 1
            n = n + 1
    end_time = time.time()
    total_time = end_time-start_time
    time_string = "Needed {:d} min and {:.1f} s to recommend 100 new points."
    print(time_string.format(int(total_time//60), total_time%60))

    print("Efficency 1 sigma = ", num_good_1sigma/100)
    print("Efficency 2 sigma = ", num_good_2sigma/100)
    
    return newdat

def recommend_uncertainty(newdat, model, obs_calc, mu, sigma, num, scale =1):
    n = 0
    num_good_2sigma = 0
    num_good_1sigma = 0
    start_time = time.time()
    while n < 100:
        a = random.random()*scale
        b = random.random()*scale
        xy= [a,b]
        for i in range(10):
            model.eval()
            outs = model(torch.tensor(xy).float()/scale).detach().numpy()
            if i == 0: full_output = outs
            else: full_output = np.append(full_output,outs,axis=-1)
        mean = full_output.mean()
        var = full_output.var()
        if  random.random() < likelihood_u(mean,var,mu, sigma, num):
            newdat = np.row_stack((newdat,[a,b,obs_calc(a,b)]))
            if obs_calc(a,b) <= 120 and obs_calc(a,b) >= 80: num_good_2sigma += 1
            if obs_calc(a,b) <= 110 and obs_calc(a,b) >= 90: num_good_1sigma += 1
            n = n + 1
    end_time = time.time()
    total_time = end_time-start_time
    time_string = "Needed {:d} min and {:.1f} s to recommend 100 new points."
    print(time_string.format(int(total_time//60), total_time%60))

    print("Efficency 1 sigma = ", num_good_1sigma/100)
    print("Efficency 2 sigma = ", num_good_2sigma/100)
    
    return newdat