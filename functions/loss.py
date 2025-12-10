import numpy as np
from scipy.stats import pareto, lognorm

def num_poisson_vect(mean, n_sim):
    # returns a vect of size n_sim of random variables w/ Poisson distr of mean
    return np.random.poisson(mean, size=n_sim)

# the following func gives you a single array of severities sampled from pareto
def pareto_sev(scale, b_exp, N):

    # also returns vect of size N (sampled from above func)
    # pareto distr rand variabs, with typ value scale and tail exp b_exp
    return pareto.rvs(b_exp, scale=scale, size=N)

# the following samples from lognormal
def lognorm_sev(shape, scale, N):

    # also returns vect of size N (sampled from above func)
    # pareto distr rand variabs, with typ value scale and tail exp b_exp
    return lognorm.rvs(shape, scale=scale, size=N)

# the following computes the sum directly from the first function (samples the max amount needed, then does masked sum)
# in this way, there is no explicit python looping, fast! But, samples a few times in excess
def aggregate_pareto_losses(scale, b_exp, N_vec):
    # maximum number of claims in any simulation year
    maxN = N_vec.max()

    # sample a full rectangular array of Pareto samples
    big = pareto.rvs(b_exp, scale=scale, size=(len(N_vec), maxN))

    # create a mask so row i only keeps its first N_vec[i] entries
    mask = np.arange(maxN) < N_vec[:, None]

    # sum only the valid entries
    return (big * mask).sum(axis=1)

#this function is similar to the above but calculates the vector of frequencies inside
# simpler to use
def full_aggregate_pareto_losses(b_exp, mean_freq, n_sim, scale=1.0):
    # maximum number of claims in any simulation year
    # scale is set to 1.0 by default

    N_vec = num_poisson_vect(mean_freq, n_sim)

    maxN = N_vec.max()

    # sample a full rectangular array of Pareto samples
    big = pareto.rvs(b_exp, scale=scale, size=(len(N_vec), maxN))

    # create a mask so row i only keeps its first N_vec[i] entries
    mask = np.arange(maxN) < N_vec[:, None]

    # sum only the valid entries
    return (big * mask).sum(axis=1)

# same as above, but for lognormal
def full_aggregate_lognorm_losses(shape, mean_freq, n_sim, scale=1.0):
    # maximum number of claims in any simulation year
    # scale is set to 1.0 by default

    N_vec = num_poisson_vect(mean_freq, n_sim)

    maxN = N_vec.max()

    # sample a full rectangular array of Pareto samples
    big = lognorm.rvs(shape, scale=scale, size=(len(N_vec), maxN))

    # create a mask so row i only keeps its first N_vec[i] entries
    mask = np.arange(maxN) < N_vec[:, None]

    # sum only the valid entries
    return (big * mask).sum(axis=1)
