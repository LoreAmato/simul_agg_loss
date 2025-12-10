import numpy as np

# analysis of mean excess loss over a threshold
def mean_excess_loss(losses, u):
    # losses: array of simulated losses
    # u: threshold
    exceedances = losses[losses > u]
    if len(exceedances) == 0:
        return 0.0
    else:
        return np.mean(exceedances - u)
    
# hill estimator for tail index
def hill_estimator(losses, k):
    # losses: array of simulated losses
    # k: number of top order statistics to use
    sorted_losses = np.sort(losses)
    n = len(sorted_losses)
    if k >= n:
        raise ValueError("k must be less than the number of losses")
    x_k = sorted_losses[-k-1]
    sum_log_ratios = np.sum(np.log(sorted_losses[-k:]) - np.log(x_k))
    return k / sum_log_ratios

def pareto_tail_mle(X, p_min=0.99):
    x = np.asarray(X)
    u = np.quantile(x, p_min)
    tail = X[X > u]
    m = len(tail)
    
    return m / np.sum(np.log(tail / u))

def pareto_tail_survival(X,p_min=0.99):
    xs = np.sort(np.asarray(X)) # must sort the losses!
    n = len(xs)
    surv = 1.0 - np.arange(n) / n   # survival at xs
    # choose tail (e.g. top 1% as default)
    u = np.quantile(xs, p_min)
    mask = xs >= u
    logx = np.log(xs[mask])
    logsurv = np.log(surv[mask])
    slope, intercept = np.polyfit(logx, logsurv, 1)
    alpha_surv = -slope

    return alpha_surv

# VaR calculator (just the p quantile)
def calc_var(S, p):
    
    S = np.asarray(S)
    return np.quantile(S, p)

# TVaR calculator, i.e. average in the tail part of the distribution, above the p-percentile
def calc_tvar(S, p):
    
    S = np.asarray(S)

    var = calc_var(S, p)

    tail = S[S > var]
    
    return tail.mean()

# following is the resampling function that allows to compute the confidence interval
# (and possibly other properties of the distribution) of any computed statistic (such as VaR and TVaR)
# The method is also known as bootstrapping
def boot_stat(S, stat_func, p, reps=2000, alpha=0.05, seed=None):
    """
    CI for a statistic computed by stat_func(S, p).
    - S: 1D array
    - stat_func: function(S, p) -> scalar statistic
    - p: parameter passed to stat_func (e.g. quantile level)
    - reps: bootstrap reps
    - alpha: significance level for two-sided CI (default 0.05 -> 95% CI)
    Returns: (stat, lower_CI, upper_CI, bootstrap_samples_array)
    """
    rng = np.random.default_rng(seed)
    S = np.asarray(S)
    n = S.shape[0]
    stat = stat_func(S, p)
    boots = np.empty(reps)
    for b in range(reps): 
        idx = rng.integers(0, n, n) # we resample the same size, with substitution! Alternatively, we could sample just a subset
        Sb = S[idx]
        boots[b] = stat_func(Sb, p=p)
    lower = np.percentile(boots, 100 * (alpha/2))
    upper = np.percentile(boots, 100 * (1 - alpha/2))
    return stat, lower, upper, boots 

