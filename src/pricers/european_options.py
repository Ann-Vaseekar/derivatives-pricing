import numpy as np
from scipy.stats import norm


def european_opt_pricer(
        N: int, 
        S0: float, 
        K: float, 
        r: float, 
        sigma: float, 
        T: float, 
        alpha: float=0.05,
        seed: int=42,
        option_type: str = "call",
        antithetic: bool=False,
        control: bool=False):
    """
    Calculates price of a European option using a MC simulation

    Parameters:
        N (int): Number of iterations
        S0 (float): Underlying price of asset
        K (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility of underlying asset
        T (float): time until maturity (in years)
        alpha (float): significance level
        seed (int): Seed for MC
        option_type (str): call for call option, put for put option
        antithetic (bool): If set true, uses antithetic variates
        control (bool): If set true, uses control variates
    """

    if S0 <= 0:
        raise ValueError('S0 must be positive')
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    if T <= 0:
        raise ValueError('T must be positive')
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put'")

    rng = np.random.default_rng(seed)

    if antithetic:
        N = N if N % 2 == 0 else N + 1
        Z = rng.normal(0, 1, N//2) 
        Z = np.concatenate((Z, -Z), axis=None)
    else:
        Z = rng.normal(0, 1, N)
    
    ST = S0 * np.exp((r-sigma**2/2)*T + sigma * np.sqrt(T)*Z)

    if option_type == "call":
        payoff = np.maximum(ST-K, 0)
    else:
        payoff = np.maximum(K-ST, 0)

    payoff_discounted = np.exp(-r*T) * payoff
    if antithetic:
        payoff_discounted = 0.5 * (payoff_discounted[:N//2] + payoff_discounted[N//2:])
        ST = 0.5 * (ST[:N//2] + ST[N//2:])

    C = np.mean(payoff_discounted)

    if control:
        discounted_ST = np.exp(-r*T) * ST
        beta = np.cov(payoff_discounted, discounted_ST)[0,1] / np.var(discounted_ST)
        C += - beta * (np.mean(discounted_ST) - S0)
        payoff_discounted = payoff_discounted - beta * (discounted_ST - S0)

    std = np.std(payoff_discounted)
    stderr = std / np.sqrt(len(payoff_discounted))
    MoE = norm.ppf(1-alpha/2) * stderr

    return C, (C - MoE,C + MoE)
