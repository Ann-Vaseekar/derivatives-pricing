import numpy as np
from scipy.stats import norm


def bs_analytical_solution(
        S0: float, 
        K: float, 
        r: float, 
        sigma: float, 
        T: float, 
        option_type: str = "call"
):
    """
    Calculates price of a European option using BS analytical solution

    Parameters:
        S0 (float): Underlying price of asset
        K (float): Strike price
        r (float): Risk-free rate
        sigma (float): Volatility of underlying asset
        T (float): time until maturity (in years)
        option_type (str): call for call option, put for put option
    """
    assert option_type in ("call", "put"), 'option_type should be call or put'

    d1 = (np.log(S0/K) + (r+sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    C_BS = S0 * norm.cdf(d1) - K * np.exp(-r*T)*norm.cdf(d2)

    P_BS = C_BS - S0 + K * np.exp(-r*T)

    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
    vega = S0 * norm.pdf(d1) * np.sqrt(T)

    if option_type == "call":
        delta = norm.cdf(d1)
        return C_BS, delta, gamma, vega
    else:
        delta = norm.cdf(d1) - 1
        return P_BS, delta, gamma, vega