import numpy as np
from scipy.linalg import solve_banded


def payoff(
    S: float,
    K: float,
    option_type: str,
):
    """
    Calculates payoff

    Parameters:
        S (float): Current spot price of asset
        K (float): Strike price
        option_type (str): call for call option, put for put option
    """

    if option_type == 'call':
        return np.maximum(S - K, 0)
    elif option_type == 'put':
        return np.maximum(K - S, 0)


def american_opt_pricer_binomial(
    N: int,
    S0: float,
    K: float,
    sigma: float,
    r: float,
    T: float,
    option_type: str,
):
    """
    Calculates price of an American option using a Binomial model

    Parameters:
        N (int): Tree depth
        S0 (float): Current spot price of the underlying asset
        K (float): Strike price
        sigma (float): Volatility of underlying asset
        r (float): Risk-free rate
        T (float): time until maturity (in years)
        option_type (str): call for call option, put for put option
    """

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    
    dt = T / N
    
    d = np.exp(-sigma * np.sqrt(dt))
    u = np.exp(sigma * np.sqrt(dt))
    discount_term = np.exp(-r*dt)

    q = (np.exp(r*dt) - d)/(u-d)

    curr_V = np.arange(N + 1)
    curr_V = payoff(S0 * u ** curr_V * d ** (N - curr_V), K, option_type)

    for i in range(N-1, -1, -1):

        exercise_val = np.arange(i+1)
        exercise_val = payoff(S0 * u ** exercise_val * d ** (i-exercise_val), K, option_type)
        cont_val = discount_term * (q * curr_V[1:] + (1-q) * curr_V[:-1])

        curr_V = np.maximum(exercise_val, cont_val)
    
    return curr_V[0]


def greeks(
    N: int,
    S0: float,
    K: float,
    sigma: float,
    r: float,
    T: float,
    option_type: str,
    dS: float=1,
    dT: float=0.01,
    dSigma: float=0.01,
    dr: float=0.0001
):
    """
    Calculates the greeks for an American option using a Binomial model

    Parameters:
        N (int): Tree depth
        S0 (float): Current spot price of the underlying asset
        K (float): Strike price
        sigma (float): Volatility of underlying asset
        r (float): Risk-free rate
        T (float): time until maturity (in years)
        option_type (str): call for call option, put for put option
        dS (float): abs movement in S for finite diff calc
        dT (float): abs movement in T for finite diff calc
        dSigma (float): abs movement in sigma for finite diff calc
        dr (float): abs movement in r for finite diff calc
    """

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")

    V_Sup = american_opt_pricer_binomial(
        N, S0+dS, K, sigma, r, T, option_type,
    )

    V0 = american_opt_pricer_binomial(
        N, S0, K, sigma, r, T, option_type,
    )

    V_Sdown = american_opt_pricer_binomial(
        N, S0-dS, K, sigma, r, T, option_type,
    )

    V_Tdown = american_opt_pricer_binomial(
        N, S0, K, sigma, r, T-dT, option_type,
    )

    V_sigup = american_opt_pricer_binomial(
        N, S0, K, sigma+dSigma, r, T, option_type,
    )

    V_sigdown = american_opt_pricer_binomial(
        N, S0, K, sigma-dSigma, r, T, option_type,
    )

    V_rup = american_opt_pricer_binomial(
        N, S0, K, sigma, r+dr, T, option_type,
    )

    V_rdown = american_opt_pricer_binomial(
        N, S0, K, sigma, r-dr, T, option_type,
    )

    delta = (V_Sup - V_Sdown) / (2 * dS)

    gamma = (V_Sup - 2*V0 + V_Sdown) / dS**2

    theta = - (V_Tdown- V0) / (dT * 365)

    vega = (V_sigup - V_sigdown) / (2 * dSigma)

    rho = (V_rup - V_rdown) / (2 * dr)

    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


def american_opt_pricer_CN(
    N: int,
    M: int,
    S0: float,
    K: float,
    sigma: float,
    r: float,
    T: float,
    option_type: str,
):
    """
    Uses Crank-Nicolson (BS finite differences) to approximate price of an American option

    Parameters:
        N (int): Steps in asset price
        M (int): Steps in time
        S0 (float): Current spot price of the underlying asset
        K (float): Strike price
        sigma (float): Volatility of underlying asset
        r (float): Risk-free rate
        T (float): time until maturity (in years)
        option_type (str): call for call option, put for put option
    """

    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put', got '{option_type}'")
    
    dt = T/M

    S_min = max(1e-6, S0 * np.exp(-5 * sigma * np.sqrt(T)))
    S_max = S0 * np.exp(5 * sigma * np.sqrt(T))
    x_min = np.log(S_min)
    x_max = np.log(S_max)
    x_grid = np.linspace(x_min, x_max, N+1)
    dx = (x_max - x_min) / N
    S_grid = np.exp(x_grid)

    V = payoff(S_grid, K, option_type)

    alpha = 0.25 * dt * (sigma**2 / dx**2 - (r - 0.5*sigma**2) / dx)
    beta  = -0.5  * dt * (sigma**2 / dx**2 + r)
    gamma = 0.25  * dt * (sigma**2 / dx**2 + (r - 0.5*sigma**2) / dx)

    A_band = np.zeros((3, N-1))
    A_band[1, :] = 1 - beta
    A_band[0, 1:] = -gamma
    A_band[2, :-1] = -alpha
    B_band = np.zeros((3, N-1))
    B_band[1, :] = 1 + beta
    B_band[0, 1:] = gamma
    B_band[2, :-1] = alpha

    intrinsic = payoff(S_grid, K, option_type)

    for i in range(M, 0, -1):

        t = i * dt

        if option_type == "put":
            V0 = K 
            VN = 0.0
        else:
            V0 = 0.0
            VN = S_max - K * np.exp(-r * (T - t))
        
        V[0] = V0
        V[-1] = VN
        
        rhs = (1 + beta) * V[1:N] + alpha * V[0:N-1] + gamma * V[2:N+1]

        rhs[0]  += alpha  * V0
        rhs[-1] += gamma * VN
        
        V[1:N] = solve_banded((1,1), A_band, rhs)
        V = np.maximum(V, intrinsic)
    
    return  np.interp(S0, S_grid, V)


