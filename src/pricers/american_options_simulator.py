import numpy as np
from scipy.stats import norm
from src.pricers.american_options import payoff


def american_opt_pricer_MC(
    N: int, 
    M: int,
    S0: float, 
    K: float, 
    r: float, 
    sigma: float, 
    T: float, 
    alpha: float=0.05,
    seed: int=42,
    option_type: str = "call",
    ):
    """
    Longstaff-Schwartz Monte Carlo pricer for American options.
    """

    # parameter validation
    if S0 <= 0:
        raise ValueError('S0 must be positive')
    if sigma <= 0:
        raise ValueError('sigma must be positive')
    if T <= 0:
        raise ValueError('T must be positive')
    if option_type not in ("call", "put"):
        raise ValueError(f"option_type must be 'call' or 'put'")
    

    rng = np.random.default_rng(seed)

    Z = rng.normal(0, 1, (N, M))
    dt = T / M
    discount = np.exp(-r * dt)

    # Simulate paths
    S = np.zeros((N, M+1))
    S[:, 0] = S0

    for t in range(1, M+1):
        S[:, t] = S[:, t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1])

    # Terminal payoff
    exercise_val = payoff(S[:, M], K, option_type)

    # LSMC step
    for t in range(M-1, -1, -1):
        intrinsic = payoff(S[:, t], K, option_type)
        discounted = discount * exercise_val
        itm = intrinsic > 0
        if itm.sum() < 10:
            exercise_val = discounted
            continue

        X = S[itm, t] / S0
        Y = discounted[itm]
        L0 = np.exp(-X/2)
        L1 = np.exp(-X/2) * (1 - X)
        L2 = np.exp(-X/2) * (1 - 2*X + X**2/2)
        A = np.column_stack([L0, L1, L2])
        beta = np.linalg.lstsq(A, Y, rcond=None)[0]
        continuation_est = A @ beta

        exercise_now = intrinsic[itm] > continuation_est
        exercise_val[itm] = np.where(
            exercise_now,
            intrinsic[itm],
            discounted[itm]
        )
        exercise_val[~itm] = discounted[~itm]

    price = np.mean(exercise_val)
    stderr = np.std(exercise_val, ddof=1) / np.sqrt(N)
    MoE = norm.ppf(1 - alpha/2) * stderr

    return price, (price - MoE, price + MoE)

        
    
