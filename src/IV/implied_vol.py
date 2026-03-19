import numpy as np
from src.utils import bs_analytical_solution


def implied_vol(market_price, S0, K, r, T, option_type="call", tol=1e-6, bounds = [1e-6, 10], max_iters=1e6):
    """
    Uses Brent's method to calculate implied volatility from market price

    Parameters:
        market_price (float): Market price
        S0 (float): Underlying price of asset
        K (float): Strike price
        r (float): Risk-free rate
        T (float): time until maturity (in years)
        option_type (str): call for call option, put for put option
        tol (float): tolerance level for brent's method
        bounds (list): list of lower and upper bound to search for volatility
    """

    # checking inputs are valid

    if option_type not in ("call", "put"):
        raise ValueError('option type must be call or put')

    if market_price <= 0:
        raise ValueError("Option price must be positive")

    if option_type == "call":
        intrinsic = max(S0 - K * np.exp(-r * T), 0)
        upper = S0

        if market_price < intrinsic:
            raise ValueError(f"Call price {market_price} below intrinsic value {intrinsic:.3f}")
        if market_price >= upper:
            raise ValueError(f"Call price {market_price} exceeds upper bound {upper:.3f}")
    else:
        intrinsic = max(K * np.exp(-r * T) - S0, 0)
        upper = K * np.exp(-r * T)  # put can never be worth more than PV of strike

        if market_price < intrinsic:
            raise ValueError(f"Put price {market_price} below intrinsic value {intrinsic:.3f}")
        if market_price >= upper:
            raise ValueError(f"Put price {market_price} exceeds upper bound {upper:.3f}")

    # defining function and checking validility of bounds

    f = lambda sigma: bs_analytical_solution(S0, K, r, sigma, T, option_type)[0] - market_price

    fa, fb = f(bounds[0]), f(bounds[1])
    if fa * fb > 0:
        raise ValueError("Bounds do not bracket the root")
    

    # implementing brent's method to find volatility

    a = bounds[0]
    b = bounds[1]
    c = a
    s = b
    d = 0.0
    mflag = True
    fc = fa
    iter = 0
    while np.abs(b - a) > tol and fb != 0 and iter <= max_iters:
        
        if fa != fc and fb != fc:
            s = a * fb * fc / ((fa - fb) * (fa - fc))
            s += b * fa * fc / ((fb - fa) * (fb - fc))
            s += c * fa * fb / ((fc - fa) * (fc - fb))
        else:
            s = b - fb*(b-a)/(fb - fa)
        
        lo = min(a, b)
        hi = max(a, b)

        if ((not ((3*lo+hi)/4 < s < hi)) or (mflag and np.abs(s-b) >= np.abs(b-c)/2) or
            (not mflag and np.abs(s-b) >= np.abs(c-d)/2) or
            (mflag and np.abs(b-a) < tol) or
            (not mflag and np.abs(c-d) < tol)):
            s = (a+b)/2
            mflag = True
        else:
            mflag = False
        fs = f(s)
        d = c
        c = b
        fc = fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs
        
        if np.abs(fa) < np.abs(fb):
            a, b = b, a
            fa, fb = fb, fa
        
        iter += 1
    
    if iter >= max_iters:
        print(f"Warning: Brent's method did not converge within {int(max_iters)} iterations")

    return b if abs(fb) < abs(fa) else a
