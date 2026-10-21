"""
Configuration for Cost-Sensitive Online Window Size Selection experiments.

Parameters derived from:
- Paper: "Cost-Sensitive Online Window Size Selection for Portfolio Management"
- Section 2.1: Expert definition, c=0.001
- Section 4.1: Synthetic experiment parameters
- Section 4.2 / Appendix B.4: DJIA parameters
- Appendix B.5: S&P 500 parameters
"""

# ─── Transaction cost ────────────────────────────────────────────────────────
TRANSACTION_COST_C: float = 0.001  # confirmed: p.2 eq.(1), p.7 synthetic, p.17 DJIA

# ─── Synthetic experiment (exp_1 and exp_4) ──────────────────────────────────
SYNTHETIC_NUM_ASSETS: int = 32          # confirmed: p.7 "32 stocks labeled SYN_1 to SYN_32"
SYNTHETIC_TOTAL_DAYS: int = 2000        # confirmed: p.7 "2,000 business days"
SYNTHETIC_REGIME_1_WINDOW: int = 63     # confirmed: p.7 j_t=63 first 1,000 days
SYNTHETIC_REGIME_2_WINDOW: int = 5      # confirmed: p.7 j_t=5 remaining 1,000 days
SYNTHETIC_START_DATE: str = "2020-01-01"  # confirmed: p.7

# Assumed DGP parameters (not specified by paper; exposed as configurable)
SYNTHETIC_MU: float = 5e-4             # assumed: small positive daily expected return
SYNTHETIC_ALPHA: float = 0.3           # assumed: signal-to-noise ratio (0 < alpha < 1)
SYNTHETIC_SIGMA: float = 0.01          # assumed: daily noise standard deviation
SYNTHETIC_RANDOM_SEED: int = 42        # assumed: for reproducibility

# Candidate window sets
WINDOWS_4: tuple = (5, 10, 21, 63)            # confirmed: p.7 synthetic experiment
WINDOWS_8: tuple = (5, 10, 21, 30, 41, 50, 63, 126)  # confirmed: p.17 DJIA, Appendix B.5

# ─── DJIA experiment (exp_2) ─────────────────────────────────────────────────
DJIA_START_DATE: str = "2021-03-04"    # confirmed: p.8 / p.17
DJIA_END_DATE: str = "2024-02-25"      # confirmed: p.8 / p.17
DJIA_TICKERS: tuple = (
    "WBA", "CRM", "HON", "AMGN", "DOW", "AAPL", "GS", "V", "NKE", "UNH",
    "TRV", "CSCO", "CVX", "VZ", "MSFT", "HD", "INTC", "JNJ", "WMT", "CAT",
    "JPM", "DIS", "BA", "KO", "MCD", "AXP", "IBM", "MRK", "MMM", "PG", "BIL",
)  # confirmed: p.23 Appendix C

# ─── S&P 500 experiment (exp_3) ──────────────────────────────────────────────
SP500_START_DATE: str = "2020-01-02"   # confirmed: Appendix B.5
SP500_END_DATE: str = "2024-08-01"     # note: paper says "2026/08/01" but data only exists until 2024
# SP500 tickers are loaded from data/sp500_tickers.txt (481 components)

# ─── Online aggregation parameters ───────────────────────────────────────────
# Learning rate eta for Hedge: set from minimax optimisation (paper eq.)
# For unknown T, use eta ~ sqrt(8 ln n / T) with estimated T
# If T is known: eta* = sqrt(8 ln(n) / (T * L_max^2))
# Assumed: normalise losses to [0,1] and use eta ~ sqrt(8 ln n / T)
# These are computed at runtime from T and n.

# Mixing parameter alpha for Fixed Share:
# From paper: alpha* computed from switching budget K
# K(T) = floor(T / (log(T) * (log(log(T)))^(1+eps))) as T -> inf
FIXED_SHARE_EPSILON: float = 0.5  # assumed: eps > 0 from paper's asymptotic formula

# Initial portfolio: uniform equal-weight
# w_0 = 1/n uniform confirmed as standard initialisation (assumed from paper context)

# ─── Covariance regularisation ───────────────────────────────────────────────
# Not specified by paper; documented as implementation choice
COV_REGULARIZATION_LAMBDA: float = 1e-4  # small ridge for invertibility

# ─── Account value initialisation ────────────────────────────────────────────
INITIAL_ACCOUNT_VALUE: float = 1.0  # assumed: normalised to 1.0

# ─── Baseline algorithms ─────────────────────────────────────────────────────
# UP: Universal Portfolio (Cover 1991)
# EG: Exponentiated Gradient (Helmbold et al. 1998), eta=0.05
# PAMR: PAMR (Li et al. 2012), eps=0.5, C=500
# CWMR: CWMR-Var (Li et al. 2013), phi=10, eps=0.5
# OLMAR: OLMAR (Li & Hoi 2012), m=5, eps=10
EG_ETA: float = 0.05
PAMR_EPSILON: float = 0.5
PAMR_C: float = 500.0
CWMR_PHI: float = 10.0
CWMR_EPSILON: float = 0.5
OLMAR_WINDOW: int = 5
OLMAR_EPSILON: float = 10.0

# ─── Pre-sample lookback ─────────────────────────────────────────────────────
MAX_WINDOW: int = 126  # largest window for S&P500 / DJIA; 63 for synthetic
