# ============================================================
# FX HEDGE RATIO PROJECT — COMPLETE CODE
# MSc Applied Statistics
# ============================================================
# Papers:
# Ederington (1979) — OLS hedge ratio
# Bollerslev (1986) — GARCH
# Johansen (1991)   — Cointegration
# Kroner & Sultan (1993) — VECM-GARCH
# Glosten et al. (1993)  — GJR-GARCH
# Engle & Kroner (1995)  — BEKK-GARCH
# Engle (2002)           — DCC-GARCH
# Pandey (2008)          — Indian market application
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

os.chdir(r'D:\Users\Lenovo\Downloads\main')

# ============================================================
# STEP 1 — DATA LOADING AND PREPROCESSING
# ============================================================

usdinr_spot = pd.read_csv('USD_INR Historical Data.csv')
eurinr_spot = pd.read_csv('EUR_INR Historical Data.csv')
usdinr_fut  = pd.read_csv('USD_INR Futures Historical Data.csv')
eurinr_fut  = pd.read_csv('EUR_INR Futures Historical Data.csv')
vix         = pd.read_csv('VIXCLS.csv')

def clean_investing_file(df, col_name):
    df = df[['Date', 'Price']].copy()
    df.columns = ['Date', col_name]
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    return df

usdinr_spot = clean_investing_file(usdinr_spot, 'USDINR_Spot')
eurinr_spot = clean_investing_file(eurinr_spot, 'EURINR_Spot')
usdinr_fut  = clean_investing_file(usdinr_fut,  'USDINR_Futures')
eurinr_fut  = clean_investing_file(eurinr_fut,  'EURINR_Futures')

vix.columns = ['Date', 'VIX']
vix['Date'] = pd.to_datetime(vix['Date'])
vix = vix.sort_values('Date').reset_index(drop=True)

df = usdinr_spot.merge(eurinr_spot, on='Date', how='inner')
df = df.merge(usdinr_fut,  on='Date', how='inner')
df = df.merge(eurinr_fut,  on='Date', how='inner')
df = df.merge(vix,         on='Date', how='inner')
df.set_index('Date', inplace=True)
df['VIX'] = df['VIX'].ffill()
df.dropna(inplace=True)

returns = np.log(df / df.shift(1)).dropna()
returns.columns = ['r_USDINR_Spot', 'r_EURINR_Spot',
                   'r_USDINR_Futures', 'r_EURINR_Futures', 'r_VIX']

train = returns[returns.index < '2024-01-01']
test  = returns[returns.index >= '2024-01-01']

df.to_csv('prices_clean.csv')
returns.to_csv('returns_clean.csv')
train.to_csv('train_returns.csv')
test.to_csv('test_returns.csv')

print(f"Data ready: {df.shape[0]} price obs, "
      f"{returns.shape[0]} return obs")
print(f"Train: {len(train)}, Test: {len(test)}")

# ============================================================
# STEP 2 — DIAGNOSTIC TESTS
# ============================================================

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.stats.diagnostic import het_arch
from scipy import stats

prices = df.copy()

print("\n" + "="*60)
print("ADF TEST — PRICE LEVELS (expect non-stationary)")
print("="*60)
for col in ['USDINR_Spot','EURINR_Spot',
            'USDINR_Futures','EURINR_Futures']:
    r = adfuller(prices[col], autolag='AIC')
    status = "REJECT H0" if r[1] < 0.05 else "FAIL TO REJECT"
    print(f"{col:20s} p={r[1]:.4f} {status}")

print("\nADF TEST — LOG RETURNS (expect stationary)")
for col in returns.columns[:4]:
    r = adfuller(returns[col], autolag='AIC')
    status = "REJECT H0 ✓" if r[1] < 0.05 else "FAIL TO REJECT"
    print(f"{col:25s} p={r[1]:.4f} {status}")

print("\nJOHANSEN COINTEGRATION TEST")
for pair, c1, c2 in [
    ('USD/INR', 'USDINR_Spot', 'USDINR_Futures'),
    ('EUR/INR', 'EURINR_Spot', 'EURINR_Futures')
]:
    joh = coint_johansen(prices[[c1, c2]].dropna(),
                         det_order=0, k_ar_diff=1)
    ts  = joh.lr1[0]
    cv5 = joh.cvt[0, 1]
    print(f"{pair}: Trace={ts:.2f}, CV5%={cv5:.2f} "
          f"{'COINTEGRATED ✓' if ts > cv5 else 'NOT cointegrated'}")

print("\nARCH-LM TEST")
for col in ['r_USDINR_Spot', 'r_EURINR_Spot',
            'r_USDINR_Futures', 'r_EURINR_Futures']:
    lm, p, _, _ = het_arch(returns[col], nlags=10)
    print(f"{col:25s} LM={lm:.2f} p={p:.6f} "
          f"{'ARCH present ✓' if p < 0.05 else 'No ARCH'}")

print("\nJARQUE-BERA TEST")
for col in ['r_USDINR_Spot', 'r_EURINR_Spot',
            'r_USDINR_Futures', 'r_EURINR_Futures']:
    jb  = stats.jarque_bera(returns[col])
    k   = stats.kurtosis(returns[col], fisher=False)
    print(f"{col:25s} Kurtosis={k:.2f} p={jb.pvalue:.6f} "
          f"{'NON-NORMAL ✓' if jb.pvalue < 0.05 else 'Normal'}")

# ============================================================
# STEP 3 — OLS BASELINE HEDGE RATIO
# Ederington (1979)
# ============================================================

def hedging_effectiveness(spot, futures, hr):
    """Ederington (1979) HE = 1 - Var(hedged)/Var(unhedged)"""
    if np.isscalar(hr):
        hedged = spot - hr * futures
    else:
        n = min(len(spot), len(futures), len(hr))
        hedged = spot[:n] - hr[:n] * futures[:n]
        spot   = spot[:n]
    return 1 - np.var(hedged) / np.var(spot)

print("\n" + "="*60)
print("OLS HEDGE RATIO — Ederington (1979)")
print("="*60)

y_usd    = train['r_USDINR_Spot']
X_usd    = sm.add_constant(train['r_USDINR_Futures'])
ols_usd  = sm.OLS(y_usd, X_usd).fit()
h_ols_usd = ols_usd.params['r_USDINR_Futures']

y_eur    = train['r_EURINR_Spot']
X_eur    = sm.add_constant(train['r_EURINR_Futures'])
ols_eur  = sm.OLS(y_eur, X_eur).fit()
h_ols_eur = ols_eur.params['r_EURINR_Futures']

HE_ols_usd_IS  = hedging_effectiveness(
    train['r_USDINR_Spot'].values,
    train['r_USDINR_Futures'].values, h_ols_usd)
HE_ols_eur_IS  = hedging_effectiveness(
    train['r_EURINR_Spot'].values,
    train['r_EURINR_Futures'].values, h_ols_eur)
HE_ols_usd_OOS = hedging_effectiveness(
    test['r_USDINR_Spot'].values,
    test['r_USDINR_Futures'].values, h_ols_usd)
HE_ols_eur_OOS = hedging_effectiveness(
    test['r_EURINR_Spot'].values,
    test['r_EURINR_Futures'].values, h_ols_eur)

print(f"USD/INR h*={h_ols_usd:.4f} "
      f"HE_IS={HE_ols_usd_IS*100:.2f}% "
      f"HE_OOS={HE_ols_usd_OOS*100:.2f}%")
print(f"EUR/INR h*={h_ols_eur:.4f} "
      f"HE_IS={HE_ols_eur_IS*100:.2f}% "
      f"HE_OOS={HE_ols_eur_OOS*100:.2f}%")

pd.DataFrame([{
    'USD_h_star'  : h_ols_usd,
    'USD_HE_IS'   : HE_ols_usd_IS,
    'USD_HE_OOS'  : HE_ols_usd_OOS,
    'EUR_h_star'  : h_ols_eur,
    'EUR_HE_IS'   : HE_ols_eur_IS,
    'EUR_HE_OOS'  : HE_ols_eur_OOS
}]).to_csv('ols_results.csv', index=False)

# ============================================================
# STEP 4 — VECM HEDGE RATIO
# Kroner & Sultan (1993)
# ============================================================

from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.vector_ar.vecm import VECM

print("\n" + "="*60)
print("VECM HEDGE RATIO — Kroner & Sultan (1993)")
print("="*60)

train_prices = prices[prices.index < '2024-01-01']

def fit_vecm(data, col1, col2, name):
    d    = data[[col1, col2]]
    lag  = max(int(VAR(d).select_order(10).bic), 1)
    vecm = VECM(d, k_ar_diff=lag,
                coint_rank=1,
                deterministic='ci').fit()
    resid   = vecm.resid
    cov     = np.cov(resid.T)
    h_vecm  = cov[0,1] / cov[1,1]
    print(f"{name}: h*={h_vecm:.4f} "
          f"lambda_spot={vecm.alpha[0,0]:.4f} "
          f"lambda_fut={vecm.alpha[1,0]:.4f}")
    return h_vecm, vecm

h_vecm_usd, vecm_usd_fit = fit_vecm(
    train_prices, 'USDINR_Spot', 'USDINR_Futures', 'USD/INR')
h_vecm_eur, vecm_eur_fit = fit_vecm(
    train_prices, 'EURINR_Spot', 'EURINR_Futures', 'EUR/INR')

HE_vecm_usd_IS  = hedging_effectiveness(
    train['r_USDINR_Spot'].values,
    train['r_USDINR_Futures'].values, h_vecm_usd)
HE_vecm_eur_IS  = hedging_effectiveness(
    train['r_EURINR_Spot'].values,
    train['r_EURINR_Futures'].values, h_vecm_eur)
HE_vecm_usd_OOS = hedging_effectiveness(
    test['r_USDINR_Spot'].values,
    test['r_USDINR_Futures'].values, h_vecm_usd)
HE_vecm_eur_OOS = hedging_effectiveness(
    test['r_EURINR_Spot'].values,
    test['r_EURINR_Futures'].values, h_vecm_eur)

print(f"USD/INR HE_IS={HE_vecm_usd_IS*100:.2f}% "
      f"HE_OOS={HE_vecm_usd_OOS*100:.2f}%")
print(f"EUR/INR HE_IS={HE_vecm_eur_IS*100:.2f}% "
      f"HE_OOS={HE_vecm_eur_OOS*100:.2f}%")

pd.DataFrame([{
    'USD_h_star'         : h_vecm_usd,
    'USD_HE_IS'          : HE_vecm_usd_IS,
    'USD_HE_OOS'         : HE_vecm_usd_OOS,
    'EUR_h_star'         : h_vecm_eur,
    'EUR_HE_IS'          : HE_vecm_eur_IS,
    'EUR_HE_OOS'         : HE_vecm_eur_OOS,
    'USD_lambda_spot'    : vecm_usd_fit.alpha[0,0],
    'USD_lambda_futures' : vecm_usd_fit.alpha[1,0],
    'EUR_lambda_spot'    : vecm_eur_fit.alpha[0,0],
    'EUR_lambda_futures' : vecm_eur_fit.alpha[1,0]
}]).to_csv('vecm_results.csv', index=False)

# ============================================================
# STEP 5 — GJR-DCC-GARCH HEDGE RATIO
# Glosten et al. (1993) + Engle (2002)
# ============================================================

from arch import arch_model
from scipy.optimize import minimize

print("\n" + "="*60)
print("GJR-DCC-GARCH — Glosten et al. (1993) + Engle (2002)")
print("="*60)

scale = 100

def fit_gjr_garch(ret_series, name):
    """
    GJR-GARCH(1,1,1) with Student-t innovations.
    Glosten, Jagannathan & Runkle (1993).
    vol='GARCH', o=1 adds the asymmetric gamma term.
    """
    r      = ret_series.values * scale
    model  = arch_model(r, mean='Constant',
                        vol='GARCH', p=1, o=1, q=1,
                        dist='t')
    result = model.fit(disp='off', show_warning=False)

    # Store GJR parameters for OOS recursion
    params = {
        'mu'      : result.params['mu'],
        'omega'   : result.params['omega'],
        'alpha'   : result.params['alpha[1]'],
        'gamma'   : result.params['gamma[1]'],  # asymmetry term
        'beta'    : result.params['beta[1]'],
        'nu'      : result.params['nu']
    }
    print(f"{name}: alpha={params['alpha']:.4f} "
          f"gamma={params['gamma']:.4f} "
          f"beta={params['beta']:.4f} "
          f"alpha+beta={params['alpha']+params['beta']:.4f}")

    cond_vol  = result.conditional_volatility / scale
    std_resid = result.std_resid
    return result, params, cond_vol, std_resid

print("\nUSD/INR:")
g_usd_s, p_usd_s, v_usd_s, z_usd_s = fit_gjr_garch(
    train['r_USDINR_Spot'],    'USD/INR Spot')
g_usd_f, p_usd_f, v_usd_f, z_usd_f = fit_gjr_garch(
    train['r_USDINR_Futures'], 'USD/INR Futures')

print("\nEUR/INR:")
g_eur_s, p_eur_s, v_eur_s, z_eur_s = fit_gjr_garch(
    train['r_EURINR_Spot'],    'EUR/INR Spot')
g_eur_f, p_eur_f, v_eur_f, z_eur_f = fit_gjr_garch(
    train['r_EURINR_Futures'], 'EUR/INR Futures')

def estimate_dcc(z1, z2, name):
    """DCC(1,1) — Engle (2002)"""
    Z     = np.column_stack([z1, z2])
    T     = len(Z)
    Q_bar = np.corrcoef(Z.T)
    print(f"\n{name} unconditional correlation: {Q_bar[0,1]:.4f}")

    def dcc_loglik(params):
        a, b = params
        if a <= 0 or b <= 0 or a + b >= 0.9999:
            return 1e10
        Q = Q_bar.copy()
        ll = 0.0
        for t in range(1, T):
            zt   = Z[t-1].reshape(-1, 1)
            Q    = (1-a-b)*Q_bar + a*(zt @ zt.T) + b*Q
            d    = np.sqrt(np.diag(Q))
            R    = Q / np.outer(d, d)
            sign, logdet = np.linalg.slogdet(R)
            if sign <= 0:
                return 1e10
            zt_t = Z[t].reshape(1, -1)
            quad = (zt_t @ np.linalg.inv(R) @ zt_t.T).item()
            self_= (zt_t @ zt_t.T).item()
            ll  += -0.5 * (logdet + quad - self_)
        return -ll

    best_val, best_res = 1e10, None
    for x0 in [[0.05,0.90],[0.03,0.95],[0.02,0.97]]:
        res = minimize(dcc_loglik, x0=x0,
                       method='L-BFGS-B',
                       bounds=[(1e-6,0.2),(0.7,0.9999)],
                       options={'maxiter':10000,
                                'ftol':1e-12})
        if res.fun < best_val:
            best_val, best_res = res.fun, res

    a, b = best_res.x
    print(f"{name} DCC: a={a:.4f} b={b:.4f} a+b={a+b:.4f}")
    return a, b, Q_bar

a_usd, b_usd, Qbar_usd = estimate_dcc(z_usd_s, z_usd_f, 'USD/INR')
a_eur, b_eur, Qbar_eur = estimate_dcc(z_eur_s, z_eur_f, 'EUR/INR')

def dcc_hedge_ratios_IS(z1, z2, sig1, sig2, a, b, Q_bar):
    """In-sample time-varying hedge ratios"""
    T    = len(z1)
    Z    = np.column_stack([z1, z2])
    Q    = Q_bar.copy()
    hrs  = []
    rhos = []
    for t in range(1, T):
        zt  = Z[t-1].reshape(-1, 1)
        Q   = (1-a-b)*Q_bar + a*(zt @ zt.T) + b*Q
        d   = np.sqrt(np.diag(Q))
        R   = Q / np.outer(d, d)
        rho = R[0, 1]
        h12 = rho * sig1[t] * sig2[t]
        h22 = sig2[t] ** 2
        hrs.append(h12 / h22)
        rhos.append(rho)
    return np.array(hrs), np.array(rhos)

hr_usd_IS, rho_usd_IS = dcc_hedge_ratios_IS(
    z_usd_s, z_usd_f, v_usd_s, v_usd_f, a_usd, b_usd, Qbar_usd)
hr_eur_IS, rho_eur_IS = dcc_hedge_ratios_IS(
    z_eur_s, z_eur_f, v_eur_s, v_eur_f, a_eur, b_eur, Qbar_eur)

def gjr_garch_forecast_oos(params_s, params_f,
                            last_var_s, last_var_f,
                            last_resid_s, last_resid_f,
                            test_r_s, test_r_f,
                            z_train_s, z_train_f,
                            a, b, Q_bar, scale=100):
    """
    CORRECTED OOS forecasting — properly includes GJR
    gamma term in variance recursion.
    GJR variance update:
    sigma^2_t = omega + alpha*eps^2_{t-1}
              + gamma*eps^2_{t-1}*I[eps_{t-1}<0]
              + beta*sigma^2_{t-1}
    """
    T_train = len(z_train_s)
    Z       = np.column_stack([z_train_s, z_train_f])
    Q       = Q_bar.copy()
    for t in range(1, T_train):
        zt = Z[t-1].reshape(-1, 1)
        Q  = (1-a-b)*Q_bar + a*(zt @ zt.T) + b*Q

    n_test  = len(test_r_s)
    hr_oos  = []
    rho_oos = []
    var_s   = last_var_s
    var_f   = last_var_f
    eps_s   = last_resid_s
    eps_f   = last_resid_f

    test_scaled_s = test_r_s * scale
    test_scaled_f = test_r_f * scale

    for t in range(n_test):
        # GJR-GARCH variance forecast — CORRECTED
        # Includes gamma term for negative residuals
        ind_s = 1.0 if eps_s < 0 else 0.0
        ind_f = 1.0 if eps_f < 0 else 0.0

        var_s = (params_s['omega'] +
                 params_s['alpha'] * eps_s**2 +
                 params_s['gamma'] * eps_s**2 * ind_s +
                 params_s['beta']  * var_s)
        var_f = (params_f['omega'] +
                 params_f['alpha'] * eps_f**2 +
                 params_f['gamma'] * eps_f**2 * ind_f +
                 params_f['beta']  * var_f)

        sig_s = np.sqrt(max(var_s, 1e-10)) / scale
        sig_f = np.sqrt(max(var_f, 1e-10)) / scale

        # DCC correlation update
        if t == 0:
            zt_last = Z[-1].reshape(-1, 1)
        else:
            prev_sig_s = np.sqrt(max(prev_var_s, 1e-10))
            prev_sig_f = np.sqrt(max(prev_var_f, 1e-10))
            zs = (test_scaled_s[t-1] -
                  params_s['mu']) / prev_sig_s
            zf = (test_scaled_f[t-1] -
                  params_f['mu']) / prev_sig_f
            zt_last = np.array([[zs], [zf]])

        Q   = (1-a-b)*Q_bar + a*(zt_last @ zt_last.T) + b*Q
        d   = np.sqrt(np.diag(Q))
        R   = Q / np.outer(d, d)
        rho = R[0, 1]

        h12 = rho * sig_s * sig_f
        h22 = sig_f ** 2
        hr_oos.append(h12 / h22)
        rho_oos.append(rho)

        prev_var_s = var_s
        prev_var_f = var_f
        eps_s = (test_scaled_s[t] - params_s['mu'])
        eps_f = (test_scaled_f[t] - params_f['mu'])

    return np.array(hr_oos), np.array(rho_oos)

# Get last variance states and residuals from training
last_var_usd_s  = (v_usd_s[-1] * scale) ** 2
last_var_usd_f  = (v_usd_f[-1] * scale) ** 2
last_resid_usd_s = g_usd_s.resid[-1]
last_resid_usd_f = g_usd_f.resid[-1]

last_var_eur_s  = (v_eur_s[-1] * scale) ** 2
last_var_eur_f  = (v_eur_f[-1] * scale) ** 2
last_resid_eur_s = g_eur_s.resid[-1]
last_resid_eur_f = g_eur_f.resid[-1]

hr_usd_OOS, rho_usd_OOS = gjr_garch_forecast_oos(
    p_usd_s, p_usd_f,
    last_var_usd_s, last_var_usd_f,
    last_resid_usd_s, last_resid_usd_f,
    test['r_USDINR_Spot'].values,
    test['r_USDINR_Futures'].values,
    z_usd_s, z_usd_f,
    a_usd, b_usd, Qbar_usd)

hr_eur_OOS, rho_eur_OOS = gjr_garch_forecast_oos(
    p_eur_s, p_eur_f,
    last_var_eur_s, last_var_eur_f,
    last_resid_eur_s, last_resid_eur_f,
    test['r_EURINR_Spot'].values,
    test['r_EURINR_Futures'].values,
    z_eur_s, z_eur_f,
    a_eur, b_eur, Qbar_eur)

HE_dcc_usd_IS  = hedging_effectiveness(
    train['r_USDINR_Spot'].values[1:],
    train['r_USDINR_Futures'].values[1:], hr_usd_IS)
HE_dcc_eur_IS  = hedging_effectiveness(
    train['r_EURINR_Spot'].values[1:],
    train['r_EURINR_Futures'].values[1:], hr_eur_IS)
HE_dcc_usd_OOS = hedging_effectiveness(
    test['r_USDINR_Spot'].values,
    test['r_USDINR_Futures'].values, hr_usd_OOS)
HE_dcc_eur_OOS = hedging_effectiveness(
    test['r_EURINR_Spot'].values,
    test['r_EURINR_Futures'].values, hr_eur_OOS)

print(f"\nGJR-DCC USD/INR HE_IS={HE_dcc_usd_IS*100:.2f}% "
      f"HE_OOS={HE_dcc_usd_OOS*100:.2f}%")
print(f"GJR-DCC EUR/INR HE_IS={HE_dcc_eur_IS*100:.2f}% "
      f"HE_OOS={HE_dcc_eur_OOS*100:.2f}%")

pd.DataFrame([{
    'USD_a': a_usd, 'USD_b': b_usd,
    'USD_HE_IS': HE_dcc_usd_IS, 'USD_HE_OOS': HE_dcc_usd_OOS,
    'EUR_a': a_eur, 'EUR_b': b_eur,
    'EUR_HE_IS': HE_dcc_eur_IS, 'EUR_HE_OOS': HE_dcc_eur_OOS
}]).to_csv('dcc_results.csv', index=False)

pd.DataFrame({
    'hr_usd_IS': hr_usd_IS,
    'rho_usd_IS': rho_usd_IS,
    'hr_eur_IS': hr_eur_IS,
    'rho_eur_IS': rho_eur_IS
}, index=train.index[1:]).to_csv('dcc_hedge_ratios_IS.csv')

pd.DataFrame({
    'hr_usd_OOS': hr_usd_OOS,
    'rho_usd_OOS': rho_usd_OOS,
    'hr_eur_OOS': hr_eur_OOS,
    'rho_eur_OOS': rho_eur_OOS
}, index=test.index).to_csv('dcc_hedge_ratios_OOS.csv')

# ============================================================
# STEP 6 — BEKK-GARCH HEDGE RATIO
# Engle & Kroner (1995)
# ============================================================

print("\n" + "="*60)
print("BEKK-GARCH — Engle & Kroner (1995)")
print("="*60)

def bekk_filter(params, returns):
    T = len(returns)
    c11,c21,c22 = params[0],params[1],params[2]
    a1, a2      = params[3],params[4]
    b1, b2      = params[5],params[6]
    C  = np.array([[c11, 0.0],[c21, c22]])
    A  = np.diag([a1, a2])
    B  = np.diag([b1, b2])
    H  = np.cov(returns.T)
    CC = C.T @ C
    ll = 0.0
    H_list = []
    for t in range(1, T):
        eps = returns[t-1:t].T
        H   = CC + A @ (eps @ eps.T) @ A + B @ H @ B
        H   = (H + H.T) / 2.0
        if np.any(np.linalg.eigvalsh(H) <= 1e-8):
            return 1e10, []
        logdet = np.log(np.linalg.det(H))
        if not np.isfinite(logdet):
            return 1e10, []
        H_inv = np.linalg.inv(H)
        eps_t = returns[t:t+1].T
        quad  = float(np.squeeze(eps_t.T @ H_inv @ eps_t))
        if not np.isfinite(quad):
            return 1e10, []
        ll += -0.5 * (2*np.log(2*np.pi) + logdet + quad)
        H_list.append(H.copy())
    return -ll, H_list

def fit_bekk(s_series, f_series, name):
    rets = np.column_stack([s_series.values, f_series.values])
    S    = np.cov(rets.T)
    x0   = [np.sqrt(abs(S[0,0]))*0.3, 0.0,
             np.sqrt(abs(S[1,1]))*0.3,
             0.15, 0.15, 0.85, 0.85]
    res  = minimize(lambda p: bekk_filter(p, rets)[0],
                    x0=x0,
                    method='L-BFGS-B',
                    bounds=[(1e-6,None),(None,None),(1e-6,None),
                            (1e-6,0.5),(1e-6,0.5),
                            (0.5,0.9999),(0.5,0.9999)],
                    options={'maxiter':1000,'ftol':1e-9})
    print(f"{name}: converged={res.success} "
          f"loglik={-res.fun:.2f}")
    _, H_list = bekk_filter(res.x, rets)
    return res.x, H_list, rets

params_usd_b, H_usd_b, ret_usd_b = fit_bekk(
    train['r_USDINR_Spot'],
    train['r_USDINR_Futures'], 'USD/INR')
params_eur_b, H_eur_b, ret_eur_b = fit_bekk(
    train['r_EURINR_Spot'],
    train['r_EURINR_Futures'], 'EUR/INR')

def bekk_hrs(H_list):
    return np.array([H[0,1]/H[1,1] for H in H_list])

def bekk_oos(params, last_H, test_rets):
    c11,c21,c22 = params[0],params[1],params[2]
    a1, a2      = params[3],params[4]
    b1, b2      = params[5],params[6]
    C  = np.array([[c11, 0.0],[c21, c22]])
    A  = np.diag([a1, a2])
    B  = np.diag([b1, b2])
    CC = C.T @ C
    H  = last_H.copy()
    hrs = []
    for t in range(len(test_rets)):
        hrs.append(H[0,1]/H[1,1])
        eps = test_rets[t:t+1].T
        H   = CC + A @ (eps @ eps.T) @ A + B @ H @ B
        H   = (H + H.T) / 2.0
    return np.array(hrs)

hr_bekk_usd    = bekk_hrs(H_usd_b)
hr_bekk_eur    = bekk_hrs(H_eur_b)
last_H_usd_b   = H_usd_b[-1]
last_H_eur_b   = H_eur_b[-1]

ret_test_usd_b = np.column_stack([
    test['r_USDINR_Spot'].values,
    test['r_USDINR_Futures'].values])
ret_test_eur_b = np.column_stack([
    test['r_EURINR_Spot'].values,
    test['r_EURINR_Futures'].values])

hr_bekk_usd_oos = bekk_oos(params_usd_b, last_H_usd_b,
                             ret_test_usd_b)
hr_bekk_eur_oos = bekk_oos(params_eur_b, last_H_eur_b,
                             ret_test_eur_b)

HE_bekk_usd_IS  = hedging_effectiveness(
    train['r_USDINR_Spot'].values[1:],
    train['r_USDINR_Futures'].values[1:], hr_bekk_usd)
HE_bekk_eur_IS  = hedging_effectiveness(
    train['r_EURINR_Spot'].values[1:],
    train['r_EURINR_Futures'].values[1:], hr_bekk_eur)
HE_bekk_usd_OOS = hedging_effectiveness(
    test['r_USDINR_Spot'].values,
    test['r_USDINR_Futures'].values, hr_bekk_usd_oos)
HE_bekk_eur_OOS = hedging_effectiveness(
    test['r_EURINR_Spot'].values,
    test['r_EURINR_Futures'].values, hr_bekk_eur_oos)

print(f"BEKK USD/INR HE_IS={HE_bekk_usd_IS*100:.2f}% "
      f"HE_OOS={HE_bekk_usd_OOS*100:.2f}%")
print(f"BEKK EUR/INR HE_IS={HE_bekk_eur_IS*100:.2f}% "
      f"HE_OOS={HE_bekk_eur_OOS*100:.2f}%")

pd.DataFrame([{
    'USD_HE_IS' : HE_bekk_usd_IS,
    'USD_HE_OOS': HE_bekk_usd_OOS,
    'EUR_HE_IS' : HE_bekk_eur_IS,
    'EUR_HE_OOS': HE_bekk_eur_OOS
}]).to_csv('bekk_results.csv', index=False)

# ============================================================
# STEP 7 — FINAL COMPARISON TABLE
# ============================================================

print(f"\n{'='*68}")
print(f"FINAL MODEL COMPARISON TABLE")
print(f"{'='*68}")
print(f"\n{'Model':<16} {'USD HE IS':>10} {'USD HE OOS':>12} "
      f"{'EUR HE IS':>10} {'EUR HE OOS':>12}")
print(f"{'-'*68}")
for name, ui, uo, ei, eo in [
    ('OLS',         HE_ols_usd_IS,  HE_ols_usd_OOS,
                    HE_ols_eur_IS,  HE_ols_eur_OOS),
    ('VECM',        HE_vecm_usd_IS, HE_vecm_usd_OOS,
                    HE_vecm_eur_IS, HE_vecm_eur_OOS),
    ('GJR-DCC',     HE_dcc_usd_IS,  HE_dcc_usd_OOS,
                    HE_dcc_eur_IS,  HE_dcc_eur_OOS),
    ('BEKK-GARCH',  HE_bekk_usd_IS, HE_bekk_usd_OOS,
                    HE_bekk_eur_IS, HE_bekk_eur_OOS)
]:
    print(f"{name:<16} {ui*100:>9.2f}% {uo*100:>11.2f}% "
          f"{ei*100:>9.2f}% {eo*100:>11.2f}%")

# ============================================================
# STEP 8 — GARCH VOLATILITY FORECAST (30-DAY)
# Used in dashboard
# ============================================================

print("\n" + "="*60)
print("GARCH 30-DAY VOLATILITY FORECAST")
print("Bollerslev (1986) multi-step forecast")
print("="*60)

def garch_forecast_30d(garch_params, last_var,
                        last_resid, horizon=30,
                        scale=100):
    """
    Multi-step GARCH variance forecast.
    Bollerslev (1986):
    sigma^2_{t+h} = omega + (alpha+beta)*sigma^2_{t+h-1}
    For GJR: persistence = alpha + gamma/2 + beta
    (expected value of indicator = 0.5 under symmetry)
    """
    om   = garch_params['omega']
    al   = garch_params['alpha']
    gm   = garch_params['gamma']
    be   = garch_params['beta']

    # GJR persistence (Engle & Ng, 1993 correction)
    pers = min(al + 0.5 * gm + be, 0.9999)

    forecasts = []
    var_h     = last_var

    for h in range(1, horizon + 1):
        if h == 1:
            # One-step uses actual last residual
            ind   = 1.0 if last_resid < 0 else 0.0
            var_h = (om + al * last_resid**2 +
                     gm * last_resid**2 * ind +
                     be * last_var)
        else:
            # Multi-step uses persistence
            var_h = om / (1 - pers) + pers**(h-1) * (
                    var_h - om / (1 - pers))
        forecasts.append(np.sqrt(max(var_h, 1e-10)) / scale)

    return np.array(forecasts)

# USD/INR 30-day forecast
forecast_usd_s = garch_forecast_30d(
    p_usd_s, last_var_usd_s, last_resid_usd_s)
forecast_usd_f = garch_forecast_30d(
    p_usd_f, last_var_usd_f, last_resid_usd_f)

# EUR/INR 30-day forecast
forecast_eur_s = garch_forecast_30d(
    p_eur_s, last_var_eur_s, last_resid_eur_s)
forecast_eur_f = garch_forecast_30d(
    p_eur_f, last_var_eur_f, last_resid_eur_f)

print(f"USD/INR spot vol forecast (day 1):  "
      f"{forecast_usd_s[0]*100:.4f}%")
print(f"USD/INR spot vol forecast (day 30): "
      f"{forecast_usd_s[-1]*100:.4f}%")
print(f"EUR/INR spot vol forecast (day 1):  "
      f"{forecast_eur_s[0]*100:.4f}%")
print(f"EUR/INR spot vol forecast (day 30): "
      f"{forecast_eur_s[-1]*100:.4f}%")

# Save forecasts for dashboard
pd.DataFrame({
    'day'           : range(1, 31),
    'usd_spot_vol'  : forecast_usd_s,
    'usd_fut_vol'   : forecast_usd_f,
    'eur_spot_vol'  : forecast_eur_s,
    'eur_fut_vol'   : forecast_eur_f
}).to_csv('volatility_forecasts.csv', index=False)

print("Volatility forecasts saved")

# ============================================================
# STEP 9 — RISK METRICS (VaR, CVaR, CFaR)
# Jorion (2006)
# ============================================================

print("\n" + "="*60)
print("RISK METRICS — Jorion (2006)")
print("VaR, CVaR, CFaR")
print("="*60)

from scipy.stats import norm

def compute_risk_metrics(exposure_inr,
                          daily_vol,
                          confidence=0.95,
                          horizon_days=30):
    """
    Jorion (2006) parametric risk metrics.

    VaR:  maximum loss at confidence level over horizon
    CVaR: expected loss beyond VaR (Expected Shortfall)
    CFaR: corporate treasury version — applies to cashflows
    """
    z_alpha = norm.ppf(1 - confidence)

    # Scale daily vol to horizon
    horizon_vol = daily_vol * np.sqrt(horizon_days)

    # Parametric VaR
    VaR = abs(z_alpha) * horizon_vol * exposure_inr

    # CVaR (Expected Shortfall)
    # CVaR = phi(z_alpha) / (1-confidence) * sigma * N
    CVaR = (norm.pdf(norm.ppf(1-confidence)) /
            (1-confidence)) * horizon_vol * exposure_inr

    # CFaR — same as VaR but framed as cashflow risk
    CFaR = VaR

    return VaR, CVaR, CFaR

# Example: Indian exporter with $1M exposure at 85 INR/USD
exposure_usd = 1_000_000
spot_rate    = df['USDINR_Spot'].iloc[-1]
exposure_inr = exposure_usd * spot_rate

current_vol_usd = forecast_usd_s[0]  # today's GARCH vol
current_vol_eur = forecast_eur_s[0]

VaR_usd, CVaR_usd, CFaR_usd = compute_risk_metrics(
    exposure_inr, current_vol_usd)
VaR_eur, CVaR_eur, CFaR_eur = compute_risk_metrics(
    exposure_inr, current_vol_eur)

print(f"\nExposure: ${exposure_usd:,} = "
      f"₹{exposure_inr:,.0f}")
print(f"\nUSD/INR Risk Metrics (95%, 30-day):")
print(f"  Daily Vol:  {current_vol_usd*100:.4f}%")
print(f"  VaR:        ₹{VaR_usd:,.0f}")
print(f"  CVaR:       ₹{CVaR_usd:,.0f}")
print(f"  CFaR:       ₹{CFaR_usd:,.0f}")

print(f"\nEUR/INR Risk Metrics (95%, 30-day):")
print(f"  Daily Vol:  {current_vol_eur*100:.4f}%")
print(f"  VaR:        ₹{VaR_eur:,.0f}")
print(f"  CVaR:       ₹{CVaR_eur:,.0f}")
print(f"  CFaR:       ₹{CFaR_eur:,.0f}")

# Save risk metrics
pd.DataFrame([{
    'exposure_usd'   : exposure_usd,
    'exposure_inr'   : exposure_inr,
    'spot_rate'      : spot_rate,
    'usd_vol'        : current_vol_usd,
    'eur_vol'        : current_vol_eur,
    'VaR_usd'        : VaR_usd,
    'CVaR_usd'       : CVaR_usd,
    'CFaR_usd'       : CFaR_usd,
    'VaR_eur'        : VaR_eur,
    'CVaR_eur'       : CVaR_eur,
    'CFaR_eur'       : CFaR_eur,
    'h_ols_usd'      : h_ols_usd,
    'h_ols_eur'      : h_ols_eur,
    'h_dcc_usd_today': hr_usd_OOS[-1],
    'h_dcc_eur_today': hr_eur_OOS[-1]
}]).to_csv('risk_metrics.csv', index=False)

print("\nRisk metrics saved")

# ============================================================
# PLOTS — ALL STEPS
# ============================================================

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('FX Hedge Ratio Project — Complete Results',
             fontsize=14, fontweight='bold')

dates_IS = train.index[1:]

# Plot 1 — USD/INR time-varying hedge ratios
axes[0,0].plot(dates_IS, hr_usd_IS,
               color='steelblue', linewidth=0.7,
               label='GJR-DCC', alpha=0.8)
axes[0,0].plot(dates_IS[:len(hr_bekk_usd)],
               hr_bekk_usd,
               color='green', linewidth=0.7,
               label='BEKK', alpha=0.6)
axes[0,0].axhline(y=h_ols_usd, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_usd:.3f}')
axes[0,0].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.15, color='red', label='COVID')
axes[0,0].set_title('USD/INR Time-Varying Hedge Ratios')
axes[0,0].set_ylabel('Hedge Ratio h*_t')
axes[0,0].legend(fontsize=7)

# Plot 2 — EUR/INR time-varying hedge ratios
axes[0,1].plot(dates_IS, hr_eur_IS,
               color='darkorange', linewidth=0.7,
               label='GJR-DCC', alpha=0.8)
axes[0,1].plot(dates_IS[:len(hr_bekk_eur)],
               hr_bekk_eur,
               color='green', linewidth=0.7,
               label='BEKK', alpha=0.6)
axes[0,1].axhline(y=h_ols_eur, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_eur:.3f}')
axes[0,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.15, color='red')
axes[0,1].set_title('EUR/INR Time-Varying Hedge Ratios')
axes[0,1].set_ylabel('Hedge Ratio h*_t')
axes[0,1].legend(fontsize=7)

# Plot 3 — OOS hedge ratios 2024 USD/INR
axes[1,0].plot(test.index, hr_usd_OOS,
               color='steelblue', linewidth=1.2,
               label='GJR-DCC OOS')
axes[1,0].plot(test.index, hr_bekk_usd_oos,
               color='green', linewidth=1.2,
               label='BEKK OOS', alpha=0.8)
axes[1,0].axhline(y=h_ols_usd, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_usd:.3f}')
axes[1,0].set_title('OOS Hedge Ratios 2024 — USD/INR')
axes[1,0].set_ylabel('Hedge Ratio')
axes[1,0].legend(fontsize=7)

# Plot 4 — OOS hedge ratios 2024 EUR/INR
axes[1,1].plot(test.index, hr_eur_OOS,
               color='darkorange', linewidth=1.2,
               label='GJR-DCC OOS')
axes[1,1].plot(test.index, hr_bekk_eur_oos,
               color='green', linewidth=1.2,
               label='BEKK OOS', alpha=0.8)
axes[1,1].axhline(y=h_ols_eur, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_eur:.3f}')
axes[1,1].set_title('OOS Hedge Ratios 2024 — EUR/INR')
axes[1,1].set_ylabel('Hedge Ratio')
axes[1,1].legend(fontsize=7)

# Plot 5 — 30-day volatility forecast
axes[2,0].plot(range(1,31), forecast_usd_s*100,
               color='steelblue', linewidth=2,
               label='USD/INR Spot Vol')
axes[2,0].plot(range(1,31), forecast_eur_s*100,
               color='darkorange', linewidth=2,
               label='EUR/INR Spot Vol')
axes[2,0].set_title('30-Day GARCH Volatility Forecast\nBollerslev (1986)')
axes[2,0].set_xlabel('Days ahead')
axes[2,0].set_ylabel('Daily Volatility (%)')
axes[2,0].legend(fontsize=8)

# Plot 6 — HE comparison bar chart
models  = ['OLS', 'VECM', 'GJR-DCC', 'BEKK']
he_usd  = [HE_ols_usd_OOS, HE_vecm_usd_OOS,
            HE_dcc_usd_OOS, HE_bekk_usd_OOS]
he_eur  = [HE_ols_eur_OOS, HE_vecm_eur_OOS,
            HE_dcc_eur_OOS, HE_bekk_eur_OOS]
x       = np.arange(len(models))
w       = 0.35
axes[2,1].bar(x - w/2, [h*100 for h in he_usd],
              w, label='USD/INR', color='steelblue',
              alpha=0.8)
axes[2,1].bar(x + w/2, [h*100 for h in he_eur],
              w, label='EUR/INR', color='darkorange',
              alpha=0.8)
axes[2,1].axhline(y=0, color='black', linewidth=0.8)
axes[2,1].set_title('OOS Hedging Effectiveness Comparison')
axes[2,1].set_ylabel('HE (%)')
axes[2,1].set_xticks(x)
axes[2,1].set_xticklabels(models)
axes[2,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('complete_results.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Main results plot saved")
print("\nAll steps complete. Ready for dashboard build.")

# ============================================================
# ALL PLOTS — 6 INDIVIDUAL + 1 COMBINED
# ============================================================

# --- Plot 1: Data Overview ---
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('FX Data Overview — Jan 2015 to Dec 2024',
             fontsize=14, fontweight='bold')

axes[0,0].plot(df.index, df['USDINR_Spot'],
               color='steelblue', linewidth=0.8)
axes[0,0].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red', label='COVID')
axes[0,0].set_title('USD/INR Spot Price')
axes[0,0].set_ylabel('INR per USD')
axes[0,0].legend(fontsize=8)

axes[0,1].plot(df.index, df['EURINR_Spot'],
               color='darkorange', linewidth=0.8)
axes[0,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red')
axes[0,1].set_title('EUR/INR Spot Price')
axes[0,1].set_ylabel('INR per EUR')

axes[1,0].plot(returns.index,
               returns['r_USDINR_Spot'],
               color='steelblue', linewidth=0.5, alpha=0.8)
axes[1,0].axhline(y=0, color='black', linewidth=0.5)
axes[1,0].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red')
axes[1,0].set_title('USD/INR Log Returns')
axes[1,0].set_ylabel('Log Return')

axes[1,1].plot(returns.index,
               returns['r_EURINR_Spot'],
               color='darkorange', linewidth=0.5, alpha=0.8)
axes[1,1].axhline(y=0, color='black', linewidth=0.5)
axes[1,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red')
axes[1,1].set_title('EUR/INR Log Returns')
axes[1,1].set_ylabel('Log Return')

axes[2,0].plot(df.index, df['USDINR_Spot'],
               color='steelblue', linewidth=0.8, label='Spot')
axes[2,0].plot(df.index, df['USDINR_Futures'],
               color='red', linewidth=0.8, alpha=0.7,
               label='Futures')
axes[2,0].set_title('USD/INR Spot vs Futures')
axes[2,0].set_ylabel('INR per USD')
axes[2,0].legend(fontsize=8)

axes[2,1].plot(df.index, df['VIX'],
               color='purple', linewidth=0.8)
axes[2,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red', label='COVID')
axes[2,1].set_title('VIX — Global Fear Index')
axes[2,1].set_ylabel('VIX Level')
axes[2,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('data_overview.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Plot 1 saved: data_overview.png")

# --- Plot 2: Diagnostic Tests ---
from scipy import stats as scipy_stats

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Diagnostic Test Visualisations',
             fontsize=14, fontweight='bold')

for col, color, label in [
    ('r_USDINR_Spot', 'steelblue', 'USD/INR'),
    ('r_EURINR_Spot', 'darkorange', 'EUR/INR')
]:
    axes[0,0].hist(returns[col], bins=80,
                   alpha=0.5, color=color,
                   label=label, density=True)
x_line = np.linspace(-0.03, 0.03, 200)
axes[0,0].plot(x_line,
               scipy_stats.norm.pdf(
                   x_line,
                   returns['r_USDINR_Spot'].mean(),
                   returns['r_USDINR_Spot'].std()),
               'b--', linewidth=1.5, label='Normal fit')
axes[0,0].set_title('Return Distributions vs Normal')
axes[0,0].set_xlabel('Log Return')
axes[0,0].legend(fontsize=8)

axes[0,1].plot(returns.index,
               returns['r_USDINR_Spot']**2,
               color='steelblue', linewidth=0.5, alpha=0.8)
axes[0,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red', label='COVID')
axes[0,1].set_title('Squared Returns — Volatility Clustering')
axes[0,1].set_ylabel('Squared Return')
axes[0,1].legend(fontsize=8)

scipy_stats.probplot(returns['r_USDINR_Spot'],
                     dist="norm", plot=axes[1,0])
axes[1,0].set_title('QQ Plot — USD/INR vs Normal')
axes[1,0].get_lines()[1].set_color('red')

spread = df['USDINR_Spot'] - df['USDINR_Futures']
axes[1,1].plot(df.index, spread,
               color='steelblue', linewidth=0.8)
axes[1,1].axhline(y=spread.mean(), color='red',
                   linestyle='--', linewidth=1,
                   label='Mean basis')
axes[1,1].set_title('USD/INR Basis (Spot − Futures)')
axes[1,1].set_ylabel('Basis (INR)')
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('diagnostic_tests.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Plot 2 saved: diagnostic_tests.png")

# --- Plot 3: OLS Results ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('OLS Baseline Hedge Ratio — Ederington (1979)',
             fontsize=14, fontweight='bold')

axes[0,0].scatter(train['r_USDINR_Futures'],
                   train['r_USDINR_Spot'],
                   alpha=0.2, s=5, color='steelblue')
x_line = np.linspace(
    train['r_USDINR_Futures'].min(),
    train['r_USDINR_Futures'].max(), 100)
axes[0,0].plot(x_line,
               ols_usd.params['const'] + h_ols_usd * x_line,
               color='red', linewidth=2,
               label=f'h*={h_ols_usd:.4f}')
axes[0,0].set_title(f'USD/INR Spot vs Futures Returns\n'
                     f'R²={ols_usd.rsquared:.4f}')
axes[0,0].set_xlabel('Futures Return')
axes[0,0].set_ylabel('Spot Return')
axes[0,0].legend()

axes[0,1].scatter(train['r_EURINR_Futures'],
                   train['r_EURINR_Spot'],
                   alpha=0.2, s=5, color='darkorange')
x_line = np.linspace(
    train['r_EURINR_Futures'].min(),
    train['r_EURINR_Futures'].max(), 100)
axes[0,1].plot(x_line,
               ols_eur.params['const'] + h_ols_eur * x_line,
               color='red', linewidth=2,
               label=f'h*={h_ols_eur:.4f}')
axes[0,1].set_title(f'EUR/INR Spot vs Futures Returns\n'
                     f'R²={ols_eur.rsquared:.4f}')
axes[0,1].set_xlabel('Futures Return')
axes[0,1].set_ylabel('Spot Return')
axes[0,1].legend()

# Rolling OLS
window = 126
roll_h_usd, roll_h_eur, roll_dates = [], [], []
for i in range(window, len(train)):
    w = train.iloc[i-window:i]
    y = w['r_USDINR_Spot']
    X = sm.add_constant(w['r_USDINR_Futures'])
    roll_h_usd.append(sm.OLS(y,X).fit().params[
        'r_USDINR_Futures'])
    y = w['r_EURINR_Spot']
    X = sm.add_constant(w['r_EURINR_Futures'])
    roll_h_eur.append(sm.OLS(y,X).fit().params[
        'r_EURINR_Futures'])
    roll_dates.append(train.index[i])

axes[1,0].plot(roll_dates, roll_h_usd,
               color='steelblue', linewidth=0.8)
axes[1,0].axhline(y=h_ols_usd, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'Constant OLS={h_ols_usd:.4f}')
axes[1,0].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red', label='COVID')
axes[1,0].set_title('Rolling OLS Hedge Ratio — USD/INR')
axes[1,0].set_ylabel('Hedge Ratio')
axes[1,0].legend(fontsize=8)

axes[1,1].plot(roll_dates, roll_h_eur,
               color='darkorange', linewidth=0.8)
axes[1,1].axhline(y=h_ols_eur, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'Constant OLS={h_ols_eur:.4f}')
axes[1,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red', label='COVID')
axes[1,1].set_title('Rolling OLS Hedge Ratio — EUR/INR')
axes[1,1].set_ylabel('Hedge Ratio')
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('ols_results.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Plot 3 saved: ols_results.png")

# --- Plot 4: DCC-GARCH Results ---
dates_IS = train.index[1:]

fig, axes = plt.subplots(3, 2, figsize=(16, 14))
fig.suptitle('GJR-DCC-GARCH Time-Varying Hedge Ratios\n'
             'Glosten et al. (1993) + Engle (2002)',
             fontsize=14, fontweight='bold')

axes[0,0].plot(dates_IS, hr_usd_IS,
               color='steelblue', linewidth=0.7)
axes[0,0].axhline(y=h_ols_usd, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_usd:.3f}')
axes[0,0].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red', label='COVID')
axes[0,0].set_title('DCC Hedge Ratio — USD/INR (Train)')
axes[0,0].set_ylabel('Hedge Ratio h*_t')
axes[0,0].legend(fontsize=8)

axes[0,1].plot(dates_IS, hr_eur_IS,
               color='darkorange', linewidth=0.7)
axes[0,1].axhline(y=h_ols_eur, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_eur:.3f}')
axes[0,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red')
axes[0,1].set_title('DCC Hedge Ratio — EUR/INR (Train)')
axes[0,1].set_ylabel('Hedge Ratio h*_t')
axes[0,1].legend(fontsize=8)

axes[1,0].plot(dates_IS, rho_usd_IS,
               color='steelblue', linewidth=0.7)
axes[1,0].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red', label='COVID')
axes[1,0].set_title('DCC Correlation — USD/INR')
axes[1,0].set_ylabel('Correlation ρ_t')
axes[1,0].legend(fontsize=8)

axes[1,1].plot(dates_IS, rho_eur_IS,
               color='darkorange', linewidth=0.7)
axes[1,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red')
axes[1,1].set_title('DCC Correlation — EUR/INR')
axes[1,1].set_ylabel('Correlation ρ_t')

axes[2,0].plot(test.index, hr_usd_OOS,
               color='steelblue', linewidth=1.2,
               label='GJR-DCC OOS')
axes[2,0].axhline(y=h_ols_usd, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_usd:.3f}')
axes[2,0].set_title('OOS Hedge Ratio 2024 — USD/INR')
axes[2,0].set_ylabel('Hedge Ratio')
axes[2,0].legend(fontsize=8)

axes[2,1].plot(test.index, hr_eur_OOS,
               color='darkorange', linewidth=1.2,
               label='GJR-DCC OOS')
axes[2,1].axhline(y=h_ols_eur, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_eur:.3f}')
axes[2,1].set_title('OOS Hedge Ratio 2024 — EUR/INR')
axes[2,1].set_ylabel('Hedge Ratio')
axes[2,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('dcc_results.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Plot 4 saved: dcc_results.png")

# --- Plot 5: BEKK Results ---
dates_bekk = train.index[1:len(hr_bekk_usd)+1]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('BEKK-GARCH Time-Varying Hedge Ratios\n'
             'Engle & Kroner (1995)',
             fontsize=14, fontweight='bold')

axes[0,0].plot(dates_bekk, hr_bekk_usd,
               color='steelblue', linewidth=0.7)
axes[0,0].axhline(y=h_ols_usd, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_usd:.3f}')
axes[0,0].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red', label='COVID')
axes[0,0].set_title('BEKK Hedge Ratio — USD/INR')
axes[0,0].set_ylabel('Hedge Ratio')
axes[0,0].legend(fontsize=8)

axes[0,1].plot(dates_bekk, hr_bekk_eur,
               color='darkorange', linewidth=0.7)
axes[0,1].axhline(y=h_ols_eur, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_eur:.3f}')
axes[0,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red')
axes[0,1].set_title('BEKK Hedge Ratio — EUR/INR')
axes[0,1].set_ylabel('Hedge Ratio')
axes[0,1].legend(fontsize=8)

axes[1,0].plot(test.index, hr_bekk_usd_oos,
               color='steelblue', linewidth=1.2,
               label='BEKK OOS')
axes[1,0].axhline(y=h_ols_usd, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_usd:.3f}')
axes[1,0].set_title('BEKK OOS 2024 — USD/INR')
axes[1,0].set_ylabel('Hedge Ratio')
axes[1,0].legend(fontsize=8)

axes[1,1].plot(test.index, hr_bekk_eur_oos,
               color='darkorange', linewidth=1.2,
               label='BEKK OOS')
axes[1,1].axhline(y=h_ols_eur, color='red',
                   linestyle='--', linewidth=1.5,
                   label=f'OLS={h_ols_eur:.3f}')
axes[1,1].set_title('BEKK OOS 2024 — EUR/INR')
axes[1,1].set_ylabel('Hedge Ratio')
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('bekk_results.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Plot 5 saved: bekk_results.png")

# --- Plot 6: Forecasts and Risk Metrics ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('GARCH Forecasts and Risk Metrics\n'
             'Bollerslev (1986) + Jorion (2006)',
             fontsize=14, fontweight='bold')

days = range(1, 31)
axes[0,0].plot(days, forecast_usd_s * 100,
               color='steelblue', linewidth=2,
               label='USD/INR Spot')
axes[0,0].plot(days, forecast_usd_f * 100,
               color='steelblue', linewidth=1.5,
               linestyle='--', label='USD/INR Futures')
axes[0,0].set_title('30-Day Volatility Forecast — USD/INR')
axes[0,0].set_xlabel('Days Ahead')
axes[0,0].set_ylabel('Daily Volatility (%)')
axes[0,0].legend(fontsize=8)

axes[0,1].plot(days, forecast_eur_s * 100,
               color='darkorange', linewidth=2,
               label='EUR/INR Spot')
axes[0,1].plot(days, forecast_eur_f * 100,
               color='darkorange', linewidth=1.5,
               linestyle='--', label='EUR/INR Futures')
axes[0,1].set_title('30-Day Volatility Forecast — EUR/INR')
axes[0,1].set_xlabel('Days Ahead')
axes[0,1].set_ylabel('Daily Volatility (%)')
axes[0,1].legend(fontsize=8)

# Risk metrics bar chart
metrics_usd = [VaR_usd/1e5, CVaR_usd/1e5, CFaR_usd/1e5]
metrics_eur = [VaR_eur/1e5, CVaR_eur/1e5, CFaR_eur/1e5]
labels      = ['VaR', 'CVaR', 'CFaR']
x           = np.arange(len(labels))
w           = 0.35
axes[1,0].bar(x - w/2, metrics_usd, w,
              label='USD/INR', color='steelblue', alpha=0.8)
axes[1,0].bar(x + w/2, metrics_eur, w,
              label='EUR/INR', color='darkorange', alpha=0.8)
axes[1,0].set_title('Risk Metrics — $1M Exposure\n'
                     '95% Confidence, 30-Day Horizon')
axes[1,0].set_ylabel('₹ Lakhs')
axes[1,0].set_xticks(x)
axes[1,0].set_xticklabels(labels)
axes[1,0].legend(fontsize=8)

# HE comparison
models  = ['OLS', 'VECM', 'GJR-DCC', 'BEKK']
he_usd  = [HE_ols_usd_OOS, HE_vecm_usd_OOS,
            HE_dcc_usd_OOS, HE_bekk_usd_OOS]
he_eur  = [HE_ols_eur_OOS, HE_vecm_eur_OOS,
            HE_dcc_eur_OOS, HE_bekk_eur_OOS]
colors_usd = ['steelblue' if h > 0 else 'lightcoral'
               for h in he_usd]
colors_eur = ['darkorange' if h > 0 else 'lightcoral'
               for h in he_eur]
x = np.arange(len(models))
axes[1,1].bar(x - w/2, [h*100 for h in he_usd],
              w, color=colors_usd, label='USD/INR',
              alpha=0.8)
axes[1,1].bar(x + w/2, [h*100 for h in he_eur],
              w, color=colors_eur, label='EUR/INR',
              alpha=0.8)
axes[1,1].axhline(y=0, color='black', linewidth=1)
axes[1,1].set_title('OOS Hedging Effectiveness — All Models')
axes[1,1].set_ylabel('HE (%)')
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(models)
axes[1,1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('forecasts_risk.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Plot 6 saved: forecasts_risk.png")

# --- Combined Summary Plot ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('FX Hedge Ratio Project — Complete Summary',
             fontsize=14, fontweight='bold')

axes[0,0].plot(df.index, df['USDINR_Spot'],
               color='steelblue', linewidth=0.8)
axes[0,0].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red')
axes[0,0].set_title('USD/INR Price 2015–2024')
axes[0,0].set_ylabel('INR per USD')

axes[0,1].plot(dates_IS, hr_usd_IS,
               color='steelblue', linewidth=0.7,
               alpha=0.8, label='GJR-DCC')
axes[0,1].axhline(y=h_ols_usd, color='red',
                   linestyle='--', label='OLS')
axes[0,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red')
axes[0,1].set_title('USD/INR Time-Varying h*_t')
axes[0,1].legend(fontsize=8)

axes[0,2].plot(days, forecast_usd_s * 100,
               color='steelblue', linewidth=2)
axes[0,2].set_title('USD/INR 30-Day Vol Forecast')
axes[0,2].set_xlabel('Days Ahead')
axes[0,2].set_ylabel('Volatility (%)')

axes[1,0].plot(df.index, df['EURINR_Spot'],
               color='darkorange', linewidth=0.8)
axes[1,0].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red')
axes[1,0].set_title('EUR/INR Price 2015–2024')
axes[1,0].set_ylabel('INR per EUR')

axes[1,1].plot(dates_IS, hr_eur_IS,
               color='darkorange', linewidth=0.7,
               alpha=0.8, label='GJR-DCC')
axes[1,1].axhline(y=h_ols_eur, color='red',
                   linestyle='--', label='OLS')
axes[1,1].axvspan(pd.Timestamp('2020-03-01'),
                   pd.Timestamp('2020-06-30'),
                   alpha=0.2, color='red')
axes[1,1].set_title('EUR/INR Time-Varying h*_t')
axes[1,1].legend(fontsize=8)

axes[1,2].plot(days, forecast_eur_s * 100,
               color='darkorange', linewidth=2)
axes[1,2].set_title('EUR/INR 30-Day Vol Forecast')
axes[1,2].set_xlabel('Days Ahead')
axes[1,2].set_ylabel('Volatility (%)')

plt.tight_layout()
plt.savefig('summary_dashboard.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("Combined summary plot saved: summary_dashboard.png")

# ============================================================
# APPROACH 2 — MARKOV REGIME SWITCHING HEDGE RATIO
# Hamilton (1989)
# Separate hedge ratios for calm and crisis regimes
# ============================================================

import pandas as pd
import numpy as np
from statsmodels.tsa.regime_switching.markov_regression import (
    MarkovRegression)
import warnings
warnings.filterwarnings('ignore')

import os
os.chdir(r'D:\Users\Lenovo\Downloads\main')

train = pd.read_csv('train_returns.csv',
                    parse_dates=['Date'], index_col='Date')
test  = pd.read_csv('test_returns.csv',
                    parse_dates=['Date'], index_col='Date')

print("="*60)
print("MARKOV REGIME SWITCHING HEDGE RATIO")
print("Hamilton (1989)")
print("="*60)

def fit_markov(spot_series, fut_series, name, k_regimes=2):
    """
    Markov Regime Switching regression.
    Hamilton (1989) — allows hedge ratio to switch
    between calm (regime 0) and crisis (regime 1) states.

    Model: r_spot_t = alpha_s + h_s * r_fut_t + eps_t
    where s = {0,1} follows a Markov chain.
    """
    print(f"\n--- {name} ---")

    # MarkovRegression: dependent = spot, exog = futures
    # k_regimes=2: calm and crisis
    # switching_variance=True: different volatility per regime
    model = MarkovRegression(
        endog=spot_series.values,
        k_regimes=k_regimes,
        exog=fut_series.values.reshape(-1, 1),
        switching_variance=True
    )

    result = model.fit(disp=False, maxiter=1000)

    print(f"Log-likelihood: {result.llf:.4f}")
    print(f"AIC: {result.aic:.4f}")

    # Regime-specific hedge ratios
    # In statsmodels MarkovRegression, parameters are ordered
    # [regime0_const, regime1_const, ..., shared_exog]
    # OR switching_exog=True for regime-specific slopes
    print(f"\nRegime parameters:")
    print(result.summary())

    # Smoothed regime probabilities
    # P(regime=1|all data) — probability of being in crisis
    smoothed_probs = result.smoothed_marginal_probabilities

    print(f"\nAverage probability of crisis regime: "
          f"{smoothed_probs[:,1].mean():.4f}")

    # Transition matrix
    print(f"\nTransition probabilities:")
    p00 = result.regime_transition[0,0,0]
    p10 = result.regime_transition[0,1,0]
    p01 = 1 - p00
    p11 = 1 - p10
    print(f"P(calm->calm):    {p00:.4f}")
    (f"P(calm->crisis):  {p01:.4f}")
    print(f"P(crisis->calm):  {p10:.4f}")
    print(f"P(crisis->crisis):{p11:.4f}")

    return result, smoothed_probs

# Fit for USD/INR
ms_usd, probs_usd = fit_markov(
    train['r_USDINR_Spot'],
    train['r_USDINR_Futures'],
    'USD/INR')

# Fit for EUR/INR
ms_eur, probs_eur = fit_markov(
    train['r_EURINR_Spot'],
    train['r_EURINR_Futures'],
    'EUR/INR')

# Plot regime probabilities
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
fig.suptitle('Markov Regime Switching — Crisis Probabilities\n'
             'Hamilton (1989)', fontsize=13, fontweight='bold')

axes[0].fill_between(train.index,
                      probs_usd[:, 1],
                      alpha=0.6, color='steelblue',
                      label='P(Crisis regime)')
axes[0].axvspan(pd.Timestamp('2020-03-01'),
                 pd.Timestamp('2020-06-30'),
                 alpha=0.3, color='red', label='COVID')
axes[0].set_title('USD/INR — Crisis Regime Probability')
axes[0].set_ylabel('Probability')
axes[0].legend(fontsize=8)
axes[0].set_ylim(0, 1)

axes[1].fill_between(train.index,
                      probs_eur[:, 1],
                      alpha=0.6, color='darkorange',
                      label='P(Crisis regime)')
axes[1].axvspan(pd.Timestamp('2020-03-01'),
                 pd.Timestamp('2020-06-30'),
                 alpha=0.3, color='red', label='COVID')
axes[1].set_title('EUR/INR — Crisis Regime Probability')
axes[1].set_ylabel('Probability')
axes[1].legend(fontsize=8)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig('markov_regime_probs.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("\nMarkov regime plot saved")
print("\nNote: If regime-specific hedge ratios differ")
print("significantly, this approach has practical value")
print("for crisis-period hedging decisions.")