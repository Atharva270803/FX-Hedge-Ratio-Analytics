# ============================================================
# FX RISK ANALYTICS DASHBOARD
# Save as: dashboard.py
# Run: streamlit run dashboard.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm

st.set_page_config(
    page_title="FX Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    font-size: 17px;
    color: #e2e8f0;
}

.stApp {
    background: #000000;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

.main-header {
    font-size: 3.2rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 6px;
    letter-spacing: -1px;
    line-height: 1.1;
}

.sub-header {
    font-size: 1.15rem;
    color: #94a3b8;
    margin-bottom: 1.8rem;
    font-weight: 400;
}

.section-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;
}

.rec-card-success {
    background: #041f11;
    border-left: 5px solid #22c55e;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 1rem 0 1.5rem 0;
}

.rec-card-warning {
    background: #1a1000;
    border-left: 5px solid #f59e0b;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 1rem 0 1.5rem 0;
}

.rec-card-danger {
    background: #1a0000;
    border-left: 5px solid #ef4444;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin: 1rem 0 1.5rem 0;
}

.rec-tag {
    font-size: 0.8rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #22c55e;
    margin-bottom: 10px;
}

.rec-tag-warn { color: #f59e0b; }
.rec-tag-danger { color: #ef4444; }

.rec-action {
    font-size: 1.3rem;
    font-weight: 700;
    color: #f8fafc;
    margin-bottom: 8px;
    line-height: 1.3;
}

.rec-sub {
    font-size: 1rem;
    color: #94a3b8;
    line-height: 1.6;
}

div[data-testid="metric-container"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 20px 22px;
}

div[data-testid="stMetricLabel"] p {
    font-size: 0.9rem !important;
    color: #64748b !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

div[data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #f1f5f9 !important;
}

div[data-testid="stMetricDelta"] {
    font-size: 0.85rem !important;
}

section[data-testid="stSidebar"] {
    background: #000000;
    border-right: 1px solid #1e293b;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div {
    color: #cbd5e1 !important;
    font-size: 0.95rem !important;
}

section[data-testid="stSidebar"] .stRadio label {
    font-size: 1rem !important;
    color: #e2e8f0 !important;
}

.stDataFrame {
    font-size: 0.95rem !important;
}

.stDataFrame th {
    font-size: 0.85rem !important;
    color: #94a3b8 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

.stDataFrame td {
    font-size: 0.95rem !important;
    color: #e2e8f0 !important;
}

hr {
    border-color: #1e293b;
    margin: 1.5rem 0;
}

.footer-text {
    text-align: center;
    color: #1e293b;
    font-size: 0.82rem;
    padding: 0.5rem 0;
}

.sidebar-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 1rem;
}

.sidebar-meta {
    font-size: 0.82rem;
    color: #334155;
    line-height: 1.8;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# DATA LOADING
# ============================================================

DATA_PATH = r'D:\Users\Lenovo\Downloads\main'

@st.cache_data
def load_all_data():
    prices  = pd.read_csv(f'{DATA_PATH}\\prices_clean.csv',
                           parse_dates=['Date'], index_col='Date')
    returns = pd.read_csv(f'{DATA_PATH}\\returns_clean.csv',
                           parse_dates=['Date'], index_col='Date')
    hr_IS   = pd.read_csv(f'{DATA_PATH}\\dcc_hedge_ratios_IS.csv',
                           parse_dates=['Date'], index_col='Date')
    hr_OOS  = pd.read_csv(f'{DATA_PATH}\\dcc_hedge_ratios_OOS.csv',
                           parse_dates=['Date'], index_col='Date')
    vol_fc  = pd.read_csv(f'{DATA_PATH}\\volatility_forecasts.csv')
    ols_r   = pd.read_csv(f'{DATA_PATH}\\ols_results.csv')
    dcc_r   = pd.read_csv(f'{DATA_PATH}\\dcc_results.csv')
    vecm_r  = pd.read_csv(f'{DATA_PATH}\\vecm_results.csv')
    bekk_r  = pd.read_csv(f'{DATA_PATH}\\bekk_results.csv')
    return (prices, returns, hr_IS, hr_OOS,
            vol_fc, ols_r, dcc_r, vecm_r, bekk_r)

(prices, returns, hr_IS, hr_OOS,
 vol_fc, ols_r, dcc_r, vecm_r, bekk_r) = load_all_data()

h_ols_usd        = ols_r['USD_h_star'].values[0]
h_ols_eur        = ols_r['EUR_h_star'].values[0]
HE_ols_usd_OOS   = ols_r['USD_HE_OOS'].values[0]
HE_ols_eur_OOS   = ols_r['EUR_HE_OOS'].values[0]
HE_dcc_usd_OOS   = dcc_r['USD_HE_OOS'].values[0]
HE_dcc_eur_OOS   = dcc_r['EUR_HE_OOS'].values[0]
HE_vecm_usd_OOS  = vecm_r['USD_HE_OOS'].values[0]
HE_vecm_eur_OOS  = vecm_r['EUR_HE_OOS'].values[0]
HE_bekk_usd_OOS  = bekk_r['USD_HE_OOS'].values[0]
HE_bekk_eur_OOS  = bekk_r['EUR_HE_OOS'].values[0]
h_dcc_usd_today  = hr_OOS['hr_usd_OOS'].iloc[-1]
h_dcc_eur_today  = hr_OOS['hr_eur_OOS'].iloc[-1]
spot_usd         = prices['USDINR_Spot'].iloc[-1]
spot_eur         = prices['EURINR_Spot'].iloc[-1]

# ============================================================
# CHART THEME
# ============================================================

CHART = dict(
    plot_bgcolor='#000000',
    paper_bgcolor='#000000',
    font=dict(color='#94a3b8', size=13),
    margin=dict(l=50, r=20, t=30, b=50),
    xaxis=dict(
        gridcolor='#1e293b',
        linecolor='#1e293b',
        tickfont=dict(size=12, color='#64748b'),
        title_font=dict(size=13, color='#94a3b8')),
    yaxis=dict(
        gridcolor='#1e293b',
        linecolor='#1e293b',
        tickfont=dict(size=12, color='#64748b'),
        title_font=dict(size=13, color='#94a3b8'))
)

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    '<div class="sidebar-title">Firm FX Exposure</div>',
    unsafe_allow_html=True)

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard Overview", "Risk Analytics", "Model Analytics"])

currency_pair = st.sidebar.selectbox(
    "Currency Pair", ["USD/INR", "EUR/INR"])

exposure_input = st.sidebar.number_input(
    "Exposure Amount (Foreign Currency)",
    min_value=10_000, max_value=100_000_000,
    value=1_000_000, step=10_000, format="%d")

exposure_type = st.sidebar.selectbox(
    "Exposure Type",
    ["Receivable (Export)", "Payable (Import)"])

hedge_horizon = st.sidebar.slider(
    "Hedge Horizon (Days)", 7, 90, 30)

confidence = st.sidebar.selectbox(
    "Confidence Level", [0.95, 0.99],
    format_func=lambda x: f"{int(x*100)}%")

model_choice = st.sidebar.selectbox(
    "Hedge Ratio Model",
    ["GJR-DCC-GARCH", "OLS", "BEKK-GARCH"])

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div class="sidebar-meta">'
    'Data: Jan 2015 - Dec 2024<br>'
    'Source: NSE - Investing.com - FRED<br>'
    'Models: OLS - VECM - GJR-DCC - BEKK'
    '</div>',
    unsafe_allow_html=True)

# ============================================================
# COMPUTATIONS
# ============================================================

is_usd       = (currency_pair == "USD/INR")
spot_rate    = spot_usd if is_usd else spot_eur
exposure_inr = exposure_input * spot_rate

if model_choice == "GJR-DCC-GARCH":
    h_selected = h_dcc_usd_today if is_usd else h_dcc_eur_today
    he_oos     = HE_dcc_usd_OOS  if is_usd else HE_dcc_eur_OOS
elif model_choice == "OLS":
    h_selected = h_ols_usd if is_usd else h_ols_eur
    he_oos     = HE_ols_usd_OOS if is_usd else HE_ols_eur_OOS
else:
    h_selected = h_ols_usd if is_usd else h_ols_eur
    he_oos     = HE_bekk_usd_OOS if is_usd else HE_bekk_eur_OOS

futures_notional_inr = h_selected * exposure_inr
futures_notional_fc  = h_selected * exposure_input
current_vol = (vol_fc['usd_spot_vol'].iloc[0]
               if is_usd else vol_fc['eur_spot_vol'].iloc[0])
horizon_vol = current_vol * np.sqrt(hedge_horizon)
z_alpha     = norm.ppf(confidence)
VaR         = abs(z_alpha) * horizon_vol * exposure_inr
CVaR        = (norm.pdf(norm.ppf(confidence)) /
               (1-confidence)) * horizon_vol * exposure_inr
VaR_hedged  = VaR  * (1 - max(he_oos, 0))
CVaR_hedged = CVaR * (1 - max(he_oos, 0))
hist_vol    = (returns['r_USDINR_Spot'].std()
               if is_usd else returns['r_EURINR_Spot'].std())

if current_vol > hist_vol * 1.5:
    regime       = "High"
    regime_color = "#ef4444"
elif current_vol > hist_vol:
    regime       = "Elevated"
    regime_color = "#f59e0b"
else:
    regime       = "Normal"
    regime_color = "#22c55e"

direction  = "Sell" if exposure_type == "Receivable (Export)" else "Buy"
instrument = "USD/INR Futures (NSE)" if is_usd else "EUR/INR Futures (NSE)"

# ============================================================
# HEADER - shown on all pages
# ============================================================

st.markdown(
    '<div class="main-header">FX Risk Analytics</div>',
    unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Dynamic Hedge Ratio Estimation - INR Currency Futures - '
    'GJR-DCC-GARCH - BEKK - VECM - OLS'
    '</div>',
    unsafe_allow_html=True)

# ============================================================
# PAGE 1 - DASHBOARD OVERVIEW
# ============================================================

if page == "Dashboard Overview":

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.metric(
            "Hedge Ratio",
            f"{h_selected:.4f}",
            delta=f"{(h_selected - (h_ols_usd if is_usd else h_ols_eur)):+.4f} vs OLS")
    with c2:
        st.metric(
            "Exposure",
            f"₹{exposure_inr/1e7:.2f} Cr",
            delta=f"Spot ₹{spot_rate:.2f}")
    with c3:
        st.metric(
            "Futures Hedge",
            f"₹{futures_notional_inr/1e7:.2f} Cr",
            delta=f"{futures_notional_fc:,.0f} {currency_pair[:3]}")
    with c4:
        st.metric(
            "Hedge Effectiveness",
            f"{he_oos*100:.1f}%",
            delta="Out-of-sample 2024")
    with c5:
        st.metric(
            "Volatility Regime",
            regime,
            delta=f"{current_vol*100:.3f}% daily vol")

    # Recommendation card
    if he_oos > 0.2:
        card_class = "rec-card-success"
        tag_class  = "rec-tag"
        status     = "Hedge Recommended"
    elif he_oos > 0:
        card_class = "rec-card-warning"
        tag_class  = "rec-tag rec-tag-warn"
        status     = "Hedge with Caution"
    else:
        card_class = "rec-card-danger"
        tag_class  = "rec-tag rec-tag-danger"
        status     = "Hedging Ineffective for this Pair"

    st.markdown(f"""
    <div class="{card_class}">
        <div class="{tag_class}">{status}</div>
        <div class="rec-action">
            {direction} {instrument}
            &nbsp;-&nbsp;
            Notional ₹{futures_notional_inr/1e7:.2f} crore
            &nbsp;-&nbsp;
            {futures_notional_fc:,.0f} {currency_pair[:3]}
        </div>
        <div class="rec-sub">
            Hedge ratio &nbsp;<strong style="color:#f1f5f9;">{h_selected:.4f}</strong>
            &nbsp;-&nbsp; Model: <strong style="color:#f1f5f9;">{model_choice}</strong>
            &nbsp;-&nbsp; Variance reduction: <strong style="color:#f1f5f9;">{max(he_oos,0)*100:.1f}%</strong>
            &nbsp;-&nbsp; Horizon: <strong style="color:#f1f5f9;">{hedge_horizon} days</strong>
            &nbsp;-&nbsp; Confidence: <strong style="color:#f1f5f9;">{int(confidence*100)}%</strong>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown(
            '<div class="section-label">Time-varying hedge ratio</div>',
            unsafe_allow_html=True)

        hr_col = 'hr_usd_IS' if is_usd else 'hr_eur_IS'
        h_ols  = h_ols_usd if is_usd else h_ols_eur

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hr_IS.index,
            y=hr_IS[hr_col],
            mode='lines',
            name='GJR-DCC h*_t',
            line=dict(color='#60a5fa', width=1.5),
            opacity=0.9))
        fig.add_hline(
            y=h_ols,
            line_dash="dash",
            line_color="#f87171",
            line_width=1.2,
            annotation_text=f"OLS = {h_ols:.3f}",
            annotation_font_color="#f87171",
            annotation_font_size=12)
        fig.add_vrect(
            x0="2020-03-01", x1="2020-06-30",
            fillcolor="#ef4444", opacity=0.07,
            annotation_text="COVID",
            annotation_font_color="#ef4444",
            annotation_font_size=11)
        fig.update_layout(
            **CHART, height=360,
            xaxis_title="Date",
            yaxis_title="Hedge Ratio h*_t",
            showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown(
            '<div class="section-label">30-day volatility forecast</div>',
            unsafe_allow_html=True)

        vol_col = 'usd_spot_vol' if is_usd else 'eur_spot_vol'
        vols    = vol_fc[vol_col].values * 100
        days    = list(range(1, 31))

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=days + days[::-1],
            y=list(vols * 1.2) + list((vols * 0.8)[::-1]),
            fill='toself',
            fillcolor='rgba(96,165,250,0.07)',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False))
        fig2.add_trace(go.Scatter(
            x=days,
            y=list(vols),
            mode='lines+markers',
            name='Forecast',
            line=dict(color='#60a5fa', width=2.5),
            marker=dict(size=5, color='#60a5fa'),
            showlegend=False))
        fig2.add_vline(
            x=hedge_horizon,
            line_dash="dash",
            line_color="#f87171",
            line_width=1.2,
            annotation_text=f"{hedge_horizon}d",
            annotation_font_color="#f87171",
            annotation_font_size=12)
        fig2.update_layout(
            **CHART, height=360,
            xaxis_title="Days Ahead",
            yaxis_title="Daily Volatility (%)")
        st.plotly_chart(fig2, use_container_width=True)

# ============================================================
# PAGE 2 - RISK ANALYTICS
# ============================================================

elif page == "Risk Analytics":

    st.markdown(
        f'<div class="section-label">'
        f'Risk metrics - {currency_pair} - '
        f'₹{exposure_inr/1e7:.2f} Cr exposure - '
        f'{hedge_horizon}-day - {int(confidence*100)}% confidence'
        f'</div>',
        unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("VaR (Unhedged)",
                  f"₹{VaR/1e5:.2f}L")
    with c2:
        st.metric("VaR (Hedged)",
                  f"₹{VaR_hedged/1e5:.2f}L",
                  delta=f"₹{(VaR-VaR_hedged)/1e5:.2f}L saved",
                  delta_color="inverse")
    with c3:
        st.metric("CVaR (Unhedged)",
                  f"₹{CVaR/1e5:.2f}L")
    with c4:
        st.metric("CVaR (Hedged)",
                  f"₹{CVaR_hedged/1e5:.2f}L",
                  delta=f"₹{(CVaR-CVaR_hedged)/1e5:.2f}L saved",
                  delta_color="inverse")

    st.markdown("---")

    col_bar, col_price = st.columns([1, 1.5])

    with col_bar:
        st.markdown(
            '<div class="section-label">Unhedged vs hedged risk</div>',
            unsafe_allow_html=True)

        fig_bar = go.Figure(go.Bar(
            x=['VaR\nUnhedged', 'VaR\nHedged',
               'CVaR\nUnhedged', 'CVaR\nHedged'],
            y=[VaR/1e5, VaR_hedged/1e5,
               CVaR/1e5, CVaR_hedged/1e5],
            marker_color=['#ef4444', '#22c55e',
                          '#ef4444', '#22c55e'],
            text=[f"₹{v/1e5:.1f}L"
                  for v in [VaR, VaR_hedged, CVaR, CVaR_hedged]],
            textposition='outside',
            textfont=dict(size=13, color='#cbd5e1')))
        fig_bar.update_layout(
            **CHART, height=400,
            yaxis_title="₹ Lakhs",
            showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_price:
        st.markdown(
            f'<div class="section-label">'
            f'FX rate history - {currency_pair}'
            f'</div>',
            unsafe_allow_html=True)

        price_col  = 'USDINR_Spot' if is_usd else 'EURINR_Spot'
        cutoff_date = prices.index.max() - pd.Timedelta(days=756)
        price_data = prices[price_col][prices.index > cutoff_date]
        line_color = '#60a5fa' if is_usd else '#fb923c'

        fig_p = go.Figure()
        fig_p.add_hrect(
            y0=price_data.mean() - price_data.std(),
            y1=price_data.mean() + price_data.std(),
            fillcolor='rgba(96,165,250,0.05)',
            line_width=0,
            annotation_text="±1σ",
            annotation_font_color="#475569",
            annotation_font_size=11)
        fig_p.add_trace(go.Scatter(
            x=price_data.index,
            y=price_data.values,
            mode='lines',
            line=dict(color=line_color, width=1.8),
            showlegend=False))
        fig_p.add_hline(
            y=price_data.mean(),
            line_dash="dot",
            line_color="#334155",
            line_width=1,
            annotation_text=f"Mean {price_data.mean():.2f}",
            annotation_font_color="#64748b",
            annotation_font_size=11)
        fig_p.update_layout(
            **CHART, height=420,
            xaxis_title="Date",
            yaxis_title=f"INR per {currency_pair[:3]}")
        fig_p.update_xaxes(
            rangeslider_visible=True,
            rangeslider_thickness=0.05)
        st.plotly_chart(fig_p, use_container_width=True)

# ============================================================
# PAGE 3 - MODEL ANALYTICS
# ============================================================

elif page == "Model Analytics":

    st.markdown(
        '<div class="section-label">'
        'Model comparison - out-of-sample hedging effectiveness 2024'
        '</div>',
        unsafe_allow_html=True)

    he_df = pd.DataFrame({
        'Model'        : ['OLS', 'VECM',
                          'GJR-DCC-GARCH', 'BEKK-GARCH'],
        'USD/INR OOS'  : [f"{HE_ols_usd_OOS*100:.2f}%",
                          f"{HE_vecm_usd_OOS*100:.2f}%",
                          f"{HE_dcc_usd_OOS*100:.2f}%",
                          f"{HE_bekk_usd_OOS*100:.2f}%"],
        'EUR/INR OOS'  : [f"{HE_ols_eur_OOS*100:.2f}%",
                          f"{HE_vecm_eur_OOS*100:.2f}%",
                          f"{HE_dcc_eur_OOS*100:.2f}%",
                          f"{HE_bekk_eur_OOS*100:.2f}%"],
        'Best For'     : ['USD/INR', 'Neither',
                          'EUR/INR', 'EUR/INR']
    })

    st.dataframe(he_df, hide_index=True, use_container_width=True)

    st.markdown("---")

    col_hr, col_corr = st.columns(2)

    with col_hr:
        st.markdown(
            '<div class="section-label">'
            'OOS hedge ratios - 2024 test period'
            '</div>',
            unsafe_allow_html=True)

        hr_oos_col = 'hr_usd_OOS' if is_usd else 'hr_eur_OOS'
        h_ols      = h_ols_usd if is_usd else h_ols_eur

        fig_oos = go.Figure()
        fig_oos.add_trace(go.Scatter(
            x=hr_OOS.index,
            y=hr_OOS[hr_oos_col],
            mode='lines',
            name='GJR-DCC',
            line=dict(color='#60a5fa', width=2)))
        fig_oos.add_hline(
            y=h_ols,
            line_dash="dash",
            line_color="#f87171",
            line_width=1.2,
            annotation_text=f"OLS = {h_ols:.3f}",
            annotation_font_color="#f87171",
            annotation_font_size=12)
        fig_oos.update_layout(
            **CHART, height=360,
            xaxis_title="Date",
            yaxis_title="Hedge Ratio",
            showlegend=False)
        st.plotly_chart(fig_oos, use_container_width=True)

    with col_corr:
        st.markdown(
            '<div class="section-label">'
            'Dynamic spot-futures correlation'
            '</div>',
            unsafe_allow_html=True)

        rho_col   = 'rho_usd_IS' if is_usd else 'rho_eur_IS'
        rho_color = '#60a5fa' if is_usd else '#fb923c'
        rho_fill  = ('rgba(96,165,250,0.08)' if is_usd
                     else 'rgba(251,146,60,0.08)')

        fig_rho = go.Figure()
        fig_rho.add_trace(go.Scatter(
            x=hr_IS.index,
            y=hr_IS[rho_col],
            mode='lines',
            fill='tozeroy',
            fillcolor=rho_fill,
            line=dict(color=rho_color, width=1.5),
            showlegend=False))
        fig_rho.add_vrect(
            x0="2020-03-01", x1="2020-06-30",
            fillcolor="#ef4444", opacity=0.07,
            annotation_text="COVID",
            annotation_font_color="#ef4444",
            annotation_font_size=11)
        fig_rho.update_layout(
            **CHART, height=360,
            xaxis_title="Date",
            yaxis_title="Correlation ρ_t")
        st.plotly_chart(fig_rho, use_container_width=True)

    st.markdown("---")

    st.markdown(
        '<div class="section-label">'
        'Returns - volatility clustering'
        '</div>',
        unsafe_allow_html=True)

    ret_col  = 'r_USDINR_Spot' if is_usd else 'r_EURINR_Spot'
    cutoff_date = returns.index.max() - pd.Timedelta(days=756)
    ret_data = returns[ret_col][returns.index > cutoff_date]

    fig_ret = go.Figure(go.Bar(
        x=ret_data.index,
        y=ret_data.values * 100,
        marker_color=np.where(
            ret_data.values > 0, '#ef4444', '#22c55e'),
        showlegend=False))
    fig_ret.add_vrect(
        x0="2020-03-01", x1="2020-06-30",
        fillcolor="#ef4444", opacity=0.07,
        annotation_text="COVID",
        annotation_font_color="#ef4444",
        annotation_font_size=11)
    fig_ret.update_layout(
        **CHART, height=300,
        xaxis_title="Date",
        yaxis_title="Log Return (%)")
    st.plotly_chart(fig_ret, use_container_width=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="footer-text">'
    'FX Risk Analytics - MSc Applied Statistics - '
    'Data: NSE India - Investing.com - FRED - Jan 2015 - Dec 2024'
    '</div>',
    unsafe_allow_html=True)