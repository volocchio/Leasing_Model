import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from pathlib import Path
from typing import Dict

from app_sense import render_sensitivity_app
from model_engine import (
    run_model_full,
    compute_derived_params,
    DEFAULT_OPEX,
    METRIC_CARD_CSS,
    RADIO_CSS,
    DOWNLOAD_BUTTON_CSS,
)

st.set_page_config(layout='wide')

_asset_dir = Path(__file__).resolve().parent
_img_a320 = _asset_dir / 'A320Tam.jpg'
_img_logo = _asset_dir / 'logo.png'

col_img_1, col_img_2, _ = st.columns([2, 2, 4])
with col_img_1:
    if _img_a320.exists():
        st.image(str(_img_a320), width=340)
with col_img_2:
    if _img_logo.exists():
        st.image(str(_img_logo), width=500)

st.title('Tamarack Aerospace A320 Financial Model')

st.markdown(RADIO_CSS, unsafe_allow_html=True)
st.markdown(METRIC_CARD_CSS, unsafe_allow_html=True)

def _biz_label(v: str) -> str:
    return 'Leasing' if str(v).startswith('Leasing') else 'Kit Sale'

def _mode_label(v: str) -> str:
    return 'Standalone' if str(v).startswith('Standalone') else 'Sensitivity'

col_biz, col_mode, col_ev = st.columns([2, 2.5, 1.5], gap="medium")
with col_biz:
    model_type = st.radio(
        'Business Model',
        options=['Leasing (Split Savings)', 'Kit Sale (Payback Pricing)'],
        horizontal=True,
        format_func=_biz_label,
    )
with col_mode:
    mode = st.radio(
        'Mode',
        options=['Standalone Model', 'Sensitivity Study (3 Drivers, 1 Output)'],
        horizontal=True,
        format_func=_mode_label,
    )
with col_ev:
    ev_placeholder = st.empty()
    ev_placeholder.metric(label="Enterprise Value ($M)", value="—")

if mode == 'Sensitivity Study (3 Drivers, 1 Output)':
    render_sensitivity_app(baseline_params=None, show_title=False, model_type_override=str(model_type))
    st.stop()

# Simplified Sidebar with key sliders
st.sidebar.header('Fuel')
fuel_saving_pct = st.sidebar.slider('Fuel Savings % per Aircraft', min_value=5.0, max_value=15.0, value=10.0, step=0.5) / 100
block_hours = st.sidebar.slider('Block Hours per Aircraft per Year', min_value=1000, max_value=5000, value=3500, step=200)
base_fuel_burn_gal_per_hour = st.sidebar.slider('Base Fuel Burn (gal/hour)', min_value=600, max_value=1200, value=750, step=50)
base_fuel_price = st.sidebar.slider('Base Fuel Price at First Revenue Year ($/gal)', min_value=1.0, max_value=6.0, value=2.75, step=0.1)
fuel_inflation = st.sidebar.slider('Annual Fuel Inflation (%)', min_value=0.0, max_value=15.0, value=4.5, step=0.5) / 100

st.sidebar.header('Market')
tam_shipsets = st.sidebar.slider('Total Addressable Market (at Project Start)', min_value=1000, max_value=10000, value=7500, step=500)
tam_penetration_pct = st.sidebar.slider('TAM Penetration (%)', min_value=0.0, max_value=100.0, value=100.0, step=1.0) / 100

st.sidebar.header('Commercial')
if model_type == 'Kit Sale (Payback Pricing)':
    target_payback_years = st.sidebar.slider('Target Airline Payback (Years)', min_value=1.0, max_value=5.0, value=2.5, step=0.25)
    fuel_savings_split_to_tamarack = 0.50
else:
    target_payback_years = 2.5
    fuel_savings_split_to_tamarack = st.sidebar.slider('Fuel Savings Split to Tamarack (%)', min_value=0.0, max_value=100.0, value=50.0, step=1.0) / 100

if model_type == 'Kit Sale (Payback Pricing)':
    corsia_split = 0.0
    carbon_price = 0.0
else:
    st.sidebar.header('CORSIA')
    corsia_split = st.sidebar.slider('CORSIA Exposure (Share to Tamarack) (%)', min_value=0.0, max_value=100.0, value=50.0, step=1.0) / 100
    carbon_price = st.sidebar.slider('Carbon Price ($/tCO2)', min_value=0.0, max_value=200.0, value=30.0, step=5.0)

st.sidebar.header('Engine Overhaul')
overhaul_extension_pct = st.sidebar.slider('TBO Extension from Lower Temps (%)', min_value=5.0, max_value=20.0, value=10.0, step=1.0) / 100
shop_visit_cost_m = st.sidebar.slider('Shop Visit Cost per Engine ($M)', min_value=3.0, max_value=12.0, value=6.0, step=0.5)
base_tbo_hours = st.sidebar.slider('Base Time Between Overhauls (Hours)', min_value=10000, max_value=25000, value=18000, step=1000)
overhaul_split = st.sidebar.slider('Overhaul Savings Split to Tamarack (%)', min_value=0.0, max_value=100.0, value=50.0, step=1.0) / 100

st.sidebar.header('Fleet Dynamics')
fleet_retirements_per_month = st.sidebar.slider('Fleet Retirements (Aircraft per Month)', min_value=0, max_value=50, value=0, step=1)
include_forward_fit = st.sidebar.checkbox('Include Forward-Fit Aircraft Entering Market', value=False)
if include_forward_fit:
    forward_fit_per_month = st.sidebar.slider('Forward-Fit Additions (Aircraft per Month)', min_value=0, max_value=50, value=0, step=1)
else:
    forward_fit_per_month = 0

st.sidebar.header('Program')
cert_duration_years = st.sidebar.slider('Certification Duration (Years)', min_value=0.25, max_value=5.0, value=2.0, step=0.25)
cert_duration_quarters = max(1, int(round(float(cert_duration_years) * 4.0)))
inventory_kits_pre_install = st.sidebar.slider('Inventory Kits Before First Install', min_value=50, max_value=200, value=90, step=10)

st.sidebar.header('Financial')
cert_readiness_cost = st.sidebar.slider('Equity ($M)', min_value=100.0, max_value=300.0, value=180.0, step=10.0)
cogs_inflation = st.sidebar.slider('Annual COGS Inflation (%)', min_value=0.0, max_value=15.0, value=4.0, step=0.5) / 100
base_cogs_k = st.sidebar.slider('Base COGS per Kit at First Revenue Year ($000)', min_value=100, max_value=800, value=400, step=10)
base_cogs = float(base_cogs_k) * 1000.0
debt_amount = st.sidebar.slider('Max Debt Available ($M)', min_value=0.0, max_value=500.0, value=float(cert_readiness_cost), step=10.0)
debt_apr = st.sidebar.slider('Debt APR (%)', min_value=0.0, max_value=20.0, value=10.0, step=0.5) / 100
debt_term_years = st.sidebar.slider('Debt Term (Years)', min_value=1, max_value=15, value=7, step=1)
tax_rate = st.sidebar.slider('Income Tax Rate (%)', min_value=0.0, max_value=40.0, value=21.0, step=0.5) / 100
wacc = st.sidebar.slider('WACC (%)', min_value=0.0, max_value=30.0, value=11.5, step=0.5) / 100
terminal_growth = st.sidebar.slider('Terminal Growth Rate (%)', min_value=-2.0, max_value=8.0, value=3.0, step=0.5) / 100

st.sidebar.header('Install Rates')
q1_installs = st.sidebar.slider('Q1 Installs', min_value=0, max_value=200, value=98, step=10)
q2_installs = st.sidebar.slider('Q2 Installs', min_value=0, max_value=200, value=98, step=10)
q3_installs = st.sidebar.slider('Q3 Installs', min_value=0, max_value=200, value=98, step=10)
q4_installs = st.sidebar.slider('Q4 Installs and beyond', min_value=0, max_value=200, value=96, step=10)
year2_installs = st.sidebar.slider('Year 2 Annual Installs', min_value=0, max_value=2000, value=910, step=10)
year3_installs = st.sidebar.slider('Year 3+ Annual Installs', min_value=0, max_value=2000, value=1040, step=10)

# ── Derived parameters ───────────────────────────────────────────────────
derived = compute_derived_params(cert_duration_years)
cert_duration_quarters = derived["cert_duration_quarters"]
revenue_start_year = derived["revenue_start_year"]
revenue_start_quarter = derived["revenue_start_quarter"]
inventory_year = derived["inventory_year"]
inventory_quarter = derived["inventory_quarter"]

opex = dict(DEFAULT_OPEX)
base_cogs = float(base_cogs_k) * 1000.0

cert_spend_by_year: Dict[str, float] = {}
cert_spend_per_quarter = (cert_readiness_cost / cert_duration_quarters) if cert_duration_quarters > 0 else 0.0
for q in range(cert_duration_quarters):
    yr = 2026 + (q // 4)
    cert_spend_by_year[yr] = cert_spend_by_year.get(yr, 0.0) + cert_spend_per_quarter

years = list(range(2026, 2036))

# ── Build params dict & run shared engine ────────────────────────────────
params = {
    "model_type": model_type,
    "fuel_inflation": fuel_inflation,
    "base_fuel_price": base_fuel_price,
    "block_hours": block_hours,
    "base_fuel_burn_gal_per_hour": base_fuel_burn_gal_per_hour,
    "cogs_inflation": cogs_inflation,
    "base_cogs": base_cogs,
    "fuel_saving_pct": fuel_saving_pct,
    "fuel_savings_split_to_tamarack": fuel_savings_split_to_tamarack,
    "target_payback_years": target_payback_years,
    "corsia_split": corsia_split,
    "carbon_price": carbon_price,
    "overhaul_extension_pct": overhaul_extension_pct,
    "shop_visit_cost_m": shop_visit_cost_m,
    "base_tbo_hours": base_tbo_hours,
    "overhaul_split": overhaul_split,
    "cert_readiness_cost": cert_readiness_cost,
    "cert_duration_years": cert_duration_years,
    "cert_duration_quarters": cert_duration_quarters,
    "revenue_start_q_index": derived["revenue_start_q_index"],
    "inventory_purchase_q_index": derived["inventory_purchase_q_index"],
    "revenue_start_year": revenue_start_year,
    "inventory_year": inventory_year,
    "inventory_kits_pre_install": inventory_kits_pre_install,
    "tam_shipsets": tam_shipsets,
    "tam_penetration_pct": tam_penetration_pct,
    "fleet_retirements_per_month": fleet_retirements_per_month,
    "include_forward_fit": include_forward_fit,
    "forward_fit_per_month": forward_fit_per_month,
    "debt_amount": debt_amount,
    "debt_apr": debt_apr,
    "debt_term_years": debt_term_years,
    "tax_rate": tax_rate,
    "wacc": wacc,
    "terminal_growth": terminal_growth,
    "q1_installs": q1_installs,
    "q2_installs": q2_installs,
    "q3_installs": q3_installs,
    "q4_installs": q4_installs,
    "year2_installs": year2_installs,
    "year3_installs": year3_installs,
    "opex": opex,
}

result = run_model_full(params)

df = result["df"]
enterprise_value = result["enterprise_value"]
pv_explicit = result["pv_explicit"]
pv_tv = result["pv_tv"]
tv = result["tv"]
unlevered_fcf = result["unlevered_fcf"]
discount_factor = result["discount_factor"]
pv_fcf = result["pv_fcf"]

# ── Enterprise Value metric ──────────────────────────────────────────────
ev_placeholder.metric(label="Enterprise Value ($M)", value=f"{enterprise_value:,.1f}")

# ── Three-statement derivations ──────────────────────────────────────────
net_income = df['EBITDA ($M)'] - df['Debt Interest ($M)'] - df['Taxes ($M)']
pl_df = df[['Revenue ($M)', 'COGS ($M)', 'Gross Profit ($M)', 'OpEx ($M)', 'EBITDA ($M)', 'Debt Interest ($M)', 'Taxes ($M)']].copy()
pl_df['Net Income ($M)'] = net_income.round(1)

equity_paid_in = df['Equity Contribution ($M)'].cumsum()
retained_earnings = net_income.cumsum()
bs_df = pd.DataFrame({
    'Cash ($M)': df['Cumulative Cash ($M)'],
    'Debt Balance ($M)': df['Debt Balance ($M)'],
    'Equity Paid-In ($M)': equity_paid_in,
    'Retained Earnings ($M)': retained_earnings,
}, index=df.index)
bs_df['Total Assets ($M)'] = bs_df['Cash ($M)']
bs_df['Total Liab + Equity ($M)'] = bs_df['Debt Balance ($M)'] + bs_df['Equity Paid-In ($M)'] + bs_df['Retained Earnings ($M)']

operating_cf = df['EBITDA ($M)'] - df['Taxes ($M)']
investing_cf = -df['CapEx/Inv ($M)']
financing_cf = df['Debt Draw ($M)'] - df['Debt Payment ($M)'] + df['Equity Contribution ($M)']
cash_net_change = operating_cf + investing_cf + financing_cf
cf_df = pd.DataFrame({
    'Operating CF ($M)': operating_cf,
    'Investing CF ($M)': investing_cf,
    'Financing CF ($M)': financing_cf,
    'Net Change in Cash ($M)': cash_net_change,
    'Ending Cash ($M)': df['Cumulative Cash ($M)'],
}, index=df.index)

discount_year0 = int(df.index.min())

dcf_df = pd.DataFrame({
    'Unlevered FCF ($M)': unlevered_fcf,
    'Discount Factor': discount_factor,
    'PV of FCF ($M)': pv_fcf,
}, index=df.index)

dcf_summary_df = pd.DataFrame({
    'PV Explicit FCF ($M)': [round(pv_explicit, 1)],
    'Terminal Value ($M)': [round(0.0 if np.isnan(tv) else tv, 1)],
    'PV Terminal Value ($M)': [round(0.0 if np.isnan(pv_tv) else pv_tv, 1)],
    'Enterprise Value ($M)': [round(enterprise_value, 1)],
})

# ── Display table formatting ─────────────────────────────────────────────
df_display = df.copy()
if model_type != 'Kit Sale (Payback Pricing)' and 'Kit Price ($/kit)' in df_display.columns:
    df_display = df_display.drop(columns=['Kit Price ($/kit)'])
elif model_type == 'Kit Sale (Payback Pricing)' and 'Kit Price ($/kit)' in df_display.columns:
    df_display['Kit Price ($/kit)'] = df_display['Kit Price ($/kit)'].apply(
        lambda v: (round(v / 100000.0) * 100000.0) if pd.notna(v) else np.nan
    )

df_display_view = df_display
if model_type == 'Kit Sale (Payback Pricing)' and 'Kit Price ($/kit)' in df_display.columns:
    _fmt: Dict[str, str] = {}
    for _col in list(df_display.columns):
        if _col == 'Kit Price ($/kit)':
            _fmt[_col] = '{:,.0f}'
        elif '($M)' in _col:
            _fmt[_col] = '{:,.1f}'
        elif '(%)' in _col:
            _fmt[_col] = '{:,.1f}'
        elif _col in ['New Installs', 'Cum Shipsets', 'Fleet Size']:
            _fmt[_col] = '{:,.0f}'
    _align_cols = list(_fmt.keys())
    df_display_view = df_display.style.format(_fmt, na_rep='')
    df_display_view = df_display_view.set_properties(subset=_align_cols, **{'text-align': 'right'})

st.dataframe(df_display_view, use_container_width=True)

# ── Three-Statement Output ───────────────────────────────────────────────
st.header('Three-Statement Output')
st.subheader('P&L')
st.dataframe(pl_df, use_container_width=True)
st.subheader('Balance Sheet')
st.dataframe(bs_df, use_container_width=True)
st.subheader('Statement of Cash Flows')
st.dataframe(cf_df, use_container_width=True)

# ── DCF Analysis ─────────────────────────────────────────────────────────
st.header('DCF Analysis')
st.dataframe(dcf_df, use_container_width=True)

st.subheader('DCF Supporting Information')
st.write(f"Discount base year: {discount_year0}")
st.write(f"WACC: {wacc * 100:.2f}%")
st.write(f"Terminal growth rate: {terminal_growth * 100:.2f}%")
st.write(f"PV of explicit period FCF ($M): {pv_explicit:.1f}")
st.write(f"PV of terminal value ($M): {0.0 if np.isnan(pv_tv) else pv_tv:.1f}")
st.dataframe(dcf_summary_df, use_container_width=True)

# ── Interactive Plotly Charts ────────────────────────────────────────────
st.header('Financial Projection Plots')

plot_years = [str(y) for y in df.index]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(name='Revenue', x=plot_years, y=df['Revenue ($M)'], marker_color='#F97316'))
fig_bar.add_trace(go.Bar(name='Gross Profit', x=plot_years, y=df['Gross Profit ($M)'], marker_color='#10B981'))
fig_bar.add_trace(go.Bar(name='Free Cash Flow', x=plot_years, y=df['Free Cash Flow ($M)'], marker_color='#3B82F6'))
fig_bar.update_layout(
    barmode='group',
    title='Annual Revenue, Gross Profit, and Free Cash Flow',
    yaxis_title='$M',
    xaxis_title='Year',
    template='plotly_white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    height=450,
)
st.plotly_chart(fig_bar, use_container_width=True)

cumulative_cash = df['Free Cash Flow ($M)'].cumsum()
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=plot_years, y=cumulative_cash, mode='lines+markers',
    name='Cumulative Free Cash Flow', line=dict(color='#8B5CF6', width=3),
    marker=dict(size=8),
))
fig_cum.add_hline(y=0, line_dash='dash', line_color='gray', opacity=0.5)
fig_cum.update_layout(
    title='Cumulative Cash (Cumulative Free Cash Flow)',
    yaxis_title='$M',
    xaxis_title='Year',
    template='plotly_white',
    height=350,
)
st.plotly_chart(fig_cum, use_container_width=True)

# ── Fleet Dynamics Chart ─────────────────────────────────────────────────
st.header('Fleet Dynamics')
fig_fleet = go.Figure()
fig_fleet.add_trace(go.Scatter(
    x=plot_years, y=df['Fleet Size'], mode='lines+markers',
    name='Eligible Fleet', line=dict(color='#6366F1', width=2),
))
fig_fleet.add_trace(go.Scatter(
    x=plot_years, y=df['Cum Shipsets'], mode='lines+markers',
    name='Installed Base', line=dict(color='#10B981', width=2),
))
fig_fleet.add_trace(go.Bar(
    x=plot_years, y=df['New Installs'], name='New Installs',
    marker_color='#F59E0B', opacity=0.6,
))
fig_fleet.update_layout(
    title='Fleet Size, Installed Base & New Installs Over Time',
    yaxis_title='Aircraft',
    xaxis_title='Year',
    template='plotly_white',
    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
    height=400,
)
st.plotly_chart(fig_fleet, use_container_width=True)

# ── Assumptions Appendix ─────────────────────────────────────────────────
corsia_assumption_rows = [] if model_type == 'Kit Sale (Payback Pricing)' else [
    {'Assumption': 'CORSIA Exposure (Share of Ops)', 'Value': f"{corsia_split * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Share of operations subject to CORSIA compliance pricing'},
    {'Assumption': 'Carbon Price', 'Value': f"{carbon_price:.2f}", 'Units': '$/tCO2', 'Type': 'Slider', 'Notes': 'Used to value avoided CORSIA compliance cost'},
]

engine_overhaul_assumption_rows = [
    {'Assumption': 'TBO Extension from Lower Temps', 'Value': f"{overhaul_extension_pct * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Percent extension in TBO due to lower takeoff power/temperatures'},
    {'Assumption': 'Shop Visit Cost per Engine', 'Value': f"{shop_visit_cost_m:.1f}", 'Units': '$M', 'Type': 'Slider', 'Notes': 'Cost of hot section inspection or performance restoration per engine'},
    {'Assumption': 'Base Time Between Overhauls', 'Value': f"{int(base_tbo_hours):,}", 'Units': 'Hours', 'Type': 'Slider', 'Notes': 'Baseline flight hours between engine shop visits'},
    {'Assumption': 'Overhaul Savings Split to Tamarack', 'Value': f"{overhaul_split * 100:.1f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Share of engine overhaul savings paid to Tamarack'},
]

assumptions_rows = [
    {'Assumption': 'Annual Fuel Inflation', 'Value': f"{fuel_inflation * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to base fuel price starting in the first revenue year'},
    {'Assumption': 'Base Fuel Price (First Revenue Year)', 'Value': f"{base_fuel_price:.2f}", 'Units': '$/gal', 'Type': 'Slider', 'Notes': f"Base fuel price used in {revenue_start_year}"},
    {'Assumption': 'Block Hours per Aircraft per Year', 'Value': f"{int(block_hours)}", 'Units': 'Hours', 'Type': 'Slider', 'Notes': 'Used to compute annual fuel spend'},
    {'Assumption': 'Base Fuel Burn', 'Value': f"{int(base_fuel_burn_gal_per_hour)}", 'Units': 'Gal/hour', 'Type': 'Slider', 'Notes': 'Used to compute annual fuel spend'},
    {'Assumption': 'Annual COGS Inflation', 'Value': f"{cogs_inflation * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to base COGS per kit starting in the first revenue year'},
    {'Assumption': 'Base COGS per Kit (First Revenue Year)', 'Value': f"{base_cogs:,.0f}", 'Units': '$/kit', 'Type': 'Slider', 'Notes': f"Input slider is in $000; base COGS per kit used in {revenue_start_year}"},
    {'Assumption': 'Fuel Savings % per Aircraft', 'Value': f"{fuel_saving_pct * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Percent of annual fuel spend saved'},
    ({'Assumption': 'Target Airline Payback', 'Value': f"{target_payback_years:.2f}", 'Units': 'Years', 'Type': 'Slider', 'Notes': 'Kit price set so airline recovers cost via fuel savings over target payback period'} if model_type == 'Kit Sale (Payback Pricing)' else {'Assumption': 'Fuel Savings Split to Tamarack', 'Value': f"{fuel_savings_split_to_tamarack * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Percent of annual fuel savings paid to Tamarack'}),
    *corsia_assumption_rows,
    *engine_overhaul_assumption_rows,
    {'Assumption': 'Certification Duration', 'Value': f"{cert_duration_years:.2f}", 'Units': 'Years', 'Type': 'Slider', 'Notes': f"{cert_duration_quarters} quarters; go-live is {revenue_start_year}Q{revenue_start_quarter}"},
    {'Assumption': 'Equity', 'Value': f"{cert_readiness_cost:.1f}", 'Units': '$M', 'Type': 'Slider', 'Notes': f"Used first to fund certification / inventory outflows prior to {revenue_start_year}Q{revenue_start_quarter}"},
    {'Assumption': 'Max Debt Available', 'Value': f"{debt_amount:.1f}", 'Units': '$M', 'Type': 'Slider', 'Notes': f"Debt facility cap; model draws only what is needed prior to {revenue_start_year}Q{revenue_start_quarter}"},
    {'Assumption': 'Debt APR', 'Value': f"{debt_apr * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to outstanding debt balance'},
    {'Assumption': 'Debt Term', 'Value': f"{debt_term_years}", 'Units': 'Years', 'Type': 'Slider', 'Notes': f"Debt amortizes quarterly beginning in {revenue_start_year}Q{revenue_start_quarter}"},
    {'Assumption': 'Income Tax Rate', 'Value': f"{tax_rate * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Taxes apply only when taxable income is positive'},
    {'Assumption': 'WACC', 'Value': f"{wacc * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Used to discount unlevered free cash flows in DCF'},
    {'Assumption': 'Terminal Growth Rate', 'Value': f"{terminal_growth * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Used for terminal value if WACC > terminal growth'},
    {'Assumption': 'Inventory Kits Before First Install', 'Value': f"{inventory_kits_pre_install}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"Purchased in {inventory_year}Q{inventory_quarter} (1 quarter before go-live; 25% of full build)"},
    {'Assumption': 'Total Addressable Market', 'Value': f"{tam_shipsets}", 'Units': 'Aircraft', 'Type': 'Slider', 'Notes': 'Starting eligible aftermarket fleet size (used as the base TAM)'},
    {'Assumption': 'TAM Penetration', 'Value': f"{tam_penetration_pct * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Caps maximum installable fleet at TAM * penetration'},
    {'Assumption': 'Fleet Retirements', 'Value': f"{fleet_retirements_per_month}", 'Units': 'Aircraft/month', 'Type': 'Slider', 'Notes': 'Reduces the eligible fleet over time'},
    {'Assumption': 'Forward-Fit Enabled', 'Value': 'Yes' if include_forward_fit else 'No', 'Units': '', 'Type': 'Toggle', 'Notes': 'If enabled, adds new aircraft to the eligible fleet over time'},
    {'Assumption': 'Forward-Fit Additions', 'Value': f"{forward_fit_per_month}", 'Units': 'Aircraft/month', 'Type': 'Slider', 'Notes': 'Adds to the eligible fleet over time when forward-fit is enabled'},
    {'Assumption': 'First-Year Install Rate (Q1)', 'Value': f"{q1_installs}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"First install year ({revenue_start_year}) quarterly installs"},
    {'Assumption': 'First-Year Install Rate (Q2)', 'Value': f"{q2_installs}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"First install year ({revenue_start_year}) quarterly installs"},
    {'Assumption': 'First-Year Install Rate (Q3)', 'Value': f"{q3_installs}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"First install year ({revenue_start_year}) quarterly installs"},
    {'Assumption': 'First-Year Install Rate (Q4)', 'Value': f"{q4_installs}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"First install year ({revenue_start_year}) quarterly installs"},
    {'Assumption': 'Year 2 Annual Installs', 'Value': f"{year2_installs}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"Applies in {revenue_start_year + 1}"},
    {'Assumption': 'Year 3+ Annual Installs', 'Value': f"{year3_installs}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"Applies in {revenue_start_year + 2} and beyond"},
    {'Assumption': 'Model Years', 'Value': f"{years[0]}-{years[-1]}", 'Units': 'Years', 'Type': 'Hardwired', 'Notes': 'Annual model projection period'},
    {'Assumption': 'Certification Spend Schedule', 'Value': ', '.join([f"{k}:{v:.1f}" for k, v in cert_spend_by_year.items()]), 'Units': '$M', 'Type': 'Calculated', 'Notes': 'Evenly allocated per quarter starting in 2026 based on certification duration'},
    {'Assumption': 'OpEx Schedule', 'Value': ', '.join([f"{k}:{v}" for k, v in opex.items()]), 'Units': '$M/year', 'Type': 'Hardwired', 'Notes': 'OpEx by year; defaults to 15 after 2035'},
    {'Assumption': 'Taxes Floor', 'Value': 'Taxes = max(0, taxable income) * tax rate', 'Units': '', 'Type': 'Hardwired', 'Notes': 'No tax benefit modeled for losses'},
    {'Assumption': 'Terminal Value Condition', 'Value': 'Only computed if WACC > terminal growth', 'Units': '', 'Type': 'Hardwired', 'Notes': 'Otherwise terminal value treated as 0'},
]

assumptions_df = pd.DataFrame(assumptions_rows, columns=['Assumption', 'Value', 'Units', 'Type', 'Notes'])

st.header('Assumptions Appendix')
st.dataframe(assumptions_df, use_container_width=True)

# ── Export: PDF + Excel ──────────────────────────────────────────────────
st.header('Export Reports')

col_pdf, col_xlsx = st.columns(2)

with col_pdf:
    if st.button('Generate PDF Report'):
        pdf_buffer = BytesIO()
        with PdfPages(pdf_buffer) as pdf:
            fig_table = plt.figure(figsize=(17, 11))
            ax_table = fig_table.add_subplot(111)
            ax_table.axis('off')
            df_pdf = df_display.copy()
            kit_price_col = None
            if 'Kit Price ($/kit)' in df_pdf.columns:
                kit_price_col = list(df_pdf.columns).index('Kit Price ($/kit)')
                df_pdf['Kit Price ($/kit)'] = df_pdf['Kit Price ($/kit)'].apply(
                    lambda v: f"{(round(v / 100000.0) * 100000.0):,.0f}" if pd.notna(v) else ""
                )
            table = ax_table.table(cellText=df_pdf.values, colLabels=df_pdf.columns, rowLabels=df_pdf.index, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(6)
            table.auto_set_column_width(col=list(range(len(df_pdf.columns))))
            table.scale(1.0, 1.4)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', fontsize=7)
                    cell.set_facecolor('#E6E6E6')
                    cell._text.set_rotation(45)
                    cell._text.set_rotation_mode('anchor')
                    cell._text.set_ha('left')
                    cell._text.set_va('bottom')
                    cell._text.set_position((cell.get_x() + 0.01, cell.get_y() + 0.02))
                    cell.set_height(cell.get_height() * 1.6)
                if kit_price_col is not None and int(row) > 0 and int(col) == int(kit_price_col):
                    cell.get_text().set_ha('right')
            fig_table.suptitle('Financial Projections Table')
            pdf.savefig(fig_table, bbox_inches='tight')

            # Matplotlib versions of the charts for PDF
            _years_arr = df.index
            _x = np.arange(len(_years_arr))
            _w = 0.25
            fig_mpl, ax_mpl = plt.subplots(1, 1, figsize=(12, 6))
            ax_mpl.bar(_x - _w, df['Revenue ($M)'], width=_w, color='orange', label='Revenue')
            ax_mpl.bar(_x, df['Gross Profit ($M)'], width=_w, color='green', label='Gross Profit')
            ax_mpl.bar(_x + _w, df['Free Cash Flow ($M)'], width=_w, color='blue', label='Free Cash Flow')
            ax_mpl.set_title('Annual Revenue, Gross Profit, and Free Cash Flow')
            ax_mpl.set_ylabel('$M')
            ax_mpl.set_xticks(_x)
            ax_mpl.set_xticklabels(_years_arr)
            ax_mpl.legend(ncol=3)
            ax_mpl.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            pdf.savefig(fig_mpl, bbox_inches='tight')

            fig_mpl_cum, ax_mpl_cum = plt.subplots(1, 1, figsize=(12, 4))
            ax_mpl_cum.plot(_years_arr, cumulative_cash, color='purple', marker='o', linewidth=2, label='Cumulative Free Cash Flow')
            ax_mpl_cum.axhline(0, color='black', linewidth=1, alpha=0.4)
            ax_mpl_cum.set_title('Cumulative Cash (Cumulative Free Cash Flow)')
            ax_mpl_cum.set_ylabel('$M')
            ax_mpl_cum.legend()
            ax_mpl_cum.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            pdf.savefig(fig_mpl_cum, bbox_inches='tight')

            fig_pl = plt.figure(figsize=(17, 11))
            ax_pl = fig_pl.add_subplot(111)
            ax_pl.axis('off')
            pl_tbl = ax_pl.table(cellText=pl_df.T.round(1).values, rowLabels=pl_df.T.index, colLabels=pl_df.T.columns, loc='center', cellLoc='center')
            pl_tbl.auto_set_font_size(False)
            pl_tbl.set_fontsize(8)
            pl_tbl.scale(1.0, 1.4)
            fig_pl.suptitle('P&L Statement')
            pdf.savefig(fig_pl, bbox_inches='tight')

            fig_bs = plt.figure(figsize=(17, 11))
            ax_bs = fig_bs.add_subplot(111)
            ax_bs.axis('off')
            bs_tbl = ax_bs.table(cellText=bs_df.T.round(1).values, rowLabels=bs_df.T.index, colLabels=bs_df.T.columns, loc='center', cellLoc='center')
            bs_tbl.auto_set_font_size(False)
            bs_tbl.set_fontsize(8)
            bs_tbl.scale(1.0, 1.4)
            fig_bs.suptitle('Balance Sheet')
            pdf.savefig(fig_bs, bbox_inches='tight')

            fig_cf_mpl = plt.figure(figsize=(17, 11))
            ax_cf_mpl = fig_cf_mpl.add_subplot(111)
            ax_cf_mpl.axis('off')
            cf_tbl = ax_cf_mpl.table(cellText=cf_df.T.round(1).values, rowLabels=cf_df.T.index, colLabels=cf_df.T.columns, loc='center', cellLoc='center')
            cf_tbl.auto_set_font_size(False)
            cf_tbl.set_fontsize(8)
            cf_tbl.scale(1.0, 1.4)
            fig_cf_mpl.suptitle('Statement of Cash Flows')
            pdf.savefig(fig_cf_mpl, bbox_inches='tight')

            fig_dcf_mpl = plt.figure(figsize=(17, 11))
            ax_dcf_mpl = fig_dcf_mpl.add_subplot(111)
            ax_dcf_mpl.axis('off')
            dcf_tbl = ax_dcf_mpl.table(cellText=dcf_df.T.values, rowLabels=dcf_df.T.index, colLabels=dcf_df.T.columns, loc='center', cellLoc='center')
            dcf_tbl.auto_set_font_size(False)
            dcf_tbl.set_fontsize(8)
            dcf_tbl.scale(1.0, 1.4)
            fig_dcf_mpl.suptitle('DCF Analysis')
            pdf.savefig(fig_dcf_mpl, bbox_inches='tight')

            fig_assumptions_mpl = plt.figure(figsize=(17, 11))
            ax_assumptions_mpl = fig_assumptions_mpl.add_subplot(111)
            ax_assumptions_mpl.axis('off')
            assumptions_tbl = ax_assumptions_mpl.table(
                cellText=assumptions_df.values,
                colLabels=assumptions_df.columns,
                loc='center',
                cellLoc='left'
            )
            assumptions_tbl.auto_set_font_size(False)
            assumptions_tbl.set_fontsize(8)
            assumptions_tbl.auto_set_column_width(col=list(range(len(assumptions_df.columns))))
            assumptions_tbl.scale(1.0, 1.4)
            for (row, col), cell in assumptions_tbl.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', fontsize=9)
                    cell.set_facecolor('#E6E6E6')
            fig_assumptions_mpl.suptitle('Assumptions Appendix')
            pdf.savefig(fig_assumptions_mpl, bbox_inches='tight')

            fig_ev = plt.figure(figsize=(17, 11))
            ax_ev = fig_ev.add_subplot(111)
            ax_ev.axis('off')
            ax_ev.text(0.02, 0.80, 'Enterprise Value Summary', fontsize=20, weight='bold')
            ax_ev.text(0.02, 0.65, f"Enterprise Value ($M): {enterprise_value:.1f}", fontsize=16)
            ax_ev.text(0.02, 0.55, f"WACC: {wacc * 100:.2f}%", fontsize=12)
            ax_ev.text(0.02, 0.49, f"Terminal Growth Rate: {terminal_growth * 100:.2f}%", fontsize=12)
            pdf.savefig(fig_ev, bbox_inches='tight')

        pdf_buffer.seek(0)
        st.markdown(DOWNLOAD_BUTTON_CSS, unsafe_allow_html=True)
        st.download_button(
            label="Download Standalone Model PDF",
            data=pdf_buffer,
            file_name="Tamarack_Financial_Report.pdf",
            mime="application/pdf",
        )

with col_xlsx:
    xlsx_buffer = BytesIO()
    with pd.ExcelWriter(xlsx_buffer, engine='openpyxl') as writer:
        df_display.to_excel(writer, sheet_name='Projections')
        pl_df.to_excel(writer, sheet_name='P&L')
        bs_df.to_excel(writer, sheet_name='Balance Sheet')
        cf_df.to_excel(writer, sheet_name='Cash Flows')
        dcf_df.to_excel(writer, sheet_name='DCF')
        dcf_summary_df.to_excel(writer, sheet_name='DCF Summary', index=False)
        assumptions_df.to_excel(writer, sheet_name='Assumptions', index=False)
    xlsx_buffer.seek(0)
    st.markdown(DOWNLOAD_BUTTON_CSS, unsafe_allow_html=True)
    st.download_button(
        label="Download Excel Workbook",
        data=xlsx_buffer,
        file_name="Tamarack_Financial_Model.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )