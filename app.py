import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO
from pathlib import Path
from typing import Dict

from app_sense import render_sensitivity_app

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

st.markdown(
    """
<style>
div[data-testid="stRadio"] {
  background: #F7FAFF;
  border: 2px solid #3B82F6;
  border-radius: 12px;
  padding: 12px 14px;
  margin: 6px 0 14px 0;
}
div[data-testid="stRadio"] label p {
  font-weight: 700 !important;
  font-size: 1.05rem !important;
}
div[data-testid="stRadio"] div[role="radiogroup"] {
  gap: 10px;
  flex-wrap: nowrap;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label {
  background: #FFFFFF;
  border: 1px solid #BFDBFE;
  border-radius: 10px;
  padding: 8px 10px;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
  border-color: #1D4ED8;
  background: #DBEAFE;
  box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.18);
}

div[data-testid="stMetric"] {
  background: #ECFDF5;
  border: 3px solid #10B981;
  border-radius: 16px;
  padding: 16px 18px;
  box-shadow: 0 12px 22px rgba(16, 185, 129, 0.18);
}
div[data-testid="stMetric"] label p {
  font-weight: 900 !important;
}
div[data-testid="stMetric"] label {
  font-weight: 900 !important;
  font-size: 1.05rem !important;
  color: #065F46 !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-weight: 900 !important;
  font-size: 2.4rem !important;
  color: #064E3B !important;
}
</style>
""",
    unsafe_allow_html=True,
)

col_main, col_right = st.columns([0.6, 3.4])
with col_right:
    def _biz_label(v: str) -> str:
        return 'Leasing' if str(v).startswith('Leasing') else 'Kit Sale'

    def _mode_label(v: str) -> str:
        return 'Standalone' if str(v).startswith('Standalone') else 'Sensitivity'

    col_biz, col_mode, col_ev = st.columns([1.1, 1.6, 1.3], gap="small")
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
with col_main:
    pass

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

st.sidebar.header('First-Year Install Rates (Kits per Quarter)')
q1_installs = st.sidebar.slider('Q1 Installs', min_value=0, max_value=200, value=98, step=10)  # ~10/week * 13 weeks / 4 = approx
q2_installs = st.sidebar.slider('Q2 Installs', min_value=0, max_value=200, value=98, step=10)
q3_installs = st.sidebar.slider('Q3 Installs', min_value=0, max_value=200, value=98, step=10)
q4_installs = st.sidebar.slider('Q4 Installs and beyond', min_value=0, max_value=200, value=96, step=10)  # Total ~390 for year

# Fixed assumptions (from previous)
split_pct = fuel_savings_split_to_tamarack

# Cert costs split (assuming even over 2026-2027)
cert_spend_by_year = {}
cert_spend_per_quarter = (float(cert_readiness_cost) / float(cert_duration_quarters)) if int(cert_duration_quarters) > 0 else 0.0
for q in range(int(cert_duration_quarters)):
    yr = 2026 + (q // 4)
    cert_spend_by_year[yr] = cert_spend_by_year.get(yr, 0.0) + cert_spend_per_quarter

revenue_start_q_index = int(cert_duration_quarters)
revenue_start_year = 2026 + (int(revenue_start_q_index) // 4)
revenue_start_quarter = (int(revenue_start_q_index) % 4) + 1
inventory_purchase_q_index = max(0, int(revenue_start_q_index) - 1)
inventory_year = 2026 + (int(inventory_purchase_q_index) // 4)
inventory_quarter = (int(inventory_purchase_q_index) % 4) + 1

# OpEx fixed for simplicity (lean case)
opex = {2026: 50, 2027: 40, 2028: 40, 2029: 35, 2030: 25, 2031: 20, 2032: 18, 2033: 15, 2034: 15, 2035: 15}

# Years
years = list(range(2026, 2036))  # 10 years

corsia_assumption_rows = [] if model_type == 'Kit Sale (Payback Pricing)' else [
    {'Assumption': 'CORSIA Exposure (Share of Ops)', 'Value': f"{corsia_split * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Share of operations subject to CORSIA compliance pricing (applied starting in first revenue quarter)'},
    {'Assumption': 'Carbon Price', 'Value': f"{carbon_price:.2f}", 'Units': '$/tCO2', 'Type': 'Slider', 'Notes': 'Used to value avoided CORSIA compliance cost (fuel savings + avoided CORSIA = total value created)'},
]

assumptions_rows = [
    {'Assumption': 'Annual Fuel Inflation', 'Value': f"{fuel_inflation * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to base fuel price starting in the first revenue year'},
    {'Assumption': 'Base Fuel Price (First Revenue Year)', 'Value': f"{base_fuel_price:.2f}", 'Units': '$/gal', 'Type': 'Slider', 'Notes': f"Base fuel price used in {revenue_start_year}"},
    {'Assumption': 'Block Hours per Aircraft per Year', 'Value': f"{int(block_hours)}", 'Units': 'Hours', 'Type': 'Slider', 'Notes': 'Used to compute annual fuel spend'},
    {'Assumption': 'Base Fuel Burn', 'Value': f"{int(base_fuel_burn_gal_per_hour)}", 'Units': 'Gal/hour', 'Type': 'Slider', 'Notes': 'Used to compute annual fuel spend'},
    {'Assumption': 'Annual COGS Inflation', 'Value': f"{cogs_inflation * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to base COGS per kit starting in the first revenue year'},
    {'Assumption': 'Base COGS per Kit (First Revenue Year)', 'Value': f"{base_cogs:,.0f}", 'Units': '$/kit', 'Type': 'Slider', 'Notes': f"Input slider is in $000; base COGS per kit used in {revenue_start_year}; also used for inventory build"},
    {'Assumption': 'Fuel Savings % per Aircraft', 'Value': f"{fuel_saving_pct * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Percent of annual fuel spend saved'},
    ({'Assumption': 'Target Airline Payback', 'Value': f"{float(target_payback_years):.2f}", 'Units': 'Years', 'Type': 'Slider', 'Notes': 'Kit price is set so the airline recovers cost via fuel savings over the target payback period'} if model_type == 'Kit Sale (Payback Pricing)' else {'Assumption': 'Fuel Savings Split to Tamarack', 'Value': f"{fuel_savings_split_to_tamarack * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Percent of annual fuel savings paid to Tamarack'}),
    *corsia_assumption_rows,
    {'Assumption': 'Certification Duration', 'Value': f"{float(cert_duration_years):.2f}", 'Units': 'Years', 'Type': 'Slider', 'Notes': f"{cert_duration_quarters} quarters; go-live is {revenue_start_year}Q{revenue_start_quarter}"},
    {'Assumption': 'Equity', 'Value': f"{cert_readiness_cost:.1f}", 'Units': '$M', 'Type': 'Slider', 'Notes': f"Used first to fund certification / inventory outflows prior to {revenue_start_year}Q{revenue_start_quarter}"},
    {'Assumption': 'Max Debt Available', 'Value': f"{debt_amount:.1f}", 'Units': '$M', 'Type': 'Slider', 'Notes': f"Debt facility cap; model draws only what is needed prior to {revenue_start_year}Q{revenue_start_quarter}"},
    {'Assumption': 'Debt APR', 'Value': f"{debt_apr * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to outstanding debt balance'},
    {'Assumption': 'Debt Term', 'Value': f"{debt_term_years}", 'Units': 'Years', 'Type': 'Slider', 'Notes': f"Debt amortizes quarterly beginning in {revenue_start_year}Q{revenue_start_quarter}"},
    {'Assumption': 'Income Tax Rate', 'Value': f"{tax_rate * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Taxes apply only when taxable income is positive'},
    {'Assumption': 'WACC', 'Value': f"{wacc * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Used to discount unlevered free cash flows in DCF'},
    {'Assumption': 'Terminal Growth Rate', 'Value': f"{terminal_growth * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Used for terminal value if WACC > terminal growth'},
    {'Assumption': 'Inventory Kits Before First Install', 'Value': f"{int(inventory_kits_pre_install)}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"Purchased in {inventory_year}Q{inventory_quarter} (1 quarter before go-live; 25% of full build)"},
    {'Assumption': 'Total Addressable Market', 'Value': f"{int(tam_shipsets)}", 'Units': 'Aircraft', 'Type': 'Slider', 'Notes': 'Starting eligible aftermarket fleet size (used as the base TAM)'},
    {'Assumption': 'TAM Penetration', 'Value': f"{tam_penetration_pct * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Caps maximum installable fleet at TAM * penetration'},
    {'Assumption': 'Fleet Retirements', 'Value': f"{int(fleet_retirements_per_month)}", 'Units': 'Aircraft/month', 'Type': 'Slider', 'Notes': 'Reduces the eligible fleet over time; also retires a proportional share of installed aircraft'},
    {'Assumption': 'Forward-Fit Enabled', 'Value': 'Yes' if bool(include_forward_fit) else 'No', 'Units': '', 'Type': 'Toggle', 'Notes': 'If enabled, adds new aircraft to the eligible fleet over time'},
    {'Assumption': 'Forward-Fit Additions', 'Value': f"{int(forward_fit_per_month)}", 'Units': 'Aircraft/month', 'Type': 'Slider', 'Notes': 'Adds to the eligible fleet over time when forward-fit is enabled'},
    {'Assumption': 'First-Year Install Rate (Q1)', 'Value': f"{int(q1_installs)}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"First install year ({revenue_start_year}) quarterly installs"},
    {'Assumption': 'First-Year Install Rate (Q2)', 'Value': f"{int(q2_installs)}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"First install year ({revenue_start_year}) quarterly installs"},
    {'Assumption': 'First-Year Install Rate (Q3)', 'Value': f"{int(q3_installs)}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"First install year ({revenue_start_year}) quarterly installs"},
    {'Assumption': 'First-Year Install Rate (Q4)', 'Value': f"{int(q4_installs)}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': f"First install year ({revenue_start_year}) quarterly installs"},
    {'Assumption': 'Model Years', 'Value': f"{years[0]}-{years[-1]}", 'Units': 'Years', 'Type': 'Hardwired', 'Notes': 'Annual model projection period'},
    {'Assumption': 'Certification Spend Schedule', 'Value': ', '.join([f"{k}:{v:.1f}" for k, v in cert_spend_by_year.items()]), 'Units': '$M', 'Type': 'Calculated', 'Notes': 'Evenly allocated per quarter starting in 2026 based on certification duration'},
    {'Assumption': 'Install Ramp (Year 2 New Installs)', 'Value': '910', 'Units': 'Kits', 'Type': 'Hardwired', 'Notes': f"Applies in {revenue_start_year + 1}"},
    {'Assumption': 'Install Ramp (Year 3+ New Installs)', 'Value': '1040', 'Units': 'Kits', 'Type': 'Hardwired', 'Notes': f"Applies in {revenue_start_year + 2} and beyond"},
    {'Assumption': 'OpEx Schedule', 'Value': ', '.join([f"{k}:{v}" for k, v in opex.items()]), 'Units': '$M/year', 'Type': 'Hardwired', 'Notes': 'OpEx by year; defaults to 15 after 2035'},
    {'Assumption': 'Taxes Floor', 'Value': 'Taxes = max(0, taxable income) * tax rate', 'Units': '', 'Type': 'Hardwired', 'Notes': 'No tax benefit modeled for losses'},
    {'Assumption': 'Terminal Value Condition', 'Value': 'Only computed if WACC > terminal growth', 'Units': '', 'Type': 'Hardwired', 'Notes': 'Otherwise terminal value treated as 0'},
]

assumptions_df = pd.DataFrame(assumptions_rows, columns=['Assumption', 'Value', 'Units', 'Type', 'Notes'])

# Calculations
annual_data = {}

fleet_size = float(tam_shipsets)
installed_base = 0.0
cum_cash = 0.0

debt_balance = 0.0
debt_draw_remaining = float(debt_amount)
debt_drawn_total = 0.0

investor_cum_cf = 0.0
equity_cum_cf = 0.0
equity_reserve = float(cert_readiness_cost)
equity_amount = float(cert_readiness_cost)

debt_rate_q = float(debt_apr) / 4.0
term_quarters = int(debt_term_years) * 4
quarterly_debt_payment = None

year_sums = None
year_taxable_income = 0.0

for i in range(len(years) * 4):
    yr = years[0] + (i // 4)
    qtr = (i % 4) + 1

    if year_sums is None:
        year_sums = {
            'New Installs': 0.0,
            'Kit Price ($/kit) Sum': 0.0,
            'Kit Price Qtrs': 0.0,
            'Revenue ($M)': 0.0,
            'COGS ($M)': 0.0,
            'Gross Profit ($M)': 0.0,
            'OpEx ($M)': 0.0,
            'EBITDA ($M)': 0.0,
            'CapEx/Inv ($M)': 0.0,
            'Free Cash Flow ($M)': 0.0,
            'Taxes ($M)': 0.0,
            'FCF After Tax ($M)': 0.0,
            'Debt Draw ($M)': 0.0,
            'Debt Payment ($M)': 0.0,
            'Debt Interest ($M)': 0.0,
            'Debt Principal ($M)': 0.0,
            'Net Cash After Debt ($M)': 0.0,
            'Net Cash Change ($M)': 0.0,
            'Debt Investor CF ($M)': 0.0,
            'Equity Contribution ($M)': 0.0,
            'Equity CF ($M)': 0.0,
        }
        year_taxable_income = 0.0

    capex = 0.0
    inventory = 0.0

    fleet_beg = float(fleet_size)
    installed_beg = float(installed_base)

    retire_q = float(fleet_retirements_per_month) * 3.0
    retire_q = min(float(retire_q), float(fleet_beg)) if float(fleet_beg) > 0 else 0.0
    forward_fit_q = (float(forward_fit_per_month) * 3.0) if bool(include_forward_fit) else 0.0

    fleet_size = max(0.0, float(fleet_beg) - float(retire_q) + float(forward_fit_q))
    installed_base = float(installed_beg)

    installable_cap = float(fleet_size) * float(tam_penetration_pct)

    if i < revenue_start_q_index:
        new_installs = 0.0
        revenue = 0.0
        cogs = 0.0
        kit_price = np.nan

        if i < int(cert_duration_quarters):
            capex = float(cert_spend_per_quarter)
        if i == int(inventory_purchase_q_index):
            inventory = float(0.25 * inventory_kits_pre_install * base_cogs / 1e6)
    else:
        year_idx = int(yr - revenue_start_year)
        fuel_price = float(base_fuel_price) * float((1 + float(fuel_inflation)) ** int(year_idx))
        quarter_block_hours = float(block_hours) / 4.0
        quarter_fuel_spend = quarter_block_hours * float(base_fuel_burn_gal_per_hour) * float(fuel_price)

        quarter_saving = quarter_fuel_spend * float(fuel_saving_pct)
        quarter_gallons_burn = quarter_block_hours * float(base_fuel_burn_gal_per_hour)
        gallons_saved = quarter_gallons_burn * float(fuel_saving_pct)
        fuel_saved_tonnes = gallons_saved * 0.00304
        co2_avoided_t = fuel_saved_tonnes * 3.16
        corsia_value = 0.0 if model_type == 'Kit Sale (Payback Pricing)' else (co2_avoided_t * float(corsia_split) * float(carbon_price))
        total_value_created = quarter_saving + corsia_value

        rev_q_idx = int(i - int(revenue_start_q_index))
        planned_installs = 0.0
        if rev_q_idx == 0:
            planned_installs = float(q1_installs)
        elif rev_q_idx == 1:
            planned_installs = float(q2_installs)
        elif rev_q_idx == 2:
            planned_installs = float(q3_installs)
        elif rev_q_idx == 3:
            planned_installs = float(q4_installs)
        else:
            revenue_year = int(rev_q_idx // 4)
            if revenue_year == 1:
                planned_installs = 910.0 / 4.0
            else:
                planned_installs = 1040.0 / 4.0

        remaining_capacity = max(0.0, float(installable_cap) - float(installed_base))
        new_installs = min(float(planned_installs), float(remaining_capacity))

        installed_base = float(installed_base) + float(new_installs)

        kit_price = np.nan

        if model_type == 'Kit Sale (Payback Pricing)':
            annual_value_created = float(total_value_created) * 4.0
            rev_per_kit = float(annual_value_created) * float(target_payback_years)
            kit_price = float(rev_per_kit)
            revenue = float(new_installs) * float(rev_per_kit) / 1e6
        else:
            rev_per_shipset = total_value_created * float(split_pct)
            avg_installed = float(installed_base) - 0.5 * float(new_installs)
            revenue = float(avg_installed) * float(rev_per_shipset) / 1e6

        cogs_per_kit = float(base_cogs) * float((1 + float(cogs_inflation)) ** int(year_idx))
        cogs = float(new_installs) * float(cogs_per_kit) / 1e6

    gross_profit = float(revenue) - float(cogs)
    opex_q = float(opex.get(int(yr), 15)) / 4.0
    ebitda = float(gross_profit) - float(opex_q)
    total_outflow = float(capex) + float(inventory)
    fcf = float(ebitda) - float(total_outflow)

    equity_contribution = 0.0
    debt_draw = 0.0
    if i < revenue_start_q_index:
        equity_contribution = min(float(equity_reserve), float(total_outflow))
        equity_reserve -= float(equity_contribution)
        remaining_outflow = float(total_outflow) - float(equity_contribution)
        if float(debt_draw_remaining) > 0 and float(remaining_outflow) > 0:
            debt_draw = min(float(debt_draw_remaining), float(remaining_outflow))
            debt_draw_remaining -= float(debt_draw)
            debt_drawn_total += float(debt_draw)

    debt_balance = float(debt_balance) + float(debt_draw)

    debt_interest = 0.0
    debt_principal = 0.0
    debt_payment = 0.0
    if i >= revenue_start_q_index and float(debt_balance) > 0:
        if quarterly_debt_payment is None:
            if float(debt_rate_q) == 0:
                quarterly_debt_payment = (float(debt_balance) / float(term_quarters)) if int(term_quarters) > 0 else 0.0
            else:
                quarterly_debt_payment = float(debt_balance) * float(debt_rate_q) / (1 - (1 + float(debt_rate_q)) ** (-float(term_quarters)))

        debt_interest = float(debt_balance) * float(debt_rate_q)
        debt_payment = min(float(quarterly_debt_payment), float(debt_balance) + float(debt_interest))
        debt_principal = max(0.0, min(float(debt_balance), float(debt_payment) - float(debt_interest)))
        debt_balance = max(0.0, float(debt_balance) - float(debt_principal))

    taxable_income_q = float(ebitda) - float(debt_interest)
    year_taxable_income += float(taxable_income_q)

    taxes = 0.0
    if int(qtr) == 4:
        taxes = max(0.0, float(year_taxable_income)) * float(tax_rate)

    fcf_after_tax = float(ebitda) - float(taxes) - float(total_outflow)
    net_cash_after_debt = float(fcf_after_tax) + float(debt_draw) - float(debt_payment)
    net_cash_change = float(net_cash_after_debt) + float(equity_contribution)
    cum_cash += float(net_cash_change)

    investor_cf = (-float(debt_draw)) + float(debt_payment)
    investor_cum_cf += float(investor_cf)
    investor_roi = (float(investor_cum_cf) / float(debt_drawn_total)) if float(debt_drawn_total) > 0 else 0.0

    if i < revenue_start_q_index:
        equity_cf = -float(equity_contribution)
    else:
        equity_cf = float(net_cash_after_debt)

    equity_cum_cf += float(equity_cf)
    equity_roi = (float(equity_cum_cf) / float(equity_amount)) if float(equity_amount) > 0 else 0.0

    year_sums['New Installs'] += float(new_installs)
    if not np.isnan(kit_price):
        year_sums['Kit Price ($/kit) Sum'] += float(kit_price)
        year_sums['Kit Price Qtrs'] += 1.0
    year_sums['Revenue ($M)'] += float(revenue)
    year_sums['COGS ($M)'] += float(cogs)
    year_sums['Gross Profit ($M)'] += float(gross_profit)
    year_sums['OpEx ($M)'] += float(opex_q)
    year_sums['EBITDA ($M)'] += float(ebitda)
    year_sums['CapEx/Inv ($M)'] += float(total_outflow)
    year_sums['Free Cash Flow ($M)'] += float(fcf)
    year_sums['Taxes ($M)'] += float(taxes)
    year_sums['FCF After Tax ($M)'] += float(fcf_after_tax)
    year_sums['Debt Draw ($M)'] += float(debt_draw)
    year_sums['Debt Payment ($M)'] += float(debt_payment)
    year_sums['Debt Interest ($M)'] += float(debt_interest)
    year_sums['Debt Principal ($M)'] += float(debt_principal)
    year_sums['Net Cash After Debt ($M)'] += float(net_cash_after_debt)
    year_sums['Net Cash Change ($M)'] += float(net_cash_change)
    year_sums['Debt Investor CF ($M)'] += float(investor_cf)
    year_sums['Equity Contribution ($M)'] += float(equity_contribution)
    year_sums['Equity CF ($M)'] += float(equity_cf)

    if int(qtr) == 4:
        avg_kit_price = (float(year_sums['Kit Price ($/kit) Sum']) / float(year_sums['Kit Price Qtrs'])) if float(year_sums['Kit Price Qtrs']) > 0 else np.nan
        annual_data[int(yr)] = {
            'New Installs': int(round(float(year_sums['New Installs']), 0)),
            'Cum Shipsets': int(round(float(installed_base), 0)),
            'Kit Price ($/kit)': round(float(avg_kit_price), 0) if not np.isnan(avg_kit_price) else np.nan,
            'Revenue ($M)': round(float(year_sums['Revenue ($M)']), 1),
            'COGS ($M)': round(float(year_sums['COGS ($M)']), 1),
            'Gross Profit ($M)': round(float(year_sums['Gross Profit ($M)']), 1),
            'OpEx ($M)': round(float(year_sums['OpEx ($M)']), 1),
            'EBITDA ($M)': round(float(year_sums['EBITDA ($M)']), 1),
            'CapEx/Inv ($M)': round(float(year_sums['CapEx/Inv ($M)']), 1),
            'Free Cash Flow ($M)': round(float(year_sums['Free Cash Flow ($M)']), 1),
            'Taxes ($M)': round(float(year_sums['Taxes ($M)']), 1),
            'FCF After Tax ($M)': round(float(year_sums['FCF After Tax ($M)']), 1),
            'Debt Draw ($M)': round(float(year_sums['Debt Draw ($M)']), 1),
            'Debt Payment ($M)': round(float(year_sums['Debt Payment ($M)']), 1),
            'Debt Interest ($M)': round(float(year_sums['Debt Interest ($M)']), 1),
            'Debt Principal ($M)': round(float(year_sums['Debt Principal ($M)']), 1),
            'Debt Balance ($M)': round(float(debt_balance), 1),
            'Net Cash After Debt ($M)': round(float(year_sums['Net Cash After Debt ($M)']), 1),
            'Net Cash Change ($M)': round(float(year_sums['Net Cash Change ($M)']), 1),
            'Cumulative Cash ($M)': round(float(cum_cash), 1),
            'Debt Investor CF ($M)': round(float(year_sums['Debt Investor CF ($M)']), 1),
            'Debt Investor Cum CF ($M)': round(float(investor_cum_cf), 1),
            'Debt Investor ROI (%)': round(float(investor_roi) * 100, 1),
            'Equity Contribution ($M)': round(float(year_sums['Equity Contribution ($M)']), 1),
            'Equity CF ($M)': round(float(year_sums['Equity CF ($M)']), 1),
            'Equity Cum CF ($M)': round(float(equity_cum_cf), 1),
            'Equity ROI (%)': round(float(equity_roi) * 100, 1),
        }
        year_sums = None

df = pd.DataFrame(annual_data).T

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

unlevered_taxes_raw = (df['EBITDA ($M)'].clip(lower=0.0) * float(tax_rate)).astype(float)
unlevered_fcf_raw = (df['EBITDA ($M)'] - unlevered_taxes_raw - df['CapEx/Inv ($M)']).astype(float)
discount_year0 = int(df.index.min())
discount_t = (df.index - discount_year0 + 1).astype(int)
discount_factor = pd.Series((1 / (1 + float(wacc)) ** discount_t).astype(float), index=df.index)
pv_fcf_raw = (unlevered_fcf_raw * discount_factor).astype(float)

unlevered_taxes = unlevered_taxes_raw.round(1)
unlevered_fcf = unlevered_fcf_raw.round(1)
pv_fcf = pv_fcf_raw.round(1)

tv = np.nan
pv_tv = np.nan
if float(wacc) > float(terminal_growth):
    tv = float(unlevered_fcf.iloc[-1]) * (1 + float(terminal_growth)) / (float(wacc) - float(terminal_growth))
    pv_tv = tv * float(discount_factor.iloc[-1])

dcf_df = pd.DataFrame({
    'Unlevered FCF ($M)': unlevered_fcf,
    'Discount Factor': discount_factor.round(4),
    'PV of FCF ($M)': pv_fcf,
}, index=df.index)

pv_explicit = float(pv_fcf_raw.sum())
enterprise_value = pv_explicit + (float(pv_tv) if not np.isnan(pv_tv) else 0.0)

ev_placeholder.metric(label="Enterprise Value ($M)", value=f"{enterprise_value:,.1f}")

dcf_summary_df = pd.DataFrame({
    'PV Explicit FCF ($M)': [round(pv_explicit, 1)],
    'Terminal Value ($M)': [round(0.0 if np.isnan(tv) else float(tv), 1)],
    'PV Terminal Value ($M)': [round(0.0 if np.isnan(pv_tv) else float(pv_tv), 1)],
    'Enterprise Value ($M)': [round(enterprise_value, 1)],
})

df_display = df.copy()
if model_type != 'Kit Sale (Payback Pricing)' and 'Kit Price ($/kit)' in df_display.columns:
    df_display = df_display.drop(columns=['Kit Price ($/kit)'])
elif model_type == 'Kit Sale (Payback Pricing)' and 'Kit Price ($/kit)' in df_display.columns:
    df_display['Kit Price ($/kit)'] = df_display['Kit Price ($/kit)'].apply(
        lambda v: (round(float(v) / 100000.0) * 100000.0) if pd.notna(v) else np.nan
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
        elif _col in ['New Installs', 'Cum Shipsets']:
            _fmt[_col] = '{:,.0f}'

    _align_cols = list(_fmt.keys())
    df_display_view = df_display.style.format(_fmt, na_rep='')
    df_display_view = df_display_view.set_properties(subset=_align_cols, **{'text-align': 'right'})

st.dataframe(df_display_view, use_container_width=True)

st.header('Three-Statement Output')
st.subheader('P&L')
st.dataframe(pl_df, use_container_width=True)
st.subheader('Balance Sheet')
st.dataframe(bs_df, use_container_width=True)
st.subheader('Statement of Cash Flows')
st.dataframe(cf_df, use_container_width=True)

st.header('DCF Analysis')
st.dataframe(dcf_df, use_container_width=True)

st.subheader('DCF Supporting Information')
st.write(f"Discount base year: {discount_year0}")
st.write(f"WACC: {wacc * 100:.2f}%")
st.write(f"Terminal growth rate: {terminal_growth * 100:.2f}%")
st.write(f"PV of explicit period FCF ($M): {pv_explicit:.1f}")
st.write(f"PV of terminal value ($M): {0.0 if np.isnan(pv_tv) else pv_tv:.1f}")
st.dataframe(dcf_summary_df, use_container_width=True)

st.header('Financial Projection Plots (Annual / Non-Cumulative)')

years = df.index
x = np.arange(len(years))
width = 0.25

fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.bar(x - width, df['Revenue ($M)'], width=width, color='orange', label='Revenue')
ax.bar(x, df['Gross Profit ($M)'], width=width, color='green', label='Gross Profit')
ax.bar(x + width, df['Free Cash Flow ($M)'], width=width, color='blue', label='Free Cash Flow')
ax.set_title('Annual Revenue, Gross Profit, and Free Cash Flow')
ax.set_ylabel('$M')
ax.set_xlabel('Year')
ax.set_xticks(x)
ax.set_xticklabels(years)
ax.legend(ncol=3)
ax.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
st.pyplot(fig)

cumulative_cash = df['Free Cash Flow ($M)'].cumsum()
fig_cum, ax_cum = plt.subplots(1, 1, figsize=(12, 4))
ax_cum.plot(years, cumulative_cash, color='purple', marker='o', linewidth=2, label='Cumulative Free Cash Flow')
ax_cum.axhline(0, color='black', linewidth=1, alpha=0.4)
ax_cum.set_title('Cumulative Cash (Cumulative Free Cash Flow)')
ax_cum.set_ylabel('$M')
ax_cum.set_xlabel('Year')
ax_cum.legend()
ax_cum.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
st.pyplot(fig_cum)

st.header('Generate PDF Report')

if st.button('Download PDF Report'):
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
                lambda v: f"{(round(float(v) / 100000.0) * 100000.0):,.0f}" if pd.notna(v) else ""
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

        pdf.savefig(fig, bbox_inches='tight')
        pdf.savefig(fig_cum, bbox_inches='tight')

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

        fig_cf = plt.figure(figsize=(17, 11))
        ax_cf = fig_cf.add_subplot(111)
        ax_cf.axis('off')
        cf_tbl = ax_cf.table(cellText=cf_df.T.round(1).values, rowLabels=cf_df.T.index, colLabels=cf_df.T.columns, loc='center', cellLoc='center')
        cf_tbl.auto_set_font_size(False)
        cf_tbl.set_fontsize(8)
        cf_tbl.scale(1.0, 1.4)
        fig_cf.suptitle('Statement of Cash Flows')
        pdf.savefig(fig_cf, bbox_inches='tight')

        fig_dcf = plt.figure(figsize=(17, 11))
        ax_dcf = fig_dcf.add_subplot(111)
        ax_dcf.axis('off')
        dcf_tbl = ax_dcf.table(cellText=dcf_df.T.values, rowLabels=dcf_df.T.index, colLabels=dcf_df.T.columns, loc='center', cellLoc='center')
        dcf_tbl.auto_set_font_size(False)
        dcf_tbl.set_fontsize(8)
        dcf_tbl.scale(1.0, 1.4)
        fig_dcf.suptitle('DCF Analysis')
        pdf.savefig(fig_dcf, bbox_inches='tight')

        fig_assumptions = plt.figure(figsize=(17, 11))
        ax_assumptions = fig_assumptions.add_subplot(111)
        ax_assumptions.axis('off')
        assumptions_tbl = ax_assumptions.table(
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
        fig_assumptions.suptitle('Assumptions Appendix')
        pdf.savefig(fig_assumptions, bbox_inches='tight')

        fig_ev = plt.figure(figsize=(17, 11))
        ax_ev = fig_ev.add_subplot(111)
        ax_ev.axis('off')
        ax_ev.text(0.02, 0.80, 'Enterprise Value Summary', fontsize=20, weight='bold')
        ax_ev.text(0.02, 0.65, f"Enterprise Value ($M): {enterprise_value:.1f}", fontsize=16)
        ax_ev.text(0.02, 0.55, f"WACC: {wacc * 100:.2f}%", fontsize=12)
        ax_ev.text(0.02, 0.49, f"Terminal Growth Rate: {terminal_growth * 100:.2f}%", fontsize=12)
        pdf.savefig(fig_ev, bbox_inches='tight')

    pdf_buffer.seek(0)
    st.markdown(
        """
<style>
div[data-testid="stDownloadButton"] > button {
  width: auto;
  min-width: 360px;
  background: #1D4ED8 !important;
  color: #FFFFFF !important;
  border: 2px solid #1E40AF !important;
  border-radius: 12px !important;
  padding: 0.85rem 1.25rem !important;
  font-size: 1.15rem !important;
  font-weight: 800 !important;
  letter-spacing: 0.2px !important;
  box-shadow: 0 10px 18px rgba(29, 78, 216, 0.18) !important;
}
div[data-testid="stDownloadButton"] > button:hover {
  background: #1E40AF !important;
  border-color: #1E3A8A !important;
}
div[data-testid="stDownloadButton"] > button:focus {
  box-shadow: 0 0 0 4px rgba(29, 78, 216, 0.25) !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )
    _, c_btn, _ = st.columns([1, 2, 1])
    with c_btn:
        st.download_button(
            label="Download Standalone Model PDF",
            data=pdf_buffer,
            file_name="Tamarack_Financial_Report.pdf",
            mime="application/pdf",
            use_container_width=False,
        )

st.header('Assumptions Appendix')
st.dataframe(assumptions_df, use_container_width=True)