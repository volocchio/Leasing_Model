import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from io import BytesIO

st.set_page_config(layout='wide')

st.title('Tamarack Aerospace A320 Financial Model')

# Simplified Sidebar with key sliders
st.sidebar.header('Key Assumptions')

fuel_inflation = st.sidebar.slider('Annual Fuel Inflation (%)', min_value=0.0, max_value=15.0, value=5.0, step=0.5) / 100
base_fuel_price = st.sidebar.slider('Base Fuel Price in 2028 ($/gal)', min_value=1.0, max_value=6.0, value=3.0, step=0.1)
cogs_inflation = st.sidebar.slider('Annual COGS Inflation (%)', min_value=0.0, max_value=15.0, value=4.0, step=0.5) / 100
base_cogs = st.sidebar.slider('Base COGS per Kit in 2028 ($)', min_value=100000, max_value=800000, value=400000, step=10000)
fuel_saving_pct = st.sidebar.slider('Fuel Savings % per Aircraft', min_value=5.0, max_value=15.0, value=10.0, step=0.5) / 100
fuel_savings_split_to_tamarack = st.sidebar.slider('Fuel Savings Split to Tamarack (%)', min_value=0.0, max_value=100.0, value=50.0, step=1.0) / 100
cert_readiness_cost = st.sidebar.slider('Equity ($M)', min_value=100.0, max_value=300.0, value=180.0, step=10.0)

inventory_kits_pre_install = st.sidebar.slider('Inventory Kits Before First Install', min_value=50, max_value=200, value=130, step=10)
tam_shipsets = st.sidebar.slider('Total Addressable Market (Max Shipsets in 10 Years)', min_value=1000, max_value=10000, value=7500, step=500)

debt_amount = st.sidebar.slider('Debt Raised ($M)', min_value=0.0, max_value=500.0, value=float(cert_readiness_cost), step=10.0)
debt_apr = st.sidebar.slider('Debt APR (%)', min_value=0.0, max_value=20.0, value=10.0, step=0.5) / 100
debt_term_years = st.sidebar.slider('Debt Term (Years)', min_value=1, max_value=15, value=7, step=1)
tax_rate = st.sidebar.slider('Income Tax Rate (%)', min_value=0.0, max_value=40.0, value=21.0, step=0.5) / 100
wacc = st.sidebar.slider('WACC (%)', min_value=0.0, max_value=30.0, value=11.5, step=0.5) / 100
terminal_growth = st.sidebar.slider('Terminal Growth Rate (%)', min_value=-2.0, max_value=8.0, value=3.0, step=0.5) / 100

# Install rates for first 4 quarters (first install year, e.g., 2028)
st.sidebar.header('First Year Install Rates (Kits per Quarter)')
q1_installs = st.sidebar.slider('Q1 Installs', min_value=0, max_value=200, value=98, step=10)  # ~10/week * 13 weeks / 4 = approx
q2_installs = st.sidebar.slider('Q2 Installs', min_value=0, max_value=200, value=98, step=10)
q3_installs = st.sidebar.slider('Q3 Installs', min_value=0, max_value=200, value=98, step=10)
q4_installs = st.sidebar.slider('Q4 Installs and beyond', min_value=0, max_value=200, value=96, step=10)  # Total ~390 for year

# Fixed assumptions (from previous)
block_hours = 2800
base_fuel_burn_gal_per_hour = 640
split_pct = fuel_savings_split_to_tamarack

# Cert costs split (assuming even over 2026-2027)
cert_2026 = cert_readiness_cost / 2
cert_2027 = cert_readiness_cost / 2

# OpEx fixed for simplicity (lean case)
opex = {2026: 50, 2027: 40, 2028: 40, 2029: 35, 2030: 25, 2031: 20, 2032: 18, 2033: 15, 2034: 15, 2035: 15}

# Years
years = list(range(2026, 2036))  # 10 years

assumptions_rows = [
    {'Assumption': 'Annual Fuel Inflation', 'Value': f"{fuel_inflation * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to base fuel price starting in 2028'},
    {'Assumption': 'Base Fuel Price (2028)', 'Value': f"{base_fuel_price:.2f}", 'Units': '$/gal', 'Type': 'Slider', 'Notes': 'Used as the base for inflated fuel price'},
    {'Assumption': 'Annual COGS Inflation', 'Value': f"{cogs_inflation * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to base COGS per kit starting in 2028'},
    {'Assumption': 'Base COGS per Kit (2028)', 'Value': f"{base_cogs:,.0f}", 'Units': '$/kit', 'Type': 'Slider', 'Notes': 'Used as the base for COGS inflation and 2027 inventory build'},
    {'Assumption': 'Fuel Savings % per Aircraft', 'Value': f"{fuel_saving_pct * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Percent of annual fuel spend saved'},
    {'Assumption': 'Fuel Savings Split to Tamarack', 'Value': f"{fuel_savings_split_to_tamarack * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Percent of annual fuel savings paid to Tamarack'},
    {'Assumption': 'Equity', 'Value': f"{cert_readiness_cost:.1f}", 'Units': '$M', 'Type': 'Slider', 'Notes': 'Used first to fund pre-2028 certification / inventory outflows'},
    {'Assumption': 'Debt Raised', 'Value': f"{debt_amount:.1f}", 'Units': '$M', 'Type': 'Slider', 'Notes': 'Drawn only if equity is exhausted pre-2028'},
    {'Assumption': 'Debt APR', 'Value': f"{debt_apr * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Applied to outstanding debt balance'},
    {'Assumption': 'Debt Term', 'Value': f"{debt_term_years}", 'Units': 'Years', 'Type': 'Slider', 'Notes': 'Debt amortizes annually beginning in 2028'},
    {'Assumption': 'Income Tax Rate', 'Value': f"{tax_rate * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Taxes apply only when taxable income is positive'},
    {'Assumption': 'WACC', 'Value': f"{wacc * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Used to discount unlevered free cash flows in DCF'},
    {'Assumption': 'Terminal Growth Rate', 'Value': f"{terminal_growth * 100:.2f}%", 'Units': '%', 'Type': 'Slider', 'Notes': 'Used for terminal value if WACC > terminal growth'},
    {'Assumption': 'Inventory Kits Before First Install', 'Value': f"{int(inventory_kits_pre_install)}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': 'Purchased in 2027 at base COGS per kit'},
    {'Assumption': 'Total Addressable Market', 'Value': f"{int(tam_shipsets)}", 'Units': 'Shipsets', 'Type': 'Slider', 'Notes': 'Caps cumulative shipsets'},
    {'Assumption': 'First-Year Install Rate (Q1)', 'Value': f"{int(q1_installs)}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': 'First install year (2028) quarterly installs'},
    {'Assumption': 'First-Year Install Rate (Q2)', 'Value': f"{int(q2_installs)}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': 'First install year (2028) quarterly installs'},
    {'Assumption': 'First-Year Install Rate (Q3)', 'Value': f"{int(q3_installs)}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': 'First install year (2028) quarterly installs'},
    {'Assumption': 'First-Year Install Rate (Q4)', 'Value': f"{int(q4_installs)}", 'Units': 'Kits', 'Type': 'Slider', 'Notes': 'First install year (2028) quarterly installs; Q4 and after stabilize'},
    {'Assumption': 'Block Hours per Aircraft per Year', 'Value': f"{int(block_hours)}", 'Units': 'Hours', 'Type': 'Hardwired', 'Notes': 'Used to compute annual fuel spend'},
    {'Assumption': 'Base Fuel Burn', 'Value': f"{int(base_fuel_burn_gal_per_hour)}", 'Units': 'Gal/hour', 'Type': 'Hardwired', 'Notes': 'Used to compute annual fuel spend'},
    {'Assumption': 'Model Years', 'Value': f"{years[0]}-{years[-1]}", 'Units': 'Years', 'Type': 'Hardwired', 'Notes': 'Annual model projection period'},
    {'Assumption': 'Certification Spend Timing', 'Value': '50% in 2026, 50% in 2027', 'Units': '', 'Type': 'Hardwired', 'Notes': 'Certification spend split evenly across 2026-2027'},
    {'Assumption': 'Install Ramp (2029 New Installs)', 'Value': '910', 'Units': 'Kits', 'Type': 'Hardwired', 'Notes': 'Fixed ramp year after first installs'},
    {'Assumption': 'Install Ramp (2030+ New Installs)', 'Value': '1040', 'Units': 'Kits', 'Type': 'Hardwired', 'Notes': 'Steady-state new installs from 2030 onward'},
    {'Assumption': 'OpEx Schedule', 'Value': ', '.join([f"{k}:{v}" for k, v in opex.items()]), 'Units': '$M/year', 'Type': 'Hardwired', 'Notes': 'OpEx by year; defaults to 15 after 2035'},
    {'Assumption': 'Taxes Floor', 'Value': 'Taxes = max(0, taxable income) * tax rate', 'Units': '', 'Type': 'Hardwired', 'Notes': 'No tax benefit modeled for losses'},
    {'Assumption': 'Terminal Value Condition', 'Value': 'Only computed if WACC > terminal growth', 'Units': '', 'Type': 'Hardwired', 'Notes': 'Otherwise terminal value treated as 0'},
]

assumptions_df = pd.DataFrame(assumptions_rows, columns=['Assumption', 'Value', 'Units', 'Type', 'Notes'])

# Calculations
data = {}
cum_shipsets = 0
cum_cash = 0
debt_balance = 0.0
debt_draw_remaining = float(debt_amount)
debt_drawn_total = 0.0
investor_cum_cf = 0.0
equity_cum_cf = 0.0
equity_reserve = float(cert_readiness_cost)

equity_amount = float(cert_readiness_cost)

if float(debt_apr) == 0:
    annual_debt_payment = (float(debt_amount) / float(debt_term_years)) if float(debt_term_years) > 0 else 0.0
else:
    annual_debt_payment = float(debt_amount) * float(debt_apr) / (1 - (1 + float(debt_apr)) ** (-float(debt_term_years)))

for yr in years:
    if yr < 2028:
        new_installs = 0
        revenue = 0
        cogs = 0
        inventory = 0 if yr == 2026 else inventory_kits_pre_install * base_cogs / 1e6
        capex = cert_2026 if yr == 2026 else cert_2027
    else:
        year_idx = yr - 2028
        fuel_price = base_fuel_price * (1 + fuel_inflation) ** year_idx
        annual_fuel_spend = block_hours * base_fuel_burn_gal_per_hour * fuel_price
        annual_saving = annual_fuel_spend * fuel_saving_pct
        rev_per_shipset = annual_saving * split_pct  # in $
        
        if yr == 2028:
            new_installs = q1_installs + q2_installs + q3_installs + q4_installs
        elif yr == 2029:
            new_installs = 910  # Fixed from previous ramp
        else:
            new_installs = 1040  # Steady state
        
        new_installs = min(new_installs, tam_shipsets - cum_shipsets)  # Cap at TAM
        cum_shipsets += new_installs
        revenue = cum_shipsets * rev_per_shipset / 1e6  # $M
        
        cogs_per_kit = base_cogs * (1 + cogs_inflation) ** year_idx
        cogs = new_installs * cogs_per_kit / 1e6  # $M
        
        capex = 0
        inventory = 0
    
    gross_profit = revenue - cogs
    opex_yr = opex.get(yr, 15)  # Flat after
    ebitda = gross_profit - opex_yr
    total_outflow = capex + inventory
    fcf = ebitda - total_outflow

    equity_contribution = 0.0
    debt_draw = 0.0
    if yr < 2028:
        equity_contribution = min(float(equity_reserve), float(total_outflow))
        equity_reserve -= equity_contribution
        remaining_outflow = float(total_outflow) - float(equity_contribution)
        if debt_draw_remaining > 0 and remaining_outflow > 0:
            debt_draw = min(float(debt_draw_remaining), float(remaining_outflow))
            debt_draw_remaining -= debt_draw
            debt_drawn_total += debt_draw

    debt_balance_beg = debt_balance
    debt_balance = debt_balance + debt_draw

    debt_interest = 0.0
    debt_principal = 0.0
    debt_payment = 0.0
    if yr >= 2028 and debt_balance > 0:
        debt_interest = debt_balance * float(debt_apr)
        debt_payment = min(annual_debt_payment, debt_balance + debt_interest)
        debt_principal = max(0.0, min(debt_balance, debt_payment - debt_interest))
        debt_balance = max(0.0, debt_balance - debt_principal)

    taxable_income = ebitda - debt_interest
    taxes = max(0.0, taxable_income) * float(tax_rate)
    fcf_after_tax = ebitda - taxes - total_outflow
    net_cash_after_debt = fcf_after_tax + debt_draw - debt_payment
    net_cash_change = net_cash_after_debt + equity_contribution
    cum_cash += net_cash_change

    investor_cf = (-debt_draw) + debt_payment
    investor_cum_cf += investor_cf
    investor_roi = (investor_cum_cf / float(debt_drawn_total)) if float(debt_drawn_total) > 0 else 0.0

    if yr < 2028:
        equity_cf = -equity_contribution
    else:
        equity_cf = net_cash_after_debt

    equity_cum_cf += equity_cf
    equity_roi = (equity_cum_cf / float(equity_amount)) if float(equity_amount) > 0 else 0.0
    
    data[yr] = {
        'New Installs': new_installs,
        'Cum Shipsets': cum_shipsets,
        'Revenue ($M)': round(revenue, 1),
        'COGS ($M)': round(cogs, 1),
        'Gross Profit ($M)': round(gross_profit, 1),
        'OpEx ($M)': opex_yr,
        'EBITDA ($M)': round(ebitda, 1),
        'CapEx/Inv ($M)': round(total_outflow, 1),
        'Free Cash Flow ($M)': round(fcf, 1),
        'Taxes ($M)': round(taxes, 1),
        'FCF After Tax ($M)': round(fcf_after_tax, 1),
        'Debt Draw ($M)': round(debt_draw, 1),
        'Debt Payment ($M)': round(debt_payment, 1),
        'Debt Interest ($M)': round(debt_interest, 1),
        'Debt Principal ($M)': round(debt_principal, 1),
        'Debt Balance ($M)': round(debt_balance, 1),
        'Net Cash After Debt ($M)': round(net_cash_after_debt, 1),
        'Net Cash Change ($M)': round(net_cash_change, 1),
        'Cumulative Cash ($M)': round(cum_cash, 1),
        'Debt Investor CF ($M)': round(investor_cf, 1),
        'Debt Investor Cum CF ($M)': round(investor_cum_cf, 1),
        'Debt Investor ROI (%)': round(investor_roi * 100, 1),
        'Equity Contribution ($M)': round(equity_contribution, 1),
        'Equity CF ($M)': round(equity_cf, 1),
        'Equity Cum CF ($M)': round(equity_cum_cf, 1),
        'Equity ROI (%)': round(equity_roi * 100, 1),
    }

df = pd.DataFrame(data).T

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

unlevered_taxes = (df['EBITDA ($M)'].clip(lower=0.0) * float(tax_rate)).round(1)
unlevered_fcf = (df['EBITDA ($M)'] - unlevered_taxes - df['CapEx/Inv ($M)']).round(1)
discount_year0 = int(df.index.min())
discount_t = (df.index - discount_year0 + 1).astype(int)
discount_factor = pd.Series((1 / (1 + float(wacc)) ** discount_t).astype(float), index=df.index)
pv_fcf = (unlevered_fcf.astype(float) * discount_factor).round(1)

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

pv_explicit = float(pv_fcf.sum())
enterprise_value = pv_explicit + (float(pv_tv) if not np.isnan(pv_tv) else 0.0)

dcf_summary_df = pd.DataFrame({
    'PV Explicit FCF ($M)': [round(pv_explicit, 1)],
    'Terminal Value ($M)': [round(0.0 if np.isnan(tv) else float(tv), 1)],
    'PV Terminal Value ($M)': [round(0.0 if np.isnan(pv_tv) else float(pv_tv), 1)],
    'Enterprise Value ($M)': [round(enterprise_value, 1)],
})

st.dataframe(df, use_container_width=True)

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
st.write(f"Enterprise value ($M): {enterprise_value:.1f}")
st.dataframe(dcf_summary_df, use_container_width=True)

# Nice Plots
st.header('Financial Projections Plots')

fig, ax = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
fig.suptitle('Tamarack A320 Financial Projections', fontsize=16)

# Revenue
ax[0].bar(df.index, df['Revenue ($M)'], color='orange', label='Revenue')
ax[0].set_title('Revenue Over Time')
ax[0].set_ylabel('$M')
ax[0].legend()
ax[0].grid(True, linestyle='--', alpha=0.7)

# Gross Profit
ax[1].bar(df.index, df['Gross Profit ($M)'], color='green', label='Gross Profit')
ax[1].set_title('Gross Profit Over Time')
ax[1].set_ylabel('$M')
ax[1].legend()
ax[1].grid(True, linestyle='--', alpha=0.7)

# Free Cash Flow
ax[2].bar(df.index, df['Free Cash Flow ($M)'], color='blue', label='Free Cash Flow')
ax[2].set_title('Free Cash Flow Over Time')
ax[2].set_ylabel('$M')
ax[2].set_xlabel('Year')
ax[2].legend()
ax[2].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
st.pyplot(fig)

# PDF Report Download
st.header('Generate PDF Report')

if st.button('Download PDF Report'):
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        # Page 1: Table
        fig_table = plt.figure(figsize=(17, 11))
        ax_table = fig_table.add_subplot(111)
        ax_table.axis('off')
        table = ax_table.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(6)
        table.auto_set_column_width(col=list(range(len(df.columns))))
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
        fig_table.suptitle('Financial Projections Table')
        pdf.savefig(fig_table, bbox_inches='tight')
        
        # Page 2: Plots
        pdf.savefig(fig, bbox_inches='tight')

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
    st.download_button(
        label="Download PDF",
        data=pdf_buffer,
        file_name="Tamarack_Financial_Report.pdf",
        mime="application/pdf"
    )

st.header('Assumptions Appendix')
st.dataframe(assumptions_df, use_container_width=True)

st.header('Enterprise Value')
st.write(f"Enterprise value ($M): {enterprise_value:.1f}")