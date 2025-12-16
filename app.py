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
cogs_inflation = st.sidebar.slider('Annual COGS Inflation (%)', min_value=0.0, max_value=15.0, value=4.0, step=0.5) / 100
fuel_saving_pct = st.sidebar.slider('Fuel Savings % per Aircraft', min_value=5.0, max_value=15.0, value=10.0, step=0.5) / 100
cert_readiness_cost = st.sidebar.slider('Total Cert & Production Readiness Cost ($M)', min_value=100.0, max_value=300.0, value=180.0, step=10.0)
inventory_kits_pre_install = st.sidebar.slider('Inventory Kits Before First Install', min_value=50, max_value=200, value=130, step=10)
tam_shipsets = st.sidebar.slider('Total Addressable Market (Max Shipsets in 10 Years)', min_value=1000, max_value=10000, value=7500, step=500)

# Install rates for first 4 quarters (first install year, e.g., 2028)
st.sidebar.header('First Year Install Rates (Kits per Quarter) - Q4 and After Stabilize')
q1_installs = st.sidebar.slider('Q1 Installs', min_value=0, max_value=200, value=98, step=10)  # ~10/week * 13 weeks / 4 = approx
q2_installs = st.sidebar.slider('Q2 Installs', min_value=0, max_value=200, value=98, step=10)
q3_installs = st.sidebar.slider('Q3 Installs', min_value=0, max_value=200, value=98, step=10)
q4_installs = st.sidebar.slider('Q4 Installs', min_value=0, max_value=200, value=96, step=10)  # Total ~390 for year

# Fixed assumptions (from previous)
base_fuel_price = 3.00  # $/gal in 2028
base_cogs = 400000  # $ per kit in 2028
block_hours = 2800
base_fuel_burn_gal_per_hour = 640
split_pct = 0.5

# Cert costs split (assuming even over 2026-2027)
cert_2026 = cert_readiness_cost / 2
cert_2027 = cert_readiness_cost / 2

# OpEx fixed for simplicity (lean case)
opex = {2026: 50, 2027: 40, 2028: 40, 2029: 35, 2030: 25, 2031: 20, 2032: 18, 2033: 15, 2034: 15, 2035: 15}

# Years
years = list(range(2026, 2036))  # 10 years

# Calculations
data = {}
cum_shipsets = 0
cum_cash = 0
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
    cum_cash += fcf
    
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
        'Cumulative Cash ($M)': round(cum_cash, 1),
    }

df = pd.DataFrame(data).T

st.dataframe(df, use_container_width=True)

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
        fig_table = plt.figure(figsize=(11, 8))
        ax_table = fig_table.add_subplot(111)
        ax_table.axis('off')
        table = ax_table.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.2, 1.2)
        fig_table.suptitle('Financial Projections Table')
        pdf.savefig(fig_table, bbox_inches='tight')
        
        # Page 2: Plots
        pdf.savefig(fig, bbox_inches='tight')
    
    pdf_buffer.seek(0)
    st.download_button(
        label="Download PDF",
        data=pdf_buffer,
        file_name="Tamarack_Financial_Report.pdf",
        mime="application/pdf"
    )