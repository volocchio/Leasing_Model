"""
Shared financial model engine for Tamarack Aerospace A320 Financial Model.

Both the standalone app (app.py) and the sensitivity study (app_sense.py)
call into this module so the quarterly simulation logic lives in one place.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Tuple


# ── Shared CSS fragments ────────────────────────────────────────────────────

METRIC_CARD_CSS = """
<style>
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
"""

RADIO_CSS = """
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
  white-space: nowrap;
}
div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
  border-color: #1D4ED8;
  background: #DBEAFE;
  box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.18);
}
</style>
"""

DOWNLOAD_BUTTON_CSS = """
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
"""

TABS_CSS = """
<style>
div[data-testid="stTabs"] div[data-baseweb="tab-list"],
div[data-testid="stTabs"] div[data-baseweb="tab-list"]:has(button[role="tab"]),
div[data-testid="stTabs"] div[data-baseweb="tab-list"]:has(button[data-baseweb="tab"]) {
  gap: 12px !important;
  padding: 10px 10px !important;
  background: #FFF1D6 !important;
  border: 3px solid #F97316 !important;
  border-radius: 14px !important;
}

div[data-testid="stTabs"] button[role="tab"],
div[data-testid="stTabs"] button[data-baseweb="tab"] {
  font-weight: 900 !important;
  font-size: 1.02rem !important;
  color: #7C2D12 !important;
  border: 2px solid #FDBA74 !important;
  border-radius: 12px !important;
  background: #FFFFFF !important;
  padding: 10px 16px !important;
}

div[data-testid="stTabs"] button[role="tab"]:hover,
div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
  border-color: #F97316 !important;
  background: #FFF7ED !important;
}

div[data-testid="stTabs"] button[role="tab"][aria-selected="true"],
div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
  background: #FFEDD5 !important;
  border-color: #9A3412 !important;
  transform: translateY(-1px);
  box-shadow:
    0 0 0 5px rgba(154, 52, 18, 0.18),
    inset 0 -4px 0 0 #9A3412;
}
</style>
"""


# ── Default OpEx schedule ────────────────────────────────────────────────────

DEFAULT_OPEX = {
    2026: 50, 2027: 40, 2028: 40, 2029: 35, 2030: 25,
    2031: 20, 2032: 18, 2033: 15, 2034: 15, 2035: 15,
}


# ── Derived-parameter helpers ────────────────────────────────────────────────

def compute_derived_params(cert_duration_years: float) -> Dict[str, Any]:
    """Return cert_duration_quarters, revenue_start_*, inventory_* keys."""
    cert_duration_quarters = max(1, int(round(cert_duration_years * 4.0)))
    revenue_start_q_index = cert_duration_quarters
    revenue_start_year = 2026 + (revenue_start_q_index // 4)
    revenue_start_quarter = (revenue_start_q_index % 4) + 1
    inventory_purchase_q_index = max(0, revenue_start_q_index - 1)
    inventory_year = 2026 + (inventory_purchase_q_index // 4)
    inventory_quarter = (inventory_purchase_q_index % 4) + 1
    return {
        "cert_duration_quarters": cert_duration_quarters,
        "revenue_start_q_index": revenue_start_q_index,
        "revenue_start_year": revenue_start_year,
        "revenue_start_quarter": revenue_start_quarter,
        "inventory_purchase_q_index": inventory_purchase_q_index,
        "inventory_year": inventory_year,
        "inventory_quarter": inventory_quarter,
    }


# ── Full quarterly simulation ────────────────────────────────────────────────

def run_model_full(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the full quarterly simulation and return *all* annual-level data
    needed by both the standalone dashboard and the sensitivity study.

    Returns a dict with keys:
        "annual_data"       – dict[year] -> row dict  (full P&L / CF / DCF columns)
        "enterprise_value"  – float
        "pv_explicit"       – float
        "pv_tv"             – float (may be NaN)
        "tv"                – float (may be NaN)
        "equity_cum_cf"     – float
        "equity_roi"        – float
        "investor_cum_cf"   – float
        "investor_roi"      – float
        "cum_cash"          – float
    """
    years = list(range(2026, 2036))

    model_type = str(params.get("model_type", "Leasing (Split Savings)"))

    revenue_start_q_index = int(params["revenue_start_q_index"])
    inventory_purchase_q_index = int(params["inventory_purchase_q_index"])
    revenue_start_year = int(params["revenue_start_year"])

    fuel_inflation = params["fuel_inflation"]
    cogs_inflation = params["cogs_inflation"]
    base_fuel_price = params["base_fuel_price"]
    base_cogs = params["base_cogs"]
    block_hours = params["block_hours"]
    base_fuel_burn_gal_per_hour = params["base_fuel_burn_gal_per_hour"]
    fuel_saving_pct = params["fuel_saving_pct"]
    split_pct = params["fuel_savings_split_to_tamarack"]
    target_payback_years = params.get("target_payback_years", 2.5)

    corsia_split = params.get("corsia_split", 0.0)
    carbon_price = params.get("carbon_price", 0.0)
    if model_type == "Kit Sale (Payback Pricing)":
        corsia_split = 0.0
        carbon_price = 0.0

    overhaul_extension_pct = params.get("overhaul_extension_pct", 0.10)
    shop_visit_cost_m = params.get("shop_visit_cost_m", 6.0)
    base_tbo_hours = params.get("base_tbo_hours", 18000)
    overhaul_split = params.get("overhaul_split", 0.50)

    inventory_kits_pre_install = int(params["inventory_kits_pre_install"])
    tam_shipsets = int(params["tam_shipsets"])
    tam_penetration_pct = params.get("tam_penetration_pct", 1.0)

    fleet_retirements_per_month = params.get("fleet_retirements_per_month", 0.0)
    include_forward_fit = bool(params.get("include_forward_fit", False))
    forward_fit_per_month = params.get("forward_fit_per_month", 0.0)

    q1_installs = int(params["q1_installs"])
    q2_installs = int(params["q2_installs"])
    q3_installs = int(params["q3_installs"])
    q4_installs = int(params["q4_installs"])
    year2_installs = int(params.get("year2_installs", 910))
    year3_installs = int(params.get("year3_installs", 1040))

    cert_readiness_cost = params["cert_readiness_cost"]
    cert_duration_quarters = int(params["cert_duration_quarters"])
    cert_spend_per_quarter = (cert_readiness_cost / cert_duration_quarters) if cert_duration_quarters > 0 else 0.0

    debt_amount = params["debt_amount"]
    debt_apr = params["debt_apr"]
    debt_term_years = int(params["debt_term_years"])

    tax_rate = params["tax_rate"]
    wacc = params["wacc"]
    terminal_growth = params["terminal_growth"]

    opex = dict(params.get("opex", DEFAULT_OPEX))

    # ── State variables ──────────────────────────────────────────────────
    annual_data: Dict[int, Dict[str, Any]] = {}

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

    debt_rate_q = debt_apr / 4.0
    term_quarters = debt_term_years * 4
    quarterly_debt_payment = None

    year_sums: Dict[str, float] | None = None
    year_taxable_income = 0.0

    SUM_KEYS = [
        "New Installs", "Kit Price ($/kit) Sum", "Kit Price Qtrs",
        "Revenue ($M)", "COGS ($M)", "Gross Profit ($M)", "OpEx ($M)",
        "EBITDA ($M)", "CapEx/Inv ($M)", "Free Cash Flow ($M)",
        "Taxes ($M)", "FCF After Tax ($M)", "Debt Draw ($M)",
        "Debt Payment ($M)", "Debt Interest ($M)", "Debt Principal ($M)",
        "Net Cash After Debt ($M)", "Net Cash Change ($M)",
        "Debt Investor CF ($M)", "Equity Contribution ($M)", "Equity CF ($M)",
    ]

    for i in range(len(years) * 4):
        yr = years[0] + (i // 4)
        qtr = (i % 4) + 1

        if year_sums is None:
            year_sums = {k: 0.0 for k in SUM_KEYS}
            year_taxable_income = 0.0

        capex = 0.0
        inventory = 0.0

        fleet_beg = fleet_size
        installed_beg = installed_base

        retire_q = fleet_retirements_per_month * 3.0
        retire_q = min(retire_q, fleet_beg) if fleet_beg > 0 else 0.0
        forward_fit_q = (forward_fit_per_month * 3.0) if include_forward_fit else 0.0

        fleet_size = max(0.0, fleet_beg - retire_q + forward_fit_q)
        installed_base = installed_beg

        installable_cap = fleet_size * tam_penetration_pct

        if i < revenue_start_q_index:
            new_installs = 0.0
            revenue = 0.0
            cogs = 0.0
            kit_price = np.nan

            if i < cert_duration_quarters:
                capex = cert_spend_per_quarter
            if i == inventory_purchase_q_index:
                inventory = 0.25 * inventory_kits_pre_install * base_cogs / 1e6
        else:
            year_idx = yr - revenue_start_year
            fuel_price = base_fuel_price * ((1 + fuel_inflation) ** year_idx)
            quarter_block_hours = block_hours / 4.0
            quarter_fuel_spend = quarter_block_hours * base_fuel_burn_gal_per_hour * fuel_price

            quarter_saving = quarter_fuel_spend * fuel_saving_pct
            quarter_gallons_burn = quarter_block_hours * base_fuel_burn_gal_per_hour
            gallons_saved = quarter_gallons_burn * fuel_saving_pct
            fuel_saved_tonnes = gallons_saved * 0.00304
            co2_avoided_t = fuel_saved_tonnes * 3.16
            corsia_value = 0.0 if model_type == "Kit Sale (Payback Pricing)" else (co2_avoided_t * corsia_split * carbon_price)

            base_overhaul_cost_per_hour = (shop_visit_cost_m * 1e6 * 2) / base_tbo_hours
            hourly_overhaul_savings = base_overhaul_cost_per_hour * overhaul_extension_pct
            quarter_overhaul_value = quarter_block_hours * hourly_overhaul_savings
            overhaul_value = quarter_overhaul_value * overhaul_split

            total_value_created = quarter_saving + corsia_value + overhaul_value

            rev_q_idx = i - revenue_start_q_index
            planned_installs = 0.0
            if rev_q_idx == 0:
                planned_installs = q1_installs
            elif rev_q_idx == 1:
                planned_installs = q2_installs
            elif rev_q_idx == 2:
                planned_installs = q3_installs
            elif rev_q_idx == 3:
                planned_installs = q4_installs
            else:
                revenue_year = rev_q_idx // 4
                if revenue_year == 1:
                    planned_installs = year2_installs / 4.0
                else:
                    planned_installs = year3_installs / 4.0

            remaining_capacity = max(0.0, installable_cap - installed_base)
            new_installs = min(planned_installs, remaining_capacity)
            installed_base += new_installs

            kit_price = np.nan

            if model_type == "Kit Sale (Payback Pricing)":
                annual_value_created = total_value_created * 4.0
                rev_per_kit = annual_value_created * target_payback_years
                kit_price = rev_per_kit
                revenue = new_installs * rev_per_kit / 1e6
            else:
                rev_per_shipset = total_value_created * split_pct
                avg_installed = installed_base - 0.5 * new_installs
                revenue = avg_installed * rev_per_shipset / 1e6

            cogs_per_kit = base_cogs * ((1 + cogs_inflation) ** year_idx)
            cogs = new_installs * cogs_per_kit / 1e6

        gross_profit = revenue - cogs
        opex_q = opex.get(yr, 15) / 4.0
        ebitda = gross_profit - opex_q
        total_outflow = capex + inventory
        fcf = ebitda - total_outflow

        equity_contribution = 0.0
        debt_draw = 0.0
        if i < revenue_start_q_index:
            equity_contribution = min(equity_reserve, total_outflow)
            equity_reserve -= equity_contribution
            remaining_outflow = total_outflow - equity_contribution
            if debt_draw_remaining > 0 and remaining_outflow > 0:
                debt_draw = min(debt_draw_remaining, remaining_outflow)
                debt_draw_remaining -= debt_draw
                debt_drawn_total += debt_draw

        debt_balance += debt_draw

        debt_interest = 0.0
        debt_principal = 0.0
        debt_payment = 0.0
        if i >= revenue_start_q_index and debt_balance > 0:
            if quarterly_debt_payment is None:
                if debt_rate_q == 0:
                    quarterly_debt_payment = (debt_balance / term_quarters) if term_quarters > 0 else 0.0
                else:
                    quarterly_debt_payment = debt_balance * debt_rate_q / (1 - (1 + debt_rate_q) ** (-term_quarters))

            debt_interest = debt_balance * debt_rate_q
            debt_payment = min(quarterly_debt_payment, debt_balance + debt_interest)
            debt_principal = max(0.0, min(debt_balance, debt_payment - debt_interest))
            debt_balance = max(0.0, debt_balance - debt_principal)

        taxable_income_q = ebitda - debt_interest
        year_taxable_income += taxable_income_q

        taxes = 0.0
        if qtr == 4:
            taxes = max(0.0, year_taxable_income) * tax_rate

        fcf_after_tax = ebitda - taxes - total_outflow
        net_cash_after_debt = fcf_after_tax + debt_draw - debt_payment
        net_cash_change = net_cash_after_debt + equity_contribution
        cum_cash += net_cash_change

        investor_cf = -debt_draw + debt_payment
        investor_cum_cf += investor_cf
        investor_roi = (investor_cum_cf / debt_drawn_total) if debt_drawn_total > 0 else 0.0

        if i < revenue_start_q_index:
            equity_cf = -equity_contribution
        else:
            equity_cf = net_cash_after_debt

        equity_cum_cf += equity_cf
        equity_roi = (equity_cum_cf / equity_amount) if equity_amount > 0 else 0.0

        # Accumulate into year sums
        year_sums["New Installs"] += new_installs
        if not np.isnan(kit_price):
            year_sums["Kit Price ($/kit) Sum"] += kit_price
            year_sums["Kit Price Qtrs"] += 1.0
        year_sums["Revenue ($M)"] += revenue
        year_sums["COGS ($M)"] += cogs
        year_sums["Gross Profit ($M)"] += gross_profit
        year_sums["OpEx ($M)"] += opex_q
        year_sums["EBITDA ($M)"] += ebitda
        year_sums["CapEx/Inv ($M)"] += total_outflow
        year_sums["Free Cash Flow ($M)"] += fcf
        year_sums["Taxes ($M)"] += taxes
        year_sums["FCF After Tax ($M)"] += fcf_after_tax
        year_sums["Debt Draw ($M)"] += debt_draw
        year_sums["Debt Payment ($M)"] += debt_payment
        year_sums["Debt Interest ($M)"] += debt_interest
        year_sums["Debt Principal ($M)"] += debt_principal
        year_sums["Net Cash After Debt ($M)"] += net_cash_after_debt
        year_sums["Net Cash Change ($M)"] += net_cash_change
        year_sums["Debt Investor CF ($M)"] += investor_cf
        year_sums["Equity Contribution ($M)"] += equity_contribution
        year_sums["Equity CF ($M)"] += equity_cf

        if qtr == 4:
            avg_kit_price = (
                year_sums["Kit Price ($/kit) Sum"] / year_sums["Kit Price Qtrs"]
            ) if year_sums["Kit Price Qtrs"] > 0 else np.nan

            annual_data[yr] = {
                "New Installs": int(round(year_sums["New Installs"])),
                "Cum Shipsets": int(round(installed_base)),
                "Fleet Size": int(round(fleet_size)),
                "Kit Price ($/kit)": round(avg_kit_price, 0) if not np.isnan(avg_kit_price) else np.nan,
                "Revenue ($M)": round(year_sums["Revenue ($M)"], 1),
                "COGS ($M)": round(year_sums["COGS ($M)"], 1),
                "Gross Profit ($M)": round(year_sums["Gross Profit ($M)"], 1),
                "OpEx ($M)": round(year_sums["OpEx ($M)"], 1),
                "EBITDA ($M)": round(year_sums["EBITDA ($M)"], 1),
                "CapEx/Inv ($M)": round(year_sums["CapEx/Inv ($M)"], 1),
                "Free Cash Flow ($M)": round(year_sums["Free Cash Flow ($M)"], 1),
                "Taxes ($M)": round(year_sums["Taxes ($M)"], 1),
                "FCF After Tax ($M)": round(year_sums["FCF After Tax ($M)"], 1),
                "Debt Draw ($M)": round(year_sums["Debt Draw ($M)"], 1),
                "Debt Payment ($M)": round(year_sums["Debt Payment ($M)"], 1),
                "Debt Interest ($M)": round(year_sums["Debt Interest ($M)"], 1),
                "Debt Principal ($M)": round(year_sums["Debt Principal ($M)"], 1),
                "Debt Balance ($M)": round(debt_balance, 1),
                "Net Cash After Debt ($M)": round(year_sums["Net Cash After Debt ($M)"], 1),
                "Net Cash Change ($M)": round(year_sums["Net Cash Change ($M)"], 1),
                "Cumulative Cash ($M)": round(cum_cash, 1),
                "Debt Investor CF ($M)": round(year_sums["Debt Investor CF ($M)"], 1),
                "Debt Investor Cum CF ($M)": round(investor_cum_cf, 1),
                "Debt Investor ROI (%)": round(investor_roi * 100, 1),
                "Equity Contribution ($M)": round(year_sums["Equity Contribution ($M)"], 1),
                "Equity CF ($M)": round(year_sums["Equity CF ($M)"], 1),
                "Equity Cum CF ($M)": round(equity_cum_cf, 1),
                "Equity ROI (%)": round(equity_roi * 100, 1),
            }
            year_sums = None

    # ── DCF valuation ────────────────────────────────────────────────────
    df = pd.DataFrame(annual_data).T

    unlevered_taxes_raw = (df["EBITDA ($M)"].clip(lower=0.0) * tax_rate).astype(float)
    unlevered_fcf_raw = (df["EBITDA ($M)"] - unlevered_taxes_raw - df["CapEx/Inv ($M)"]).astype(float)
    discount_year0 = int(df.index.min())
    discount_t = (df.index - discount_year0 + 1).astype(int)
    discount_factor = pd.Series((1 / (1 + wacc) ** discount_t).astype(float), index=df.index)
    pv_fcf_raw = (unlevered_fcf_raw * discount_factor).astype(float)

    tv = np.nan
    pv_tv = np.nan
    if wacc > terminal_growth:
        tv = float(unlevered_fcf_raw.iloc[-1]) * (1 + terminal_growth) / (wacc - terminal_growth)
        pv_tv = tv * float(discount_factor.iloc[-1])

    pv_explicit = float(pv_fcf_raw.sum())
    enterprise_value = pv_explicit + (pv_tv if not np.isnan(pv_tv) else 0.0)

    return {
        "annual_data": annual_data,
        "df": df,
        "enterprise_value": enterprise_value,
        "pv_explicit": pv_explicit,
        "pv_tv": pv_tv,
        "tv": tv,
        "unlevered_fcf": unlevered_fcf_raw.round(1),
        "unlevered_taxes": unlevered_taxes_raw.round(1),
        "discount_factor": discount_factor.round(4),
        "pv_fcf": pv_fcf_raw.round(1),
        "equity_cum_cf": equity_cum_cf,
        "equity_roi": (equity_cum_cf / equity_amount) if equity_amount > 0 else 0.0,
        "investor_cum_cf": investor_cum_cf,
        "investor_roi": (investor_cum_cf / debt_drawn_total) if debt_drawn_total > 0 else 0.0,
        "cum_cash": cum_cash,
    }


def run_model_summary(params: Dict[str, Any]) -> Dict[str, float]:
    """
    Thin wrapper returning only the summary metrics used by the sensitivity study.
    """
    result = run_model_full(params)
    return {
        "Enterprise Value ($M)": result["enterprise_value"],
        "PV Explicit FCF ($M)": result["pv_explicit"],
        "PV Terminal Value ($M)": 0.0 if np.isnan(result["pv_tv"]) else result["pv_tv"],
        "Equity ROI (%)": result["equity_roi"] * 100.0,
        "Debt Investor ROI (%)": result["investor_roi"] * 100.0,
        "Ending Cash ($M)": result["cum_cash"],
    }
