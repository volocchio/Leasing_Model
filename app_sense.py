import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class DriverSpec:
    key: str
    label: str
    unit: str
    kind: str  # float | int | percent


def _to_internal_value(kind: str, v: float) -> float:
    if kind == "percent":
        return float(v) / 100.0
    return float(v)


def _to_display_value(kind: str, v: float) -> float:
    if kind == "percent":
        return float(v) * 100.0
    return float(v)


def run_model(params: Dict[str, Any]) -> Dict[str, float]:
    years = list(range(2026, 2036))

    revenue_start_year = int(params["revenue_start_year"])
    inventory_year = int(params["inventory_year"])

    fuel_inflation = float(params["fuel_inflation"])
    cogs_inflation = float(params["cogs_inflation"])

    base_fuel_price = float(params["base_fuel_price"])
    base_cogs = float(params["base_cogs"])

    block_hours = float(params["block_hours"])
    base_fuel_burn_gal_per_hour = float(params["base_fuel_burn_gal_per_hour"])

    fuel_saving_pct = float(params["fuel_saving_pct"])
    split_pct = float(params["fuel_savings_split_to_tamarack"])

    inventory_kits_pre_install = int(params["inventory_kits_pre_install"])
    tam_shipsets = int(params["tam_shipsets"])

    q1_installs = int(params["q1_installs"])
    q2_installs = int(params["q2_installs"])
    q3_installs = int(params["q3_installs"])
    q4_installs = int(params["q4_installs"])

    cert_readiness_cost = float(params["cert_readiness_cost"])
    cert_spend_by_year = dict(params["cert_spend_by_year"])

    debt_amount = float(params["debt_amount"])
    debt_apr = float(params["debt_apr"])
    debt_term_years = int(params["debt_term_years"])

    tax_rate = float(params["tax_rate"])
    wacc = float(params["wacc"])
    terminal_growth = float(params["terminal_growth"])

    opex = dict(params["opex"])

    cum_shipsets = 0
    cum_cash = 0.0

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

    rows = []

    for yr in years:
        if yr < revenue_start_year:
            new_installs = 0
            revenue = 0.0
            cogs = 0.0
            inventory = (inventory_kits_pre_install * base_cogs / 1e6) if yr == inventory_year else 0.0
            capex = float(cert_spend_by_year.get(yr, 0.0))
        else:
            year_idx = yr - revenue_start_year
            fuel_price = base_fuel_price * (1 + fuel_inflation) ** year_idx
            annual_fuel_spend = block_hours * base_fuel_burn_gal_per_hour * fuel_price
            annual_saving = annual_fuel_spend * fuel_saving_pct
            rev_per_shipset = annual_saving * split_pct

            if yr == revenue_start_year:
                new_installs = q1_installs + q2_installs + q3_installs + q4_installs
            elif yr == (revenue_start_year + 1):
                new_installs = 910
            else:
                new_installs = 1040

            new_installs = min(new_installs, tam_shipsets - cum_shipsets)
            cum_shipsets += new_installs

            revenue = cum_shipsets * rev_per_shipset / 1e6

            cogs_per_kit = base_cogs * (1 + cogs_inflation) ** year_idx
            cogs = new_installs * cogs_per_kit / 1e6

            capex = 0.0
            inventory = 0.0

        gross_profit = revenue - cogs
        opex_yr = float(opex.get(yr, 15))
        ebitda = gross_profit - opex_yr
        total_outflow = capex + inventory

        equity_contribution = 0.0
        debt_draw = 0.0
        if yr < revenue_start_year:
            equity_contribution = min(float(equity_reserve), float(total_outflow))
            equity_reserve -= equity_contribution
            remaining_outflow = float(total_outflow) - float(equity_contribution)
            if debt_draw_remaining > 0 and remaining_outflow > 0:
                debt_draw = min(float(debt_draw_remaining), float(remaining_outflow))
                debt_draw_remaining -= debt_draw
                debt_drawn_total += debt_draw

        debt_balance = debt_balance + debt_draw

        debt_interest = 0.0
        debt_principal = 0.0
        debt_payment = 0.0
        if yr >= revenue_start_year and debt_balance > 0:
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

        if yr < revenue_start_year:
            equity_cf = -equity_contribution
        else:
            equity_cf = net_cash_after_debt

        equity_cum_cf += equity_cf

        rows.append(
            {
                "Year": yr,
                "EBITDA": float(ebitda),
                "CapExInv": float(total_outflow),
                "Taxes": float(taxes),
                "UnleveredFCF": float(ebitda - max(0.0, ebitda) * float(tax_rate) - float(total_outflow)),
            }
        )

    df = pd.DataFrame(rows).set_index("Year")

    unlevered_fcf = df["UnleveredFCF"].astype(float)

    discount_year0 = int(df.index.min())
    discount_t = (df.index - discount_year0 + 1).astype(int)
    discount_factor = pd.Series((1 / (1 + float(wacc)) ** discount_t).astype(float), index=df.index)

    pv_fcf = (unlevered_fcf * discount_factor).astype(float)

    tv = np.nan
    pv_tv = np.nan
    if float(wacc) > float(terminal_growth):
        tv = float(unlevered_fcf.iloc[-1]) * (1 + float(terminal_growth)) / (float(wacc) - float(terminal_growth))
        pv_tv = tv * float(discount_factor.iloc[-1])

    pv_explicit = float(pv_fcf.sum())
    enterprise_value = pv_explicit + (float(pv_tv) if not np.isnan(pv_tv) else 0.0)

    equity_roi = (equity_cum_cf / float(equity_amount)) if float(equity_amount) > 0 else 0.0
    investor_roi = (investor_cum_cf / float(debt_drawn_total)) if float(debt_drawn_total) > 0 else 0.0

    return {
        "Enterprise Value ($M)": float(enterprise_value),
        "PV Explicit FCF ($M)": float(pv_explicit),
        "PV Terminal Value ($M)": float(0.0 if np.isnan(pv_tv) else float(pv_tv)),
        "Equity ROI (%)": float(equity_roi * 100.0),
        "Debt Investor ROI (%)": float(investor_roi * 100.0),
        "Ending Cash ($M)": float(cum_cash),
    }


def build_baseline_params() -> Dict[str, Any]:
    st.sidebar.header("Baseline Inputs")

    fuel_inflation = st.sidebar.slider("Annual Fuel Inflation (%)", min_value=0.0, max_value=15.0, value=4.5, step=0.5) / 100
    base_fuel_price = st.sidebar.slider("Base Fuel Price at First Revenue Year ($/gal)", min_value=1.0, max_value=6.0, value=2.75, step=0.1)
    block_hours = st.sidebar.slider("Block Hours per Aircraft per Year", min_value=1000, max_value=5000, value=3200, step=100)
    base_fuel_burn_gal_per_hour = st.sidebar.slider("Base Fuel Burn (gal/hour)", min_value=700, max_value=1200, value=750, step=50)

    cogs_inflation = st.sidebar.slider("Annual COGS Inflation (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.5) / 100
    base_cogs = st.sidebar.slider("Base COGS per Kit at First Revenue Year ($)", min_value=100000, max_value=800000, value=400000, step=10000)

    fuel_saving_pct = st.sidebar.slider("Fuel Savings % per Aircraft", min_value=5.0, max_value=15.0, value=10.0, step=0.5) / 100
    fuel_savings_split_to_tamarack = st.sidebar.slider("Fuel Savings Split to Tamarack (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0) / 100

    cert_readiness_cost = st.sidebar.slider("Equity ($M)", min_value=100.0, max_value=300.0, value=180.0, step=10.0)
    cert_duration_years = st.sidebar.slider("Certification Duration (Years)", min_value=0.25, max_value=5.0, value=2.0, step=0.25)
    cert_duration_quarters = max(1, int(round(float(cert_duration_years) * 4.0)))

    inventory_kits_pre_install = st.sidebar.slider("Inventory Kits Before First Install", min_value=50, max_value=200, value=90, step=10)
    tam_shipsets = st.sidebar.slider("Total Addressable Market (Max Shipsets in 10 Years)", min_value=1000, max_value=10000, value=7500, step=500)

    debt_amount = st.sidebar.slider("Debt Raised ($M)", min_value=0.0, max_value=500.0, value=float(cert_readiness_cost), step=10.0)
    debt_apr = st.sidebar.slider("Debt APR (%)", min_value=0.0, max_value=20.0, value=10.0, step=0.5) / 100
    debt_term_years = st.sidebar.slider("Debt Term (Years)", min_value=1, max_value=15, value=7, step=1)

    tax_rate = st.sidebar.slider("Income Tax Rate (%)", min_value=0.0, max_value=40.0, value=21.0, step=0.5) / 100
    wacc = st.sidebar.slider("WACC (%)", min_value=0.0, max_value=30.0, value=11.5, step=0.5) / 100
    terminal_growth = st.sidebar.slider("Terminal Growth Rate (%)", min_value=-2.0, max_value=8.0, value=3.0, step=0.5) / 100

    st.sidebar.header("Installs")
    q1_installs = st.sidebar.slider("Q1 Installs", min_value=0, max_value=200, value=98, step=10)
    q2_installs = st.sidebar.slider("Q2 Installs", min_value=0, max_value=200, value=98, step=10)
    q3_installs = st.sidebar.slider("Q3 Installs", min_value=0, max_value=200, value=98, step=10)
    q4_installs = st.sidebar.slider("Q4 Installs and beyond", min_value=0, max_value=200, value=96, step=10)

    cert_spend_by_year: Dict[int, float] = {}
    cert_spend_per_quarter = (float(cert_readiness_cost) / float(cert_duration_quarters)) if int(cert_duration_quarters) > 0 else 0.0
    for q in range(int(cert_duration_quarters)):
        yr = 2026 + (q // 4)
        cert_spend_by_year[yr] = cert_spend_by_year.get(yr, 0.0) + cert_spend_per_quarter

    revenue_start_year = 2026 + int(np.ceil(float(cert_duration_quarters) / 4.0))
    inventory_year = max(2026, revenue_start_year - 1)

    opex = {2026: 50, 2027: 40, 2028: 40, 2029: 35, 2030: 25, 2031: 20, 2032: 18, 2033: 15, 2034: 15, 2035: 15}

    return {
        "fuel_inflation": float(fuel_inflation),
        "base_fuel_price": float(base_fuel_price),
        "block_hours": float(block_hours),
        "base_fuel_burn_gal_per_hour": float(base_fuel_burn_gal_per_hour),
        "cogs_inflation": float(cogs_inflation),
        "base_cogs": float(base_cogs),
        "fuel_saving_pct": float(fuel_saving_pct),
        "fuel_savings_split_to_tamarack": float(fuel_savings_split_to_tamarack),
        "cert_readiness_cost": float(cert_readiness_cost),
        "cert_duration_years": float(cert_duration_years),
        "cert_duration_quarters": int(cert_duration_quarters),
        "cert_spend_by_year": cert_spend_by_year,
        "revenue_start_year": int(revenue_start_year),
        "inventory_year": int(inventory_year),
        "inventory_kits_pre_install": int(inventory_kits_pre_install),
        "tam_shipsets": int(tam_shipsets),
        "debt_amount": float(debt_amount),
        "debt_apr": float(debt_apr),
        "debt_term_years": int(debt_term_years),
        "tax_rate": float(tax_rate),
        "wacc": float(wacc),
        "terminal_growth": float(terminal_growth),
        "q1_installs": int(q1_installs),
        "q2_installs": int(q2_installs),
        "q3_installs": int(q3_installs),
        "q4_installs": int(q4_installs),
        "opex": opex,
    }


def build_driver_catalog(baseline: Dict[str, Any]) -> List[DriverSpec]:
    return [
        DriverSpec("fuel_inflation", "Annual Fuel Inflation", "%", "percent"),
        DriverSpec("base_fuel_price", "Base Fuel Price (First Revenue Year)", "$/gal", "float"),
        DriverSpec("block_hours", "Block Hours per Aircraft per Year", "hours", "int"),
        DriverSpec("base_fuel_burn_gal_per_hour", "Base Fuel Burn", "gal/hour", "int"),
        DriverSpec("cogs_inflation", "Annual COGS Inflation", "%", "percent"),
        DriverSpec("base_cogs", "Base COGS per Kit (First Revenue Year)", "$/kit", "int"),
        DriverSpec("fuel_saving_pct", "Fuel Savings % per Aircraft", "%", "percent"),
        DriverSpec("fuel_savings_split_to_tamarack", "Fuel Savings Split to Tamarack", "%", "percent"),
        DriverSpec("cert_readiness_cost", "Equity", "$M", "float"),
        DriverSpec("cert_duration_years", "Certification Duration", "years", "float"),
        DriverSpec("inventory_kits_pre_install", "Inventory Kits Before First Install", "kits", "int"),
        DriverSpec("tam_shipsets", "Total Addressable Market", "shipsets", "int"),
        DriverSpec("debt_amount", "Debt Raised", "$M", "float"),
        DriverSpec("debt_apr", "Debt APR", "%", "percent"),
        DriverSpec("debt_term_years", "Debt Term", "years", "int"),
        DriverSpec("tax_rate", "Income Tax Rate", "%", "percent"),
        DriverSpec("wacc", "WACC", "%", "percent"),
        DriverSpec("terminal_growth", "Terminal Growth Rate", "%", "percent"),
    ]


def _grid_values(spec: DriverSpec, low_disp: float, high_disp: float, points: int) -> List[float]:
    if points < 2:
        points = 2

    if spec.kind in {"int"}:
        vals = np.linspace(float(low_disp), float(high_disp), int(points))
        vals = np.unique(np.round(vals).astype(int)).tolist()
        return [float(v) for v in vals]

    if spec.key == "cert_duration_years":
        vals = np.linspace(float(low_disp), float(high_disp), int(points))
        vals = np.unique(np.round(vals * 4.0) / 4.0).tolist()
        return [float(v) for v in vals]

    vals = np.linspace(float(low_disp), float(high_disp), int(points)).tolist()
    return [float(v) for v in vals]


def _apply_driver_value(params: Dict[str, Any], spec: DriverSpec, disp_value: float) -> None:
    internal = _to_internal_value(spec.kind, disp_value)
    params[spec.key] = internal

    if spec.key == "cert_duration_years":
        qtrs = max(1, int(round(float(disp_value) * 4.0)))
        params["cert_duration_years"] = float(disp_value)
        params["cert_duration_quarters"] = int(qtrs)

        cert_readiness_cost = float(params["cert_readiness_cost"])
        cert_spend_by_year: Dict[int, float] = {}
        cert_spend_per_quarter = (float(cert_readiness_cost) / float(qtrs)) if int(qtrs) > 0 else 0.0
        for q in range(int(qtrs)):
            yr = 2026 + (q // 4)
            cert_spend_by_year[yr] = cert_spend_by_year.get(yr, 0.0) + cert_spend_per_quarter
        revenue_start_year = 2026 + int(np.ceil(float(qtrs) / 4.0))
        inventory_year = max(2026, revenue_start_year - 1)

        params["cert_spend_by_year"] = cert_spend_by_year
        params["revenue_start_year"] = int(revenue_start_year)
        params["inventory_year"] = int(inventory_year)


def run_sensitivity(
    baseline: Dict[str, Any],
    d1: DriverSpec,
    d2: DriverSpec,
    d3: DriverSpec,
    d1_vals: List[float],
    d2_vals: List[float],
    d3_vals: List[float],
    metric: str,
) -> pd.DataFrame:
    rows = []

    for v3, v1, v2 in itertools.product(d3_vals, d1_vals, d2_vals):
        params = dict(baseline)
        _apply_driver_value(params, d1, v1)
        _apply_driver_value(params, d2, v2)
        _apply_driver_value(params, d3, v3)

        outputs = run_model(params)

        rows.append(
            {
                d1.label: v1,
                d2.label: v2,
                d3.label: v3,
                "Metric": metric,
                "Value": float(outputs[metric]),
            }
        )

    return pd.DataFrame(rows)


def plot_heatmap(pivot: pd.DataFrame, x_label: str, y_label: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    z = pivot.values.astype(float)
    im = ax.imshow(z, aspect="auto", origin="lower")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(v) for v in pivot.columns.tolist()], rotation=45, ha="right")

    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(v) for v in pivot.index.tolist()])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")

    fig.tight_layout()
    return fig


st.set_page_config(layout="wide")
st.title("Tamarack Aerospace – 3-Driver Sensitivity Study")

baseline_params = build_baseline_params()

st.header("Baseline Outputs")
baseline_outputs = run_model(baseline_params)
st.dataframe(pd.DataFrame([baseline_outputs]).T.rename(columns={0: "Baseline"}), use_container_width=True)

st.header("Sensitivity Study")

drivers = build_driver_catalog(baseline_params)
driver_by_key = {d.key: d for d in drivers}
driver_labels = {d.key: d.label for d in drivers}

col_a, col_b, col_c = st.columns(3)
with col_a:
    d1_key = st.selectbox("Driver 1", options=[d.key for d in drivers], format_func=lambda k: driver_labels[k], index=0)
with col_b:
    d2_key = st.selectbox(
        "Driver 2",
        options=[d.key for d in drivers if d.key != d1_key],
        format_func=lambda k: driver_labels[k],
        index=0,
    )
with col_c:
    d3_key = st.selectbox(
        "Driver 3",
        options=[d.key for d in drivers if d.key not in {d1_key, d2_key}],
        format_func=lambda k: driver_labels[k],
        index=0,
    )

d1 = driver_by_key[d1_key]
d2 = driver_by_key[d2_key]
d3 = driver_by_key[d3_key]

metrics = list(baseline_outputs.keys())
metric = st.selectbox("Metric to Sensitize", options=metrics, index=0)

b1 = _to_display_value(d1.kind, float(baseline_params[d1.key]))
b2 = _to_display_value(d2.kind, float(baseline_params[d2.key]))
b3 = _to_display_value(d3.kind, float(baseline_params[d3.key]))

cfg1, cfg2, cfg3 = st.columns(3)
with cfg1:
    st.subheader("Driver 1 Range")
    d1_low = st.number_input("Low", value=float(b1) * 0.9, key="d1_low")
    d1_high = st.number_input("High", value=float(b1) * 1.1 if float(b1) != 0 else 1.0, key="d1_high")
    d1_points = st.number_input("Points", min_value=2, max_value=25, value=5, step=1, key="d1_points")
with cfg2:
    st.subheader("Driver 2 Range")
    d2_low = st.number_input("Low ", value=float(b2) * 0.9, key="d2_low")
    d2_high = st.number_input("High ", value=float(b2) * 1.1 if float(b2) != 0 else 1.0, key="d2_high")
    d2_points = st.number_input("Points ", min_value=2, max_value=25, value=5, step=1, key="d2_points")
with cfg3:
    st.subheader("Driver 3 Range")
    d3_low = st.number_input("Low  ", value=float(b3) * 0.9, key="d3_low")
    d3_high = st.number_input("High  ", value=float(b3) * 1.1 if float(b3) != 0 else 1.0, key="d3_high")
    d3_points = st.number_input("Points  ", min_value=2, max_value=25, value=3, step=1, key="d3_points")

d1_vals = _grid_values(d1, float(d1_low), float(d1_high), int(d1_points))
d2_vals = _grid_values(d2, float(d2_low), float(d2_high), int(d2_points))
d3_vals = _grid_values(d3, float(d3_low), float(d3_high), int(d3_points))

scenario_count = len(d1_vals) * len(d2_vals) * len(d3_vals)
st.write(f"Scenarios: {scenario_count}")

if scenario_count > 5000:
    st.error("Too many scenarios. Reduce points (target <= 5,000).")
else:
    run = st.button("Run Sensitivity Study")
    if run:
        results = run_sensitivity(baseline_params, d1, d2, d3, d1_vals, d2_vals, d3_vals, metric)
        st.subheader("Scenario Results (Long Form)")
        st.dataframe(results, use_container_width=True)

        st.subheader("Heatmap Slices")
        tabs = st.tabs([f"{d3.label} = {v}" for v in d3_vals])
        for tab, v3 in zip(tabs, d3_vals):
            with tab:
                slice_df = results[results[d3.label] == v3]
                pivot = slice_df.pivot(index=d2.label, columns=d1.label, values="Value").sort_index().sort_index(axis=1)
                st.dataframe(pivot, use_container_width=True)
                fig = plot_heatmap(pivot, x_label=d1.label, y_label=d2.label, title=f"{metric} | {d3.label}={v3}")
                st.pyplot(fig)
