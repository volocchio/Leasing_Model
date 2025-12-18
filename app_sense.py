import itertools
from datetime import datetime
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


@dataclass(frozen=True)
class DriverSpec:
    key: str
    label: str
    unit: str
    kind: str  # float | int | percent


def _to_internal_value(kind: str, v: float) -> float:
    if kind == "percent":
        return float(v) / 100.0
    if kind == "thousands":
        return float(v) * 1000.0
    return float(v)


def _to_display_value(kind: str, v: float) -> float:
    if kind == "percent":
        return float(v) * 100.0
    if kind == "thousands":
        return float(v) / 1000.0
    return float(v)


def run_model(params: Dict[str, Any]) -> Dict[str, float]:
    years = list(range(2026, 2036))

    revenue_start_q_index = int(params["revenue_start_q_index"])
    inventory_purchase_q_index = int(params["inventory_purchase_q_index"])
    revenue_start_year = int(params["revenue_start_year"])

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
    cert_duration_quarters = int(params["cert_duration_quarters"])
    cert_spend_per_quarter = (float(cert_readiness_cost) / float(cert_duration_quarters)) if int(cert_duration_quarters) > 0 else 0.0

    debt_amount = float(params["debt_amount"])
    debt_apr = float(params["debt_apr"])
    debt_term_years = int(params["debt_term_years"])

    tax_rate = float(params["tax_rate"])
    wacc = float(params["wacc"])
    terminal_growth = float(params["terminal_growth"])

    opex = dict(params["opex"])

    annual_data: Dict[int, Dict[str, float]] = {}

    cum_shipsets = 0.0
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
                "EBITDA": 0.0,
                "CapExInv": 0.0,
                "Taxes": 0.0,
            }
            year_taxable_income = 0.0

        capex = 0.0
        inventory = 0.0

        if i < revenue_start_q_index:
            new_installs = 0.0
            revenue = 0.0
            cogs = 0.0
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
            rev_per_shipset = quarter_saving * float(split_pct)

            rev_q_idx = int(i - int(revenue_start_q_index))
            if rev_q_idx == 0:
                new_installs = float(q1_installs)
            elif rev_q_idx == 1:
                new_installs = float(q2_installs)
            elif rev_q_idx == 2:
                new_installs = float(q3_installs)
            elif rev_q_idx == 3:
                new_installs = float(q4_installs)
            else:
                revenue_year = int(rev_q_idx // 4)
                if revenue_year == 1:
                    new_installs = 910.0 / 4.0
                else:
                    new_installs = 1040.0 / 4.0

            remaining_shipsets = max(0.0, float(tam_shipsets) - float(cum_shipsets))
            new_installs = min(float(new_installs), float(remaining_shipsets))

            cum_shipsets_beg = float(cum_shipsets)
            cum_shipsets_end = float(cum_shipsets_beg) + float(new_installs)
            avg_shipsets = float(cum_shipsets_beg) + 0.5 * float(new_installs)
            cum_shipsets = float(cum_shipsets_end)

            revenue = float(avg_shipsets) * float(rev_per_shipset) / 1e6

            cogs_per_kit = float(base_cogs) * float((1 + float(cogs_inflation)) ** int(year_idx))
            cogs = float(new_installs) * float(cogs_per_kit) / 1e6

        gross_profit = float(revenue) - float(cogs)
        opex_q = float(opex.get(int(yr), 15)) / 4.0
        ebitda = float(gross_profit) - float(opex_q)
        total_outflow = float(capex) + float(inventory)

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

        if i < revenue_start_q_index:
            equity_cf = -float(equity_contribution)
        else:
            equity_cf = float(net_cash_after_debt)

        equity_cum_cf += float(equity_cf)

        year_sums["EBITDA"] += float(ebitda)
        year_sums["CapExInv"] += float(total_outflow)
        year_sums["Taxes"] += float(taxes)

        if int(qtr) == 4:
            annual_data[int(yr)] = {
                "EBITDA": float(year_sums["EBITDA"]),
                "CapExInv": float(year_sums["CapExInv"]),
                "Taxes": float(year_sums["Taxes"]),
            }
            year_sums = None

    df = pd.DataFrame.from_dict(annual_data, orient="index")
    df.index.name = "Year"

    unlevered_taxes = (df["EBITDA"].clip(lower=0.0) * float(tax_rate)).astype(float)
    unlevered_fcf = (df["EBITDA"].astype(float) - unlevered_taxes - df["CapExInv"].astype(float)).astype(float)

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
    base_fuel_burn_gal_per_hour = st.sidebar.slider("Base Fuel Burn (gal/hour)", min_value=600, max_value=1200, value=750, step=50)

    cogs_inflation = st.sidebar.slider("Annual COGS Inflation (%)", min_value=0.0, max_value=15.0, value=4.0, step=0.5) / 100
    base_cogs_k = st.sidebar.slider("Base COGS per Kit at First Revenue Year ($1000)", min_value=100, max_value=800, value=400, step=10)
    base_cogs = float(base_cogs_k) * 1000.0

    fuel_saving_pct = st.sidebar.slider("Fuel Savings % per Aircraft", min_value=5.0, max_value=15.0, value=10.0, step=0.5) / 100
    fuel_savings_split_to_tamarack = st.sidebar.slider("Fuel Savings Split to Tamarack (%)", min_value=0.0, max_value=100.0, value=50.0, step=1.0) / 100

    cert_readiness_cost = st.sidebar.slider("Equity ($M)", min_value=100.0, max_value=300.0, value=180.0, step=10.0)
    cert_duration_years = st.sidebar.slider("Certification Duration (Years)", min_value=0.25, max_value=5.0, value=2.0, step=0.25)
    cert_duration_quarters = max(1, int(round(float(cert_duration_years) * 4.0)))

    inventory_kits_pre_install = st.sidebar.slider("Inventory Kits Before First Install", min_value=50, max_value=200, value=90, step=10)
    tam_shipsets = st.sidebar.slider("Total Addressable Market (Max Shipsets in 10 Years)", min_value=1000, max_value=10000, value=7500, step=500)

    debt_amount = st.sidebar.slider("Max Debt Available ($M)", min_value=0.0, max_value=500.0, value=float(cert_readiness_cost), step=10.0)
    debt_apr = st.sidebar.slider("Debt APR (%)", min_value=0.0, max_value=20.0, value=10.0, step=0.5) / 100
    debt_term_years = st.sidebar.slider("Debt Term (Years)", min_value=1, max_value=15, value=7, step=1)

    tax_rate = st.sidebar.slider("Income Tax Rate (%)", min_value=0.0, max_value=40.0, value=21.0, step=0.5) / 100
    wacc = st.sidebar.slider("WACC (%)", min_value=0.0, max_value=30.0, value=9.5, step=0.5) / 100
    terminal_growth = st.sidebar.slider("Terminal Growth Rate (%)", min_value=-2.0, max_value=8.0, value=2.5, step=0.5) / 100

    st.sidebar.header("Installs")
    q1_installs = st.sidebar.slider("Q1 Installs", min_value=0, max_value=200, value=98, step=10)
    q2_installs = st.sidebar.slider("Q2 Installs", min_value=0, max_value=200, value=98, step=10)
    q3_installs = st.sidebar.slider("Q3 Installs", min_value=0, max_value=200, value=98, step=10)
    q4_installs = st.sidebar.slider("Q4 Installs and beyond", min_value=0, max_value=200, value=96, step=10)

    revenue_start_q_index = int(cert_duration_quarters)
    revenue_start_year = 2026 + (int(revenue_start_q_index) // 4)
    inventory_purchase_q_index = max(0, int(revenue_start_q_index) - 1)
    inventory_year = 2026 + (int(inventory_purchase_q_index) // 4)

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
        "revenue_start_q_index": int(revenue_start_q_index),
        "inventory_purchase_q_index": int(inventory_purchase_q_index),
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
        DriverSpec("wacc", "WACC (%)", "%", "percent"),
        DriverSpec("base_fuel_price", "Fuel Price ($/gal)", "$/gal", "float"),
        DriverSpec("cert_duration_years", "Certification Duration (Years)", "years", "float"),
        DriverSpec("fuel_savings_split_to_tamarack", "Fuel Savings Split to Tamarack (%)", "%", "percent"),
        DriverSpec("fuel_inflation", "Annual Fuel Inflation (%)", "%", "percent"),
        DriverSpec("block_hours", "Block Hours per Aircraft per Year", "hours", "int"),
        DriverSpec("base_fuel_burn_gal_per_hour", "Base Fuel Burn (gal/hour)", "gal/hour", "int"),
        DriverSpec("cogs_inflation", "Annual COGS Inflation", "%", "percent"),
        DriverSpec("base_cogs", "Base COGS per Kit (First Revenue Year) ($000)", "$000/kit", "thousands"),
        DriverSpec("fuel_saving_pct", "Fuel Savings % per Aircraft", "%", "percent"),
        DriverSpec("cert_readiness_cost", "Equity", "$M", "float"),
        DriverSpec("inventory_kits_pre_install", "Inventory Kits Before First Install", "kits", "int"),
        DriverSpec("tam_shipsets", "Total Addressable Market", "shipsets", "int"),
        DriverSpec("debt_amount", "Max Debt Available", "$M", "float"),
        DriverSpec("debt_apr", "Debt APR", "%", "percent"),
        DriverSpec("debt_term_years", "Debt Term", "years", "int"),
        DriverSpec("tax_rate", "Income Tax Rate", "%", "percent"),
        DriverSpec("terminal_growth", "Terminal Growth Rate", "%", "percent"),
    ]


def _grid_values(spec: DriverSpec, low_disp: float, high_disp: float, points: int) -> List[float]:
    if points < 2:
        points = 2

    if spec.kind in {"int", "thousands"}:
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

        revenue_start_q_index = int(qtrs)
        revenue_start_year = 2026 + (int(revenue_start_q_index) // 4)
        inventory_purchase_q_index = max(0, int(revenue_start_q_index) - 1)
        inventory_year = 2026 + (int(inventory_purchase_q_index) // 4)

        params["revenue_start_q_index"] = int(revenue_start_q_index)
        params["inventory_purchase_q_index"] = int(inventory_purchase_q_index)
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

    df = pd.DataFrame(rows)
    for spec in (d1, d2, d3):
        col = spec.label
        if col in df.columns:
            if spec.kind in {"int", "thousands"}:
                df[col] = df[col].round(0).astype(int)
            elif spec.key == "cert_duration_years":
                df[col] = df[col].astype(float).round(2)
            else:
                df[col] = df[col].astype(float).round(2)
    if "Value" in df.columns:
        df["Value"] = df["Value"].astype(float).round(1)
    return df


def plot_heatmap(
    pivot: pd.DataFrame,
    x_label: str,
    y_label: str,
    title: str,
    x_tick_fmt: str = "%.1f",
    y_tick_fmt: str = "%.1f",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))

    z = np.ma.masked_invalid(pivot.values.astype(float))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#EEEEEE")
    im = ax.imshow(z, aspect="auto", origin="lower", cmap=cmap, interpolation="nearest")

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([x_tick_fmt % float(v) for v in pivot.columns.tolist()], rotation=45, ha="right")

    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([y_tick_fmt % float(v) for v in pivot.index.tolist()])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Value")
    cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    fig.tight_layout()
    return fig


def render_sensitivity_app(baseline_params: Dict[str, Any] | None = None, show_title: bool = True) -> None:
    if show_title:
        st.title("Tamarack Aerospace – 3-Driver Sensitivity Study")

    if baseline_params is None:
        baseline_params = build_baseline_params()

    st.header("Baseline Outputs")
    baseline_outputs = run_model(baseline_params)
    baseline_out_df = pd.DataFrame({
        "Metric": list(baseline_outputs.keys()),
        "Baseline": list(baseline_outputs.values()),
    })
    baseline_out_df["Baseline"] = baseline_out_df["Baseline"].astype(float).round(1)
    st.dataframe(baseline_out_df, hide_index=True, use_container_width=False)

    st.header("Sensitivity Study")

    drivers = build_driver_catalog(baseline_params)
    driver_by_key = {d.key: d for d in drivers}
    driver_labels = {d.key: d.label for d in drivers}

    def _key_index(keys: List[str], desired: str, fallback: int = 0) -> int:
        return keys.index(desired) if desired in keys else fallback

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(
            '<div style="font-size: 1.15rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 1</div>',
            unsafe_allow_html=True,
        )
        d1_options = [d.key for d in drivers]
        d1_key = st.selectbox(
            "Driver 1",
            options=d1_options,
            format_func=lambda k: driver_labels[k],
            index=_key_index(d1_options, "wacc", 0),
            label_visibility="collapsed",
        )
    with col_b:
        st.markdown(
            '<div style="font-size: 1.15rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 2</div>',
            unsafe_allow_html=True,
        )
        d2_options = [d.key for d in drivers if d.key != d1_key]
        d2_default = "base_fuel_price" if "base_fuel_price" in d2_options else (d2_options[0] if len(d2_options) > 0 else d1_key)
        d2_key = st.selectbox(
            "Driver 2",
            options=d2_options,
            format_func=lambda k: driver_labels[k],
            index=_key_index(d2_options, d2_default, 0),
            label_visibility="collapsed",
        )
    with col_c:
        st.markdown(
            '<div style="font-size: 1.15rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 3</div>',
            unsafe_allow_html=True,
        )
        d3_options = [d.key for d in drivers if d.key not in {d1_key, d2_key}]
        d3_default = "cert_duration_years" if "cert_duration_years" in d3_options else (d3_options[0] if len(d3_options) > 0 else d1_key)
        d3_key = st.selectbox(
            "Driver 3",
            options=d3_options,
            format_func=lambda k: driver_labels[k],
            index=_key_index(d3_options, d3_default, 0),
            label_visibility="collapsed",
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
        st.markdown(
            '<div style="font-size: 1.05rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 1 Range</div>',
            unsafe_allow_html=True,
        )
        d1_low = st.number_input("Low", value=float(b1) * 0.9, key=f"d1_low__{d1.key}")
        d1_high = st.number_input("High", value=float(b1) * 1.1 if float(b1) != 0 else 1.0, key=f"d1_high__{d1.key}")
        d1_points = st.number_input("Points", min_value=2, max_value=25, value=5, step=1, key=f"d1_points__{d1.key}")
    with cfg2:
        st.markdown(
            '<div style="font-size: 1.05rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 2 Range</div>',
            unsafe_allow_html=True,
        )
        d2_low = st.number_input("Low ", value=float(b2) * 0.9, key=f"d2_low__{d2.key}")
        d2_high = st.number_input("High ", value=float(b2) * 1.1 if float(b2) != 0 else 1.0, key=f"d2_high__{d2.key}")
        d2_points = st.number_input("Points ", min_value=2, max_value=25, value=5, step=1, key=f"d2_points__{d2.key}")
    with cfg3:
        st.markdown(
            '<div style="font-size: 1.05rem; font-weight: 800; margin-bottom: 0.25rem;">Driver 3 Range</div>',
            unsafe_allow_html=True,
        )
        d3_low = st.number_input("Low  ", value=float(b3) * 0.9, key=f"d3_low__{d3.key}")
        d3_high = st.number_input("High  ", value=float(b3) * 1.1 if float(b3) != 0 else 1.0, key=f"d3_high__{d3.key}")
        d3_points = st.number_input("Points  ", min_value=2, max_value=25, value=3, step=1, key=f"d3_points__{d3.key}")

    d1_vals = _grid_values(d1, float(d1_low), float(d1_high), int(d1_points))
    d2_vals = _grid_values(d2, float(d2_low), float(d2_high), int(d2_points))
    d3_vals = _grid_values(d3, float(d3_low), float(d3_high), int(d3_points))

    def _fmt_value(spec: DriverSpec, v: float) -> str:
        if spec.kind in {"int", "thousands"}:
            return f"{int(round(float(v)))}"
        if spec.key == "cert_duration_years":
            return f"{float(v):.2f}"
        return f"{float(v):.1f}"

    def _tick_fmt(spec: DriverSpec) -> str:
        if spec.kind in {"int", "thousands"}:
            return "%.0f"
        if spec.key == "cert_duration_years":
            return "%.2f"
        return "%.1f"

    def _round_series_for_spec(s: pd.Series, spec: DriverSpec) -> pd.Series:
        if spec.kind in {"int", "thousands"}:
            return s.astype(float).round(0)
        if spec.key == "cert_duration_years":
            return s.astype(float).round(2)
        return s.astype(float).round(1)

    def _disp_key(spec: DriverSpec, v: float) -> float:
        if spec.kind in {"int", "thousands"}:
            return float(int(round(float(v))))
        if spec.key == "cert_duration_years":
            return float(round(float(v), 2))
        return float(round(float(v), 1))

    scenario_count = len(d1_vals) * len(d2_vals) * len(d3_vals)
    st.write(f"Scenarios: {scenario_count}")

    if scenario_count > 5000:
        st.error("Too many scenarios. Reduce points (target <= 5,000).")
        return

    run = st.button("Run Sensitivity Study")
    if not run:
        return

    results = run_sensitivity(baseline_params, d1, d2, d3, d1_vals, d2_vals, d3_vals, metric)
    st.subheader("Scenario Results (Long Form)")
    st.dataframe(results, use_container_width=True)

    st.subheader("Heatmap Slices")
    st.markdown(
        f"**Driver 1 (X-axis):** {d1.label}  \n"
        f"**Driver 2 (Y-axis):** {d2.label}  \n"
        f"**Driver 3 (Tabs/Slices):** {d3.label}"
    )

    st.markdown(
        """
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
        """,
        unsafe_allow_html=True,
    )

    d3_disp_vals = [_disp_key(d3, float(v)) for v in d3_vals]
    seen = set()
    d3_disp_vals = [v for v in d3_disp_vals if not (v in seen or seen.add(v))]

    tabs = st.tabs([f"{d3.label} = {_fmt_value(d3, float(v))}" for v in d3_disp_vals])
    pdf_figs: List[Figure] = []
    for tab, v3 in zip(tabs, d3_disp_vals):
        with tab:
            st.markdown(f"**Slice:** {d3.label} = {_fmt_value(d3, float(v3))}")
            slice_df = results[_round_series_for_spec(results[d3.label], d3) == float(v3)]
            pivot = slice_df.pivot(index=d2.label, columns=d1.label, values="Value").sort_index().sort_index(axis=1).astype(float).round(1)
            st.dataframe(pivot, use_container_width=True)
            fig = plot_heatmap(
                pivot,
                x_label=d1.label,
                y_label=d2.label,
                title=f"{metric} | {d3.label}={_fmt_value(d3, float(v3))}",
                x_tick_fmt=_tick_fmt(d1),
                y_tick_fmt=_tick_fmt(d2),
            )
            st.pyplot(fig)
            pdf_figs.append(fig)

    buf = BytesIO()
    with PdfPages(buf) as pdf:
        title_fig = plt.figure(figsize=(8.5, 11))
        title_fig.clf()
        title_fig.text(0.5, 0.94, "Sensitivity Study Report", ha="center", va="top", fontsize=18, fontweight="bold")
        title_fig.text(0.5, 0.90, "Tamarack Aerospace – 3-Driver Sensitivity Study", ha="center", va="top", fontsize=12)
        title_fig.text(
            0.08,
            0.84,
            "\n".join(
                [
                    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    f"Metric: {metric}",
                    f"Driver 1: {d1.label}",
                    f"Driver 2: {d2.label}",
                    f"Driver 3 (Slices): {d3.label}",
                    f"Scenarios: {scenario_count}",
                ]
            ),
            ha="left",
            va="top",
            fontsize=11,
        )
        pdf.savefig(title_fig, bbox_inches="tight")
        plt.close(title_fig)

        for fig in pdf_figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

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
            label="Download Sensitivity Study PDF",
            data=buf.getvalue(),
            file_name="sensitivity_study.pdf",
            mime="application/pdf",
            use_container_width=False,
        )


if __name__ == "__main__":
    st.set_page_config(layout="wide")
    render_sensitivity_app(show_title=True)
