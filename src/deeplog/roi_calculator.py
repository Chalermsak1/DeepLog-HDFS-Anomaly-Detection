"""
DeepLog-Infinity Enterprise ROI & Total Economic Impact (TEI) Calculator
========================================================================
Implements a production-grade enterprise economic model adhering to Gartner
and Forrester Total Economic Impact (TEI) standards for evaluating the financial
return on investment of autonomous cloud observability and safe self-healing.

Key TEI Dimensions Quantified:
------------------------------
1. Downtime Cost Prevention:
   - High-fidelity modeling using Gartner's Tier-1 downtime cost benchmark
     ($5,600/minute standard, configurable from $1,000 to $10,000+/min).
   - Quantitative MTTR compression comparing:
     * Traditional Manual / Runbook Triage (industry average: 45.0 minutes)
     * Generative LLM Incident Agents (prompt/reasoning latency: 12.0 minutes)
     * DeepLog-Infinity Autonomous Safe Control (in-kernel eBPF/CBF: 2.8 seconds).
2. Cloud LLM API Cost Savings:
   - Eliminating expensive recurring LLM inference API calls (e.g., 100,000 incident
     triage calls to GPT-4o @ $0.03/call) via localized, zero-cost neural operators
     and hyperdimensional computing (HDC).
3. SRE / DevOps Toil & Labor Efficiency:
   - Quantifying SRE hours saved across triage, root-cause investigation, and
     false alarm chasing (with an 81.2% false alarm reduction via Consecutive Violations).
   - Converting reclaimed toil into Full-Time Equivalent (FTE) labor savings.
4. Formal Multi-Year Discounted Cash Flow Analysis:
   - 3-Year (or N-Year) Cash Flow projections with configurable discount rate (WACC).
   - Net Present Value (NPV), Internal Rate of Return (IRR), Payback Period (in months),
     and Return on Investment (ROI %).
   - Forrester-style conservative risk-adjustment parameters.
5. Multi-Format Reporting:
   - Structured JSON and Python dictionary export.
   - Formatted Executive Summary tables (Markdown and ASCII).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Union


@dataclass
class TEIConfig:
    """Configuration parameters for Enterprise Total Economic Impact (TEI) calculation.

    Attributes:
        downtime_cost_per_minute: Cost of Tier-1 service outage per minute in USD ($5,600/min Gartner benchmark).
        annual_incidents: Total number of Tier-1/Tier-2 service degradation incidents per year.
        manual_mttr_minutes: Mean Time To Resolution for human SRE manual runbooks (default: 45.0 min).
        llm_mttr_minutes: Mean Time To Resolution using cloud Generative LLM agents (default: 12.0 min).
        deeplog_mttr_seconds: Mean Time To Resolution for DeepLog-Infinity safe auto-control (default: 2.8 sec).
        annual_llm_triage_calls: Number of incident triage / log evaluation queries per year (default: 100,000).
        cost_per_llm_call: Blended prompt/completion cost per LLM API call in USD ($0.03 GPT-4o standard).
        sre_hourly_rate: Fully burdened hourly compensation for Site Reliability Engineers in USD (default: $95.00/hr).
        sre_hours_manual_per_incident: SRE engineering hours spent investigating an incident manually (default: 4.0 hrs).
        sre_hours_deeplog_per_incident: SRE engineering hours spent supervising DeepLog-Infinity (default: 0.25 hrs).
        baseline_false_alarms: Baseline false alarms generated annually (default: 1,683 from HDFS benchmark).
        deeplog_false_alarms: DeepLog-Infinity false alarms with consecutive violation filter (default: 317).
        sre_hours_per_false_alarm: Engineering hours wasted investigating a single false alarm (default: 0.5 hrs).
        annual_working_hours_per_fte: Standard working hours per year per FTE engineer (default: 2080.0 hrs).
        implementation_cost: Initial deployment, migration, and training CapEx in Year 0 (default: $50,000.00).
        annual_maintenance_cost: Annual operational infrastructure and maintenance OpEx (default: $12,000.00).
        annual_cost_growth_rate: Expected annual inflation / infrastructure scaling cost rate (default: 0.0).
        discount_rate: Cost of capital / discount rate for NPV calculation (default: 0.10, i.e., 10%).
        analysis_years: Multi-year projection time horizon in years (default: 3).
        risk_adjustment_pct: Forrester TEI risk adjustment factor applied to discount benefits and increase costs (default: 0.05, i.e., 5%).
    """

    downtime_cost_per_minute: float = 5600.0
    annual_incidents: int = 120
    manual_mttr_minutes: float = 45.0
    llm_mttr_minutes: float = 12.0
    deeplog_mttr_seconds: float = 2.8
    annual_llm_triage_calls: int = 100_000
    cost_per_llm_call: float = 0.03
    sre_hourly_rate: float = 95.0
    sre_hours_manual_per_incident: float = 4.0
    sre_hours_deeplog_per_incident: float = 0.25
    baseline_false_alarms: int = 1683
    deeplog_false_alarms: int = 317
    sre_hours_per_false_alarm: float = 0.5
    annual_working_hours_per_fte: float = 2080.0
    implementation_cost: float = 50_000.0
    annual_maintenance_cost: float = 12_000.0
    annual_cost_growth_rate: float = 0.0
    discount_rate: float = 0.10
    analysis_years: int = 3
    risk_adjustment_pct: float = 0.05

    def __post_init__(self) -> None:
        """Validate input parameter bounds and ensure defensive consistency."""
        if self.downtime_cost_per_minute < 0:
            raise ValueError(f"downtime_cost_per_minute must be non-negative: {self.downtime_cost_per_minute}")
        if self.annual_incidents < 0:
            raise ValueError(f"annual_incidents must be non-negative: {self.annual_incidents}")
        if self.manual_mttr_minutes < 0:
            raise ValueError(f"manual_mttr_minutes must be non-negative: {self.manual_mttr_minutes}")
        if self.llm_mttr_minutes < 0:
            raise ValueError(f"llm_mttr_minutes must be non-negative: {self.llm_mttr_minutes}")
        if self.deeplog_mttr_seconds < 0:
            raise ValueError(f"deeplog_mttr_seconds must be non-negative: {self.deeplog_mttr_seconds}")
        if self.annual_llm_triage_calls < 0:
            raise ValueError(f"annual_llm_triage_calls must be non-negative: {self.annual_llm_triage_calls}")
        if self.cost_per_llm_call < 0:
            raise ValueError(f"cost_per_llm_call must be non-negative: {self.cost_per_llm_call}")
        if self.sre_hourly_rate < 0:
            raise ValueError(f"sre_hourly_rate must be non-negative: {self.sre_hourly_rate}")
        if self.sre_hours_manual_per_incident < 0:
            raise ValueError(f"sre_hours_manual_per_incident must be non-negative: {self.sre_hours_manual_per_incident}")
        if self.sre_hours_deeplog_per_incident < 0:
            raise ValueError(f"sre_hours_deeplog_per_incident must be non-negative: {self.sre_hours_deeplog_per_incident}")
        if self.baseline_false_alarms < 0:
            raise ValueError(f"baseline_false_alarms must be non-negative: {self.baseline_false_alarms}")
        if self.deeplog_false_alarms < 0:
            raise ValueError(f"deeplog_false_alarms must be non-negative: {self.deeplog_false_alarms}")
        if self.sre_hours_per_false_alarm < 0:
            raise ValueError(f"sre_hours_per_false_alarm must be non-negative: {self.sre_hours_per_false_alarm}")
        if self.annual_working_hours_per_fte <= 0:
            raise ValueError(f"annual_working_hours_per_fte must be positive: {self.annual_working_hours_per_fte}")
        if self.implementation_cost < 0:
            raise ValueError(f"implementation_cost must be non-negative: {self.implementation_cost}")
        if self.annual_maintenance_cost < 0:
            raise ValueError(f"annual_maintenance_cost must be non-negative: {self.annual_maintenance_cost}")
        if self.discount_rate <= -1.0:
            raise ValueError(f"discount_rate must be strictly greater than -1.0: {self.discount_rate}")
        if self.analysis_years < 1:
            raise ValueError(f"analysis_years must be at least 1: {self.analysis_years}")
        if not (0.0 <= self.risk_adjustment_pct < 1.0):
            raise ValueError(f"risk_adjustment_pct must be between [0.0, 1.0): {self.risk_adjustment_pct}")


@dataclass
class MTTRComparison:
    """Detailed Mean Time To Resolution (MTTR) performance metrics."""

    manual_mttr_minutes: float
    llm_agent_mttr_minutes: float
    deeplog_mttr_seconds: float
    deeplog_mttr_minutes: float
    time_saved_vs_manual_minutes: float
    time_saved_vs_llm_minutes: float
    speedup_factor_vs_manual: float
    speedup_factor_vs_llm: float


@dataclass
class DowntimePreventionResult:
    """Financial impact of downtime cost reduction under Tier-1 SLAs."""

    cost_per_minute: float
    annual_incidents: int
    annual_downtime_minutes_manual: float
    annual_downtime_minutes_llm: float
    annual_downtime_minutes_deeplog: float
    annual_cost_manual: float
    annual_cost_llm: float
    annual_cost_deeplog: float
    annual_savings_vs_manual: float
    annual_savings_vs_llm: float


@dataclass
class LLMCostSavingsResult:
    """Direct cloud API cost savings from substituting cloud LLMs with local operators."""

    annual_calls: int
    cost_per_call: float
    baseline_cloud_llm_cost: float
    deeplog_api_cost: float
    annual_api_savings: float


@dataclass
class EngineerToilResult:
    """Engineering toil reduction and Full-Time Equivalent (FTE) savings."""

    sre_hourly_rate: float
    hours_manual_incident: float
    hours_deeplog_incident: float
    incident_hours_saved: float
    baseline_false_alarms: int
    deeplog_false_alarms: int
    false_alarm_reduction_pct: float
    false_alarm_hours_saved: float
    total_annual_hours_saved: float
    fte_saved: float
    annual_labor_savings: float


@dataclass
class YearlyCashFlow:
    """Cash flow projection for a single fiscal period."""

    year: int
    gross_benefits: float
    risk_adjusted_benefits: float
    gross_costs: float
    risk_adjusted_costs: float
    net_cash_flow: float
    discount_factor: float
    present_value: float
    cumulative_cash_flow: float
    cumulative_present_value: float


@dataclass
class TEIAnalysisResult:
    """Comprehensive executive outcome of Forrester/Gartner Total Economic Impact analysis."""

    config: TEIConfig
    mttr_comparison: MTTRComparison
    downtime_prevention: DowntimePreventionResult
    llm_cost_savings: LLMCostSavingsResult
    toil_reduction: EngineerToilResult
    cash_flows: List[YearlyCashFlow]
    total_present_value_benefits: float
    total_present_value_costs: float
    net_present_value: float
    internal_rate_of_return: float
    payback_period_months: float
    roi_percentage: float
    total_undiscounted_benefits: float
    total_undiscounted_costs: float
    total_undiscounted_net_profit: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert TEI outcome into a cleanly structured, JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize TEI outcome to a formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_executive_summary_table(self, format: str = "markdown") -> str:
        """Render a publication-ready executive summary table in Markdown or ASCII format."""
        if format.lower() == "ascii":
            return self._render_ascii_summary()
        return self._render_markdown_summary()

    def _render_markdown_summary(self) -> str:
        """Generate formatted Markdown executive summary table."""
        lines = [
            "# DeepLog-Infinity Enterprise Total Economic Impact (TEI) Summary",
            "",
            "## 1. Key Financial Summary (3-Year Horizon)",
            "| Metric | Value | Reference Standard / Notes |",
            "| :--- | :--- | :--- |",
            f"| **Net Present Value (NPV)** | **${self.net_present_value:,.2f}** | Risk-adjusted @ {self.config.discount_rate*100:.1f}% discount rate |",
            f"| **Return on Investment (ROI)** | **{self.roi_percentage:,.1f}%** | (NPV / Total PV Costs) * 100% |",
            f"| **Internal Rate of Return (IRR)** | **{self.internal_rate_of_return*100:,.1f}%** | Annualized financial internal rate of return |",
            f"| **Payback Period** | **{self.payback_period_months:.2f} months** | Breakeven time from initial deployment |",
            f"| **Total PV Benefits** | ${self.total_present_value_benefits:,.2f} | Discounted risk-adjusted gross benefits |",
            f"| **Total PV Costs** | ${self.total_present_value_costs:,.2f} | Discounted risk-adjusted CapEx + OpEx |",
            f"| **Undiscounted Net Profit** | ${self.total_undiscounted_net_profit:,.2f} | Cumulative cash flow over {self.config.analysis_years} years |",
            "",
            "## 2. MTTR & Operational Benchmark Comparison",
            "| Operational Paradigm | MTTR | Speedup vs Baseline | Outage Cost / Incident ($5,600/min) |",
            "| :--- | :--- | :--- | :--- |",
            f"| **Manual / Runbook Triage** | {self.mttr_comparison.manual_mttr_minutes:.1f} min | Baseline (1.0x) | ${self.mttr_comparison.manual_mttr_minutes * self.config.downtime_cost_per_minute:,.2f} |",
            f"| **Generative LLM Agent (Cloud)** | {self.mttr_comparison.llm_agent_mttr_minutes:.1f} min | {self.mttr_comparison.speedup_factor_vs_llm:.1f}x slower than DeepLog | ${self.mttr_comparison.llm_agent_mttr_minutes * self.config.downtime_cost_per_minute:,.2f} |",
            f"| **DeepLog-Infinity (Autonomous Safe Control)** | **{self.mttr_comparison.deeplog_mttr_seconds:.1f} sec** | **{self.mttr_comparison.speedup_factor_vs_manual:,.1f}x faster** | **${self.mttr_comparison.deeplog_mttr_minutes * self.config.downtime_cost_per_minute:,.2f}** |",
            "",
            "## 3. Annualized Economic Benefit Breakdown",
            "| Benefit Category | Annual Savings (USD) | Source / Driver |",
            "| :--- | :--- | :--- |",
            f"| **Downtime Prevention (vs Manual)** | ${self.downtime_prevention.annual_savings_vs_manual:,.2f} | {self.config.annual_incidents} Tier-1 incidents saved @ $5,600/min |",
            f"| **Downtime Prevention (vs LLM Agent)** | ${self.downtime_prevention.annual_savings_vs_llm:,.2f} | Sub-second remediation vs 12-min agent prompt loop |",
            f"| **Cloud LLM API Cost Avoidance** | ${self.llm_cost_savings.annual_api_savings:,.2f} | {self.config.annual_llm_triage_calls:,} calls @ ${self.config.cost_per_llm_call:.2f} to local HDC |",
            f"| **SRE Toil & False Alarm Reduction** | ${self.toil_reduction.annual_labor_savings:,.2f} | {self.toil_reduction.total_annual_hours_saved:,.1f} hrs saved ({self.toil_reduction.fte_saved:.2f} FTEs) |",
            f"| **False Alarm Suppression Rate** | -{self.toil_reduction.false_alarm_reduction_pct:.1f}% | Reduced from {self.toil_reduction.baseline_false_alarms} down to {self.toil_reduction.deeplog_false_alarms} alerts/yr |",
            "",
            "## 4. Multi-Year Risk-Adjusted Cash Flows",
            "| Period | Gross Benefits | Risk-Adj Benefits | Gross Costs | Risk-Adj Costs | Net Cash Flow | Disc. Factor | Present Value |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for cf in self.cash_flows:
            lines.append(
                f"| Year {cf.year} | ${cf.gross_benefits:,.2f} | ${cf.risk_adjusted_benefits:,.2f} | "
                f"${cf.gross_costs:,.2f} | ${cf.risk_adjusted_costs:,.2f} | ${cf.net_cash_flow:,.2f} | "
                f"{cf.discount_factor:.4f} | ${cf.present_value:,.2f} |"
            )
        return "\n".join(lines)

    def _render_ascii_summary(self) -> str:
        """Generate formatted ASCII text summary table."""
        sep = "+" + "-" * 78 + "+"
        header = f"| {'DEEPLOG-INFINITY ENTERPRISE TEI EXECUTIVE SUMMARY':^76} |"
        lines = [
            sep,
            header,
            sep,
            f"| {'FINANCIAL METRICS (3-YEAR RISK-ADJUSTED)':<40} | {'VALUE':>33} |",
            f"| Net Present Value (NPV):                  | ${self.net_present_value:>31,.2f} |",
            f"| Return on Investment (ROI):               | {self.roi_percentage:>31,.1f}% |",
            f"| Internal Rate of Return (IRR):            | {self.internal_rate_of_return*100:>31,.1f}% |",
            f"| Payback Period:                           | {f'{self.payback_period_months:.2f} months':>33} |",
            f"| Total PV Benefits:                        | ${self.total_present_value_benefits:>31,.2f} |",
            f"| Total PV Costs:                           | ${self.total_present_value_costs:>31,.2f} |",
            sep,
            f"| {'MTTR & SPEEDUP BENCHMARK':<40} | {'COMPARISON':>33} |",
            f"| Manual Runbook MTTR:                      | {f'{self.mttr_comparison.manual_mttr_minutes:.1f} mins':>33} |",
            f"| Cloud LLM Agent MTTR:                     | {f'{self.mttr_comparison.llm_agent_mttr_minutes:.1f} mins':>33} |",
            f"| DeepLog-Infinity MTTR:                    | {f'{self.mttr_comparison.deeplog_mttr_seconds:.1f} secs':>33} |",
            f"| Speedup vs Manual:                        | {f'{self.mttr_comparison.speedup_factor_vs_manual:,.1f}x faster':>33} |",
            sep,
            f"| {'ANNUAL OPERATIONAL SAVINGS':<40} | {'ANNUAL SAVINGS (USD)':>33} |",
            f"| Tier-1 Downtime Cost Avoidance:           | ${self.downtime_prevention.annual_savings_vs_manual:>31,.2f} |",
            f"| Cloud LLM API Cost Avoidance:             | ${self.llm_cost_savings.annual_api_savings:>31,.2f} |",
            f"| SRE Toil Reduction (Labor Savings):       | ${self.toil_reduction.annual_labor_savings:>31,.2f} |",
            f"| SRE Full-Time Equivalent (FTE) Saved:     | {f'{self.toil_reduction.fte_saved:.2f} FTEs':>33} |",
            f"| False Alarm Alert Reduction:              | {f'-{self.toil_reduction.false_alarm_reduction_pct:.1f}%':>33} |",
            sep,
        ]
        return "\n".join(lines)


class EnterpriseROICalculator:
    """Production-grade Enterprise ROI and Total Economic Impact (TEI) calculator."""

    def __init__(self, config: Optional[TEIConfig] = None) -> None:
        """Initialize calculator with given or default economic parameters."""
        self.config = config or TEIConfig()

    def compute_mttr_comparison(self) -> MTTRComparison:
        """Calculate MTTR differentials and relative speedup multipliers."""
        deeplog_min = self.config.deeplog_mttr_seconds / 60.0
        time_saved_manual = max(0.0, self.config.manual_mttr_minutes - deeplog_min)
        time_saved_llm = max(0.0, self.config.llm_mttr_minutes - deeplog_min)

        speedup_manual = (
            self.config.manual_mttr_minutes / deeplog_min
            if deeplog_min > 0
            else float("inf")
        )
        speedup_llm = (
            self.config.llm_mttr_minutes / deeplog_min
            if deeplog_min > 0
            else float("inf")
        )

        return MTTRComparison(
            manual_mttr_minutes=self.config.manual_mttr_minutes,
            llm_agent_mttr_minutes=self.config.llm_mttr_minutes,
            deeplog_mttr_seconds=self.config.deeplog_mttr_seconds,
            deeplog_mttr_minutes=deeplog_min,
            time_saved_vs_manual_minutes=time_saved_manual,
            time_saved_vs_llm_minutes=time_saved_llm,
            speedup_factor_vs_manual=speedup_manual,
            speedup_factor_vs_llm=speedup_llm,
        )

    def compute_downtime_prevention(self) -> DowntimePreventionResult:
        """Quantify financial downtime losses prevented under Tier-1 SLA parameters."""
        deeplog_min = self.config.deeplog_mttr_seconds / 60.0

        downtime_manual = self.config.annual_incidents * self.config.manual_mttr_minutes
        downtime_llm = self.config.annual_incidents * self.config.llm_mttr_minutes
        downtime_deeplog = self.config.annual_incidents * deeplog_min

        cost_manual = downtime_manual * self.config.downtime_cost_per_minute
        cost_llm = downtime_llm * self.config.downtime_cost_per_minute
        cost_deeplog = downtime_deeplog * self.config.downtime_cost_per_minute

        savings_vs_manual = max(0.0, cost_manual - cost_deeplog)
        savings_vs_llm = max(0.0, cost_llm - cost_deeplog)

        return DowntimePreventionResult(
            cost_per_minute=self.config.downtime_cost_per_minute,
            annual_incidents=self.config.annual_incidents,
            annual_downtime_minutes_manual=downtime_manual,
            annual_downtime_minutes_llm=downtime_llm,
            annual_downtime_minutes_deeplog=downtime_deeplog,
            annual_cost_manual=cost_manual,
            annual_cost_llm=cost_llm,
            annual_cost_deeplog=cost_deeplog,
            annual_savings_vs_manual=savings_vs_manual,
            annual_savings_vs_llm=savings_vs_llm,
        )

    def compute_llm_cost_savings(self) -> LLMCostSavingsResult:
        """Quantify recurring cloud LLM API fees eliminated through localized inference."""
        baseline_cost = self.config.annual_llm_triage_calls * self.config.cost_per_llm_call
        deeplog_cost = 0.0  # Zero cloud API fee: localized neural operators & HDC
        annual_savings = max(0.0, baseline_cost - deeplog_cost)

        return LLMCostSavingsResult(
            annual_calls=self.config.annual_llm_triage_calls,
            cost_per_call=self.config.cost_per_llm_call,
            baseline_cloud_llm_cost=baseline_cost,
            deeplog_api_cost=deeplog_cost,
            annual_api_savings=annual_savings,
        )

    def compute_toil_reduction(self) -> EngineerToilResult:
        """Quantify SRE engineering toil reclaimed from triage and false alarm suppression."""
        incident_hours_saved = self.config.annual_incidents * max(
            0.0,
            self.config.sre_hours_manual_per_incident - self.config.sre_hours_deeplog_per_incident,
        )

        false_alarms_delta = max(0, self.config.baseline_false_alarms - self.config.deeplog_false_alarms)
        reduction_pct = (
            (false_alarms_delta / self.config.baseline_false_alarms * 100.0)
            if self.config.baseline_false_alarms > 0
            else 0.0
        )
        false_alarm_hours_saved = false_alarms_delta * self.config.sre_hours_per_false_alarm

        total_hours_saved = incident_hours_saved + false_alarm_hours_saved
        fte_saved = total_hours_saved / self.config.annual_working_hours_per_fte
        annual_labor_savings = total_hours_saved * self.config.sre_hourly_rate

        return EngineerToilResult(
            sre_hourly_rate=self.config.sre_hourly_rate,
            hours_manual_incident=self.config.sre_hours_manual_per_incident,
            hours_deeplog_incident=self.config.sre_hours_deeplog_per_incident,
            incident_hours_saved=incident_hours_saved,
            baseline_false_alarms=self.config.baseline_false_alarms,
            deeplog_false_alarms=self.config.deeplog_false_alarms,
            false_alarm_reduction_pct=reduction_pct,
            false_alarm_hours_saved=false_alarm_hours_saved,
            total_annual_hours_saved=total_hours_saved,
            fte_saved=fte_saved,
            annual_labor_savings=annual_labor_savings,
        )

    def compute_cash_flows(
        self,
        annual_gross_benefits: float,
    ) -> List[YearlyCashFlow]:
        """Project multi-year cash flows incorporating Forrester risk adjustments and discounting."""
        flows: List[YearlyCashFlow] = []
        r = self.config.discount_rate
        risk = self.config.risk_adjustment_pct

        cum_cash_flow = 0.0
        cum_pv = 0.0

        # Year 0: Initial CapEx implementation investment
        y0_gross_benefit = 0.0
        y0_risk_benefit = 0.0
        y0_gross_cost = self.config.implementation_cost
        y0_risk_cost = y0_gross_cost * (1.0 + risk)
        y0_net_cf = -y0_risk_cost
        y0_df = 1.0
        y0_pv = y0_net_cf * y0_df

        cum_cash_flow += y0_net_cf
        cum_pv += y0_pv

        flows.append(
            YearlyCashFlow(
                year=0,
                gross_benefits=y0_gross_benefit,
                risk_adjusted_benefits=y0_risk_benefit,
                gross_costs=y0_gross_cost,
                risk_adjusted_costs=y0_risk_cost,
                net_cash_flow=y0_net_cf,
                discount_factor=y0_df,
                present_value=y0_pv,
                cumulative_cash_flow=cum_cash_flow,
                cumulative_present_value=cum_pv,
            )
        )

        # Years 1 through N: Operational recurring benefits and OpEx maintenance
        for year in range(1, self.config.analysis_years + 1):
            growth_factor = (1.0 + self.config.annual_cost_growth_rate) ** (year - 1)
            y_gross_cost = self.config.annual_maintenance_cost * growth_factor
            y_risk_cost = y_gross_cost * (1.0 + risk)

            y_gross_benefit = annual_gross_benefits
            y_risk_benefit = y_gross_benefit * (1.0 - risk)

            y_net_cf = y_risk_benefit - y_risk_cost
            y_df = 1.0 / ((1.0 + r) ** year)
            y_pv = y_net_cf * y_df

            cum_cash_flow += y_net_cf
            cum_pv += y_pv

            flows.append(
                YearlyCashFlow(
                    year=year,
                    gross_benefits=y_gross_benefit,
                    risk_adjusted_benefits=y_risk_benefit,
                    gross_costs=y_gross_cost,
                    risk_adjusted_costs=y_risk_cost,
                    net_cash_flow=y_net_cf,
                    discount_factor=y_df,
                    present_value=y_pv,
                    cumulative_cash_flow=cum_cash_flow,
                    cumulative_present_value=cum_pv,
                )
            )

        return flows

    @staticmethod
    def compute_irr(cash_flows: Sequence[float], max_iter: int = 1000, tol: float = 1e-7) -> float:
        """Compute exact Internal Rate of Return (IRR) via robust hybrid bisection and Newton-Raphson.

        Solves for rate r such that: sum_{t=0}^N CF_t / (1 + r)^t = 0.
        """
        if not cash_flows or len(cash_flows) < 2:
            return float("nan")

        # Check for standard sign change
        has_pos = any(cf > 0 for cf in cash_flows)
        has_neg = any(cf < 0 for cf in cash_flows)
        if not (has_pos and has_neg):
            return float("nan")

        def npv(rate: float) -> float:
            if rate <= -1.0:
                return float("inf")
            total = 0.0
            for t, cf in enumerate(cash_flows):
                total += cf / ((1.0 + rate) ** t)
            return total

        def npv_derivative(rate: float) -> float:
            if rate <= -1.0:
                return float("-inf")
            total = 0.0
            for t, cf in enumerate(cash_flows):
                if t > 0:
                    total -= t * cf / ((1.0 + rate) ** (t + 1))
            return total

        # Attempt Newton-Raphson starting from a reasonable initial rate
        rate = 0.20
        for _ in range(max_iter):
            val = npv(rate)
            if math.isnan(val) or math.isinf(val):
                break
            if abs(val) < tol:
                return rate
            deriv = npv_derivative(rate)
            if abs(deriv) < 1e-12:
                break
            new_rate = rate - val / deriv
            if new_rate <= -0.999:
                break
            if abs(new_rate - rate) < tol:
                return new_rate
            rate = new_rate

        # Robust bisection fallback across bounded bracket [-0.99, 100.0]
        low = -0.99
        high = 100.0
        val_low = npv(low)
        val_high = npv(high)

        # Expand upper bound if necessary for exceptionally high return profiles
        while val_low * val_high > 0 and high < 1e7:
            high *= 10.0
            val_high = npv(high)

        if val_low * val_high > 0:
            # No sign change within search envelope; return rate with minimal absolute residual
            return rate if not math.isnan(rate) else float("nan")

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            val_mid = npv(mid)
            if abs(val_mid) < tol or (high - low) / 2.0 < tol:
                return mid
            if (val_low > 0 and val_mid > 0) or (val_low < 0 and val_mid < 0):
                low = mid
                val_low = val_mid
            else:
                high = mid
                val_high = val_mid

        return (low + high) / 2.0

    @staticmethod
    def compute_payback_period_months(yearly_flows: Sequence[YearlyCashFlow]) -> float:
        """Compute the payback period in months when cumulative cash flow achieves breakeven."""
        if not yearly_flows or len(yearly_flows) < 2:
            return float("nan")

        y0 = yearly_flows[0]
        initial_investment = abs(y0.net_cash_flow)
        if initial_investment <= 1e-9:
            return 0.0

        # Cumulative tracking through periods
        for i in range(1, len(yearly_flows)):
            prev_flow = yearly_flows[i - 1]
            curr_flow = yearly_flows[i]

            if curr_flow.cumulative_cash_flow >= 0.0:
                # Breakeven occurred within year `i`
                prev_deficit = abs(prev_flow.cumulative_cash_flow)
                annual_net_gain = curr_flow.net_cash_flow
                if annual_net_gain > 0:
                    fraction = prev_deficit / annual_net_gain
                    months = (i - 1) * 12.0 + fraction * 12.0
                    return max(0.0, months)
                return float((i - 1) * 12.0)

        # Did not pay back within analysis window
        return float(len(yearly_flows) * 12.0)

    def calculate(self) -> TEIAnalysisResult:
        """Execute full Gartner/Forrester Total Economic Impact calculation."""
        mttr = self.compute_mttr_comparison()
        downtime = self.compute_downtime_prevention()
        llm = self.compute_llm_cost_savings()
        toil = self.compute_toil_reduction()

        # Gross annual benefits = Downtime avoidance + Cloud LLM API avoidance + SRE Labor savings
        annual_gross_benefits = (
            downtime.annual_savings_vs_manual
            + llm.annual_api_savings
            + toil.annual_labor_savings
        )

        cash_flows = self.compute_cash_flows(annual_gross_benefits)

        # Present values
        total_pv_benefits = sum(
            cf.risk_adjusted_benefits * cf.discount_factor
            for cf in cash_flows
            if cf.year > 0
        )
        total_pv_costs = sum(
            cf.risk_adjusted_costs * cf.discount_factor
            for cf in cash_flows
        )

        npv = total_pv_benefits - total_pv_costs
        roi_percentage = (npv / total_pv_costs * 100.0) if total_pv_costs > 0 else 0.0

        raw_cfs = [cf.net_cash_flow for cf in cash_flows]
        irr = self.compute_irr(raw_cfs)
        payback_months = self.compute_payback_period_months(cash_flows)

        total_undisc_benefits = sum(cf.risk_adjusted_benefits for cf in cash_flows)
        total_undisc_costs = sum(cf.risk_adjusted_costs for cf in cash_flows)
        total_undisc_profit = total_undisc_benefits - total_undisc_costs

        return TEIAnalysisResult(
            config=self.config,
            mttr_comparison=mttr,
            downtime_prevention=downtime,
            llm_cost_savings=llm,
            toil_reduction=toil,
            cash_flows=cash_flows,
            total_present_value_benefits=total_pv_benefits,
            total_present_value_costs=total_pv_costs,
            net_present_value=npv,
            internal_rate_of_return=irr,
            payback_period_months=payback_months,
            roi_percentage=roi_percentage,
            total_undiscounted_benefits=total_undisc_benefits,
            total_undiscounted_costs=total_undisc_costs,
            total_undiscounted_net_profit=total_undisc_profit,
        )


def compute_enterprise_roi(
    downtime_cost_per_minute: float = 5600.0,
    annual_incidents: int = 120,
    manual_mttr_minutes: float = 45.0,
    llm_mttr_minutes: float = 12.0,
    deeplog_mttr_seconds: float = 2.8,
    annual_llm_triage_calls: int = 100_000,
    cost_per_llm_call: float = 0.03,
    sre_hourly_rate: float = 95.0,
    discount_rate: float = 0.10,
    analysis_years: int = 3,
    risk_adjustment_pct: float = 0.05,
    **kwargs: Any,
) -> TEIAnalysisResult:
    """Convenience helper to compute Enterprise Total Economic Impact and ROI."""
    config = TEIConfig(
        downtime_cost_per_minute=downtime_cost_per_minute,
        annual_incidents=annual_incidents,
        manual_mttr_minutes=manual_mttr_minutes,
        llm_mttr_minutes=llm_mttr_minutes,
        deeplog_mttr_seconds=deeplog_mttr_seconds,
        annual_llm_triage_calls=annual_llm_triage_calls,
        cost_per_llm_call=cost_per_llm_call,
        sre_hourly_rate=sre_hourly_rate,
        discount_rate=discount_rate,
        analysis_years=analysis_years,
        risk_adjustment_pct=risk_adjustment_pct,
        **kwargs,
    )
    calc = EnterpriseROICalculator(config)
    return calc.calculate()
