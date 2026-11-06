"""
Comprehensive unit tests for DeepLog-Infinity Enterprise ROI and TEI Calculator
================================================================================
Verifies compliance with Gartner and Forrester Total Economic Impact (TEI) standards:
- Tier-1 downtime cost prevention calculations ($5,600/min standard)
- MTTR compression (Manual 45m vs LLM Agent 12m vs DeepLog-Infinity 2.8s)
- Cloud LLM API cost avoidance (100k queries @ $0.03/query -> $0)
- SRE toil reduction and FTE savings (with 81.2% false alarm suppression)
- 3-Year NPV, IRR, Payback Period, and ROI %
- Defensive validation of negative/zero inputs and edge cases
- Export formats (Dict, JSON, Markdown/ASCII executive tables)
"""

from __future__ import annotations

import json
import math
import pytest

from deeplog.roi_calculator import (
    EnterpriseROICalculator,
    TEIConfig,
    TEIAnalysisResult,
    MTTRComparison,
    DowntimePreventionResult,
    LLMCostSavingsResult,
    EngineerToilResult,
    YearlyCashFlow,
    compute_enterprise_roi,
)


class TestTEIConfigValidation:
    """Validate defensive bounds checking and input constraints."""

    def test_default_config_valid(self):
        config = TEIConfig()
        assert config.downtime_cost_per_minute == 5600.0
        assert config.annual_incidents == 120
        assert config.manual_mttr_minutes == 45.0
        assert config.llm_mttr_minutes == 12.0
        assert config.deeplog_mttr_seconds == 2.8
        assert config.annual_llm_triage_calls == 100_000
        assert config.cost_per_llm_call == 0.03
        assert config.sre_hourly_rate == 95.0
        assert config.baseline_false_alarms == 1683
        assert config.deeplog_false_alarms == 317
        assert config.discount_rate == 0.10
        assert config.analysis_years == 3
        assert config.risk_adjustment_pct == 0.05

    @pytest.mark.parametrize(
        "param,bad_val",
        [
            ("downtime_cost_per_minute", -1.0),
            ("annual_incidents", -5),
            ("manual_mttr_minutes", -10.0),
            ("llm_mttr_minutes", -0.5),
            ("deeplog_mttr_seconds", -2.8),
            ("annual_llm_triage_calls", -100),
            ("cost_per_llm_call", -0.01),
            ("sre_hourly_rate", -50.0),
            ("sre_hours_manual_per_incident", -1.0),
            ("sre_hours_deeplog_per_incident", -0.5),
            ("baseline_false_alarms", -10),
            ("deeplog_false_alarms", -5),
            ("sre_hours_per_false_alarm", -0.2),
            ("annual_working_hours_per_fte", 0.0),
            ("annual_working_hours_per_fte", -2080.0),
            ("implementation_cost", -1000.0),
            ("annual_maintenance_cost", -500.0),
            ("discount_rate", -1.0),
            ("discount_rate", -1.5),
            ("analysis_years", 0),
            ("analysis_years", -3),
            ("risk_adjustment_pct", -0.05),
            ("risk_adjustment_pct", 1.0),
            ("risk_adjustment_pct", 1.25),
        ],
    )
    def test_invalid_parameters_raise_value_error(self, param: str, bad_val: float):
        kwargs = {param: bad_val}
        with pytest.raises(ValueError):
            TEIConfig(**kwargs)


class TestMTTRComparison:
    """Verify MTTR calculations and speedup multipliers."""

    def test_default_mttr_metrics(self):
        calc = EnterpriseROICalculator()
        mttr = calc.compute_mttr_comparison()

        assert mttr.manual_mttr_minutes == 45.0
        assert mttr.llm_agent_mttr_minutes == 12.0
        assert mttr.deeplog_mttr_seconds == 2.8
        expected_deeplog_min = 2.8 / 60.0
        assert pytest.approx(mttr.deeplog_mttr_minutes, rel=1e-5) == expected_deeplog_min

        # Speedup vs Manual: 45 / (2.8 / 60) = 45 * 60 / 2.8 = 964.2857x
        assert pytest.approx(mttr.speedup_factor_vs_manual, rel=1e-4) == 964.2857
        # Speedup vs LLM: 12 / (2.8 / 60) = 12 * 60 / 2.8 = 257.1428x
        assert pytest.approx(mttr.speedup_factor_vs_llm, rel=1e-4) == 257.1428

        # Time saved
        assert pytest.approx(mttr.time_saved_vs_manual_minutes, rel=1e-5) == (45.0 - expected_deeplog_min)
        assert pytest.approx(mttr.time_saved_vs_llm_minutes, rel=1e-5) == (12.0 - expected_deeplog_min)

    def test_zero_deeplog_latency(self):
        config = TEIConfig(deeplog_mttr_seconds=0.0)
        calc = EnterpriseROICalculator(config)
        mttr = calc.compute_mttr_comparison()
        assert math.isinf(mttr.speedup_factor_vs_manual)
        assert mttr.deeplog_mttr_minutes == 0.0


class TestDowntimePrevention:
    """Verify downtime prevention calculations against Gartner Tier-1 standard."""

    def test_default_downtime_prevention(self):
        calc = EnterpriseROICalculator()
        res = calc.compute_downtime_prevention()

        assert res.cost_per_minute == 5600.0
        assert res.annual_incidents == 120
        # Manual downtime: 120 * 45 = 5,400 min
        assert res.annual_downtime_minutes_manual == 5400.0
        # Manual cost: 5400 * 5600 = $30,240,000
        assert res.annual_cost_manual == 30_240_000.0

        # LLM downtime: 120 * 12 = 1,440 min
        assert res.annual_downtime_minutes_llm == 1440.0
        # LLM cost: 1440 * 5600 = $8,064,000
        assert res.annual_cost_llm == 8_064_000.0

        # DeepLog downtime: 120 * (2.8 / 60) = 5.6 min
        assert pytest.approx(res.annual_downtime_minutes_deeplog, rel=1e-5) == 5.6
        # DeepLog cost: 5.6 * 5600 = $31,360.00
        assert pytest.approx(res.annual_cost_deeplog, rel=1e-5) == 31_360.0

        # Annual savings vs manual: 30,240,000 - 31,360 = $30,208,640.00
        assert pytest.approx(res.annual_savings_vs_manual, rel=1e-5) == 30_208_640.0
        # Annual savings vs LLM: 8,064,000 - 31,360 = $8,032,640.00
        assert pytest.approx(res.annual_savings_vs_llm, rel=1e-5) == 8_032_640.0

    def test_zero_incidents(self):
        config = TEIConfig(annual_incidents=0)
        calc = EnterpriseROICalculator(config)
        res = calc.compute_downtime_prevention()
        assert res.annual_cost_manual == 0.0
        assert res.annual_cost_deeplog == 0.0
        assert res.annual_savings_vs_manual == 0.0


class TestLLMCostSavings:
    """Verify cloud LLM API cost avoidance calculations."""

    def test_default_llm_savings(self):
        calc = EnterpriseROICalculator()
        res = calc.compute_llm_cost_savings()

        # 100,000 calls * $0.03 = $3,000.00
        assert res.annual_calls == 100_000
        assert res.cost_per_call == 0.03
        assert res.baseline_cloud_llm_cost == 3000.0
        assert res.deeplog_api_cost == 0.0
        assert res.annual_api_savings == 3000.0

    def test_scaled_enterprise_volume(self):
        config = TEIConfig(annual_llm_triage_calls=1_000_000, cost_per_llm_call=0.04)
        calc = EnterpriseROICalculator(config)
        res = calc.compute_llm_cost_savings()
        assert res.baseline_cloud_llm_cost == 40_000.0
        assert res.annual_api_savings == 40_000.0


class TestEngineerToilReduction:
    """Verify SRE toil reduction, false alarm reduction, and FTE savings."""

    def test_default_toil_savings(self):
        calc = EnterpriseROICalculator()
        res = calc.compute_toil_reduction()

        # Incident hours saved: 120 * (4.0 - 0.25) = 120 * 3.75 = 450.0 hrs
        assert res.incident_hours_saved == 450.0

        # False alarm reduction: 1683 -> 317 = 1366 delta
        assert res.baseline_false_alarms == 1683
        assert res.deeplog_false_alarms == 317
        assert pytest.approx(res.false_alarm_reduction_pct, rel=1e-3) == 81.1645
        # False alarm hours saved: 1366 * 0.5 hrs = 683.0 hrs
        assert res.false_alarm_hours_saved == 683.0

        # Total hours: 450 + 683 = 1133.0 hrs
        assert res.total_annual_hours_saved == 1133.0

        # FTE saved: 1133.0 / 2080.0 = 0.54471 FTE
        assert pytest.approx(res.fte_saved, rel=1e-4) == 0.54471

        # Labor savings: 1133 * $95 = $107,635.00
        assert res.annual_labor_savings == 107_635.0

    def test_zero_false_alarms(self):
        config = TEIConfig(baseline_false_alarms=0, deeplog_false_alarms=0)
        calc = EnterpriseROICalculator(config)
        res = calc.compute_toil_reduction()
        assert res.false_alarm_reduction_pct == 0.0
        assert res.false_alarm_hours_saved == 0.0
        assert res.total_annual_hours_saved == 450.0


class TestMultiYearCashFlowsAndFinancialMetrics:
    """Verify 3-year cash flows, NPV, IRR, and payback period calculations."""

    def test_default_tei_analysis(self):
        calc = EnterpriseROICalculator()
        res = calc.calculate()

        # Check gross annual benefits:
        # Downtime: $30,208,640 + LLM: $3,000 + Labor: $107,635 = $30,319,275
        expected_annual_gross = 30_208_640.0 + 3_000.0 + 107_635.0
        # Risk adjustment = 5%
        expected_annual_risk_benefit = expected_annual_gross * 0.95
        assert expected_annual_gross == 30_319_275.0
        assert pytest.approx(expected_annual_risk_benefit, rel=1e-5) == 28_803_311.25

        # Check cash flow structure
        assert len(res.cash_flows) == 4  # Year 0, 1, 2, 3
        y0 = res.cash_flows[0]
        assert y0.year == 0
        assert y0.gross_costs == 50_000.0
        assert y0.risk_adjusted_costs == 52_500.0  # 50,000 * 1.05
        assert y0.net_cash_flow == -52_500.0

        for y in res.cash_flows[1:]:
            assert y.gross_benefits == expected_annual_gross
            assert pytest.approx(y.risk_adjusted_benefits, rel=1e-5) == expected_annual_risk_benefit
            assert y.gross_costs == 12_000.0
            assert y.risk_adjusted_costs == 12_600.0  # 12,000 * 1.05
            assert pytest.approx(y.net_cash_flow, rel=1e-5) == (28_803_311.25 - 12_600.0)

        # Financial returns
        assert res.net_present_value > 70_000_000.0
        assert res.roi_percentage > 80_000.0
        assert res.internal_rate_of_return > 500.0  # Overwhelmingly positive IRR
        assert res.payback_period_months < 0.1  # Breakeven in days

    def test_convenience_helper(self):
        res1 = compute_enterprise_roi()
        res2 = EnterpriseROICalculator().calculate()
        assert pytest.approx(res1.net_present_value, rel=1e-5) == res2.net_present_value
        assert pytest.approx(res1.roi_percentage, rel=1e-5) == res2.roi_percentage

    def test_custom_cost_growth(self):
        config = TEIConfig(annual_cost_growth_rate=0.05, analysis_years=3)
        calc = EnterpriseROICalculator(config)
        res = calc.calculate()
        y1 = res.cash_flows[1]
        y2 = res.cash_flows[2]
        y3 = res.cash_flows[3]
        # y2 cost = 12000 * 1.05^1 = 12600 gross
        assert pytest.approx(y2.gross_costs, rel=1e-4) == 12_000.0 * 1.05
        assert pytest.approx(y3.gross_costs, rel=1e-4) == 12_000.0 * (1.05 ** 2)


class TestIRRAndPaybackEdgeCases:
    """Test boundary conditions for IRR and Payback algorithms."""

    def test_compute_irr_standard(self):
        # Initial: -100, Year 1: +110 -> IRR = 10% (0.10)
        flows = [-100.0, 110.0]
        irr = EnterpriseROICalculator.compute_irr(flows)
        assert pytest.approx(irr, rel=1e-4) == 0.10

    def test_compute_irr_multi_period(self):
        # Initial: -1000, Year 1: +400, Year 2: +400, Year 3: +400
        # Sum of discounted CF = 0 at r ~ 9.70%
        flows = [-1000.0, 400.0, 400.0, 400.0]
        irr = EnterpriseROICalculator.compute_irr(flows)
        assert pytest.approx(irr, rel=1e-3) == 0.09701

    def test_compute_irr_all_positive_or_negative(self):
        assert math.isnan(EnterpriseROICalculator.compute_irr([100.0, 200.0]))
        assert math.isnan(EnterpriseROICalculator.compute_irr([-100.0, -200.0]))
        assert math.isnan(EnterpriseROICalculator.compute_irr([]))

    def test_payback_instant(self):
        y0 = YearlyCashFlow(0, 0, 0, 0, 0, 0.0, 1.0, 0.0, 0.0, 0.0)
        y1 = YearlyCashFlow(1, 100, 100, 10, 10, 90.0, 0.9, 81.0, 90.0, 81.0)
        months = EnterpriseROICalculator.compute_payback_period_months([y0, y1])
        assert months == 0.0

    def test_payback_fractional(self):
        # -120 Year 0, Year 1 net +120 -> Payback exactly 12 months
        y0 = YearlyCashFlow(0, 0, 0, 120, 120, -120.0, 1.0, -120.0, -120.0, -120.0)
        y1 = YearlyCashFlow(1, 140, 140, 20, 20, 120.0, 0.9, 108.0, 0.0, -12.0)
        months = EnterpriseROICalculator.compute_payback_period_months([y0, y1])
        assert pytest.approx(months, rel=1e-4) == 12.0

        # -60 Year 0, Year 1 net +120 -> Payback 6 months
        y0_b = YearlyCashFlow(0, 0, 0, 60, 60, -60.0, 1.0, -60.0, -60.0, -60.0)
        y1_b = YearlyCashFlow(1, 140, 140, 20, 20, 120.0, 0.9, 108.0, 60.0, 48.0)
        months_b = EnterpriseROICalculator.compute_payback_period_months([y0_b, y1_b])
        assert pytest.approx(months_b, rel=1e-4) == 6.0


class TestExportFormats:
    """Verify JSON, dictionary, Markdown, and ASCII export capabilities."""

    def test_to_dict_and_to_json(self):
        res = compute_enterprise_roi()
        d = res.to_dict()
        assert isinstance(d, dict)
        assert "config" in d
        assert "mttr_comparison" in d
        assert "downtime_prevention" in d
        assert "llm_cost_savings" in d
        assert "toil_reduction" in d
        assert "cash_flows" in d
        assert "net_present_value" in d
        assert "roi_percentage" in d

        # JSON serialization
        json_str = res.to_json(indent=2)
        parsed = json.loads(json_str)
        assert parsed["config"]["downtime_cost_per_minute"] == 5600.0
        assert parsed["downtime_prevention"]["cost_per_minute"] == 5600.0

    def test_executive_summary_markdown(self):
        res = compute_enterprise_roi()
        md = res.to_executive_summary_table(format="markdown")
        assert "# DeepLog-Infinity Enterprise Total Economic Impact (TEI) Summary" in md
        assert "Net Present Value (NPV)" in md
        assert "Manual / Runbook Triage" in md
        assert "Generative LLM Agent (Cloud)" in md
        assert "DeepLog-Infinity (Autonomous Safe Control)" in md
        assert "$5,600/min" in md
        assert "Multi-Year Risk-Adjusted Cash Flows" in md

    def test_executive_summary_ascii(self):
        res = compute_enterprise_roi()
        ascii_table = res.to_executive_summary_table(format="ascii")
        assert "DEEPLOG-INFINITY ENTERPRISE TEI EXECUTIVE SUMMARY" in ascii_table
        assert "Net Present Value (NPV):" in ascii_table
        assert "Tier-1 Downtime Cost Avoidance:" in ascii_table
        assert "Speedup vs Manual:" in ascii_table
