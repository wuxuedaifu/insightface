from insightface.gui.core.models import EvaluationResult
from insightface.gui.core.reporting import generate_html_report, generate_markdown_report


def test_reporting_markdown_html():
    result = EvaluationResult(
        scenario="KYC / 1:1 Verification",
        model_name="buffalo_l",
        provider="CPU",
        threshold=0.5,
        dataset_summary={"total_pairs": 2},
        metrics={"accuracy": 1.0},
        errors=[],
        latency={"average_ms": 1.2},
        license_status="Research / Non-commercial",
        created_at="2026-05-19T00:00:00Z",
    )
    md = generate_markdown_report(result)
    html = generate_html_report(result)
    assert "InsightFace Enterprise Evaluation Report" in md
    assert "Commercial Licensing Next Steps" in md
    assert "<html" in html
