"""Enterprise evaluation report generation."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Dict, Iterable

from .constants import COMMERCIAL_NOTICE, RESPONSIBLE_USE_NOTICE
from .models import EvaluationResult
from .utils import safe_json_dumps, timestamp_for_filename


def _metric_lines(metrics: Dict[str, Any]) -> Iterable[str]:
    for key, value in metrics.items():
        if isinstance(value, float):
            yield f"- {key}: {value:.4f}"
        else:
            yield f"- {key}: {value}"


def generate_markdown_report(result: EvaluationResult) -> str:
    threshold_recommendation = (
        f"{result.threshold_recommendation:.4f}" if result.threshold_recommendation is not None else "Not available"
    )
    raw_rows = result.raw_results[:50]
    raw_block = safe_json_dumps(raw_rows)
    return "\n".join(
        [
            "# InsightFace Enterprise Evaluation Report",
            "",
            "## 1. Executive Summary",
            f"- Scenario: {result.scenario}",
            f"- Created at: {result.created_at}",
            f"- License status: {result.license_status}",
            "",
            "## 2. Evaluation Scenario",
            result.scenario,
            "",
            "## 3. Dataset Summary",
            safe_json_dumps(result.dataset_summary),
            "",
            "## 4. Model and Runtime",
            f"- Model: {result.model_name}",
            f"- Provider: {result.provider}",
            f"- Threshold: {result.threshold:.4f}",
            "",
            "## 5. License Status",
            result.license_status,
            "",
            "## 6. Metrics",
            *list(_metric_lines(result.metrics)),
            "",
            "## 7. Threshold Recommendation",
            threshold_recommendation,
            "",
            "## 8. Error Analysis",
            safe_json_dumps(result.errors[:50]),
            "",
            "## 9. Latency and Hardware",
            safe_json_dumps(result.latency),
            "",
            "## 10. Deployment Considerations",
            COMMERCIAL_NOTICE,
            "",
            "## 11. Responsible Use and Compliance Notice",
            RESPONSIBLE_USE_NOTICE,
            "This report does not provide legal advice.",
            "",
            "## 12. Commercial Licensing Next Steps",
            (
                "Contact InsightFace for commercial model licensing, private model evaluation, "
                "SDK/API access, SLA, or custom training."
            ),
            "",
            "## 13. Appendix: Raw Results",
            "```json",
            raw_block,
            "```",
            "",
        ]
    )


def generate_html_report(result: EvaluationResult) -> str:
    markdown = generate_markdown_report(result)
    lines = []
    in_code = False
    for line in markdown.splitlines():
        if line.startswith("```"):
            lines.append("</pre>" if in_code else "<pre>")
            in_code = not in_code
        elif in_code:
            lines.append(html.escape(line))
        elif line.startswith("# "):
            lines.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("## "):
            lines.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("- "):
            lines.append(f"<p>{html.escape(line)}</p>")
        elif line.strip():
            lines.append(f"<p>{html.escape(line)}</p>")
        else:
            lines.append("")
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>InsightFace Enterprise Evaluation Report</title>"
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;"
        "max-width:1040px;margin:32px auto;line-height:1.55;color:#1f2937}"
        "h1,h2{color:#111827}pre{background:#f3f4f6;padding:16px;overflow:auto}"
        "p{margin:6px 0}</style></head><body>"
        + "\n".join(lines)
        + "</body></html>"
    )


def write_reports(result: EvaluationResult, report_dir: str | Path) -> Dict[str, str]:
    root = Path(report_dir)
    root.mkdir(parents=True, exist_ok=True)
    stem = f"insightface_evaluation_{timestamp_for_filename()}"
    markdown_path = root / f"{stem}.md"
    html_path = root / f"{stem}.html"
    markdown_path.write_text(generate_markdown_report(result), encoding="utf-8")
    html_path.write_text(generate_html_report(result), encoding="utf-8")
    result.report_path = str(markdown_path)
    paths = {"markdown": str(markdown_path), "html": str(html_path)}
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        pdf_path = root / f"{stem}.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        y = 760
        for line in generate_markdown_report(result).splitlines():
            if y < 40:
                c.showPage()
                y = 760
            c.drawString(40, y, line[:110])
            y -= 14
        c.save()
        paths["pdf"] = str(pdf_path)
    except Exception:
        pass
    return paths
