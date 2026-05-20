"""Enterprise evaluation page."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import QComboBox, QFormLayout, QTextEdit

from ..core.evaluation import run_identification_evaluation, run_kyc_pairs_evaluation
from ..core.reporting import write_reports
from ..widgets.drop_input import DropInput
from .base import BasePage


class EnterpriseEvalPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(context, "Enterprise Evaluation", "No-code local model evaluation for KYC, access control, attendance, media search, video search, and face swap.", parent)
        self.scenario = QComboBox()
        self.scenario.addItems([
            "KYC / 1:1 Verification",
            "Access Control / 1:N Identification",
            "Attendance / Check-in Demo",
            "Photo Library / Media Search",
            "Video Person Search",
            "Face Swap Evaluation",
        ])
        csv_filter = "CSV (*.csv);;All Files (*)"
        self.pairs_csv = DropInput("Pairs CSV", extensions=[".csv"], dialog_filter=csv_filter)
        self.gallery_folder = DropInput("Gallery Folder", mode="folder")
        self.probe_folder = DropInput("Probe Folder", mode="folder")
        self.gt_csv = DropInput("Ground Truth CSV", extensions=[".csv"], dialog_filter=csv_filter)
        form = QFormLayout()
        form.addRow("Business scenario", self.scenario)
        form.addRow("Pairs CSV", self.pairs_csv)
        form.addRow("Gallery folder", self.gallery_folder)
        form.addRow("Probe folder", self.probe_folder)
        form.addRow("Ground truth CSV", self.gt_csv)
        self.content.addLayout(form)
        self.content.addWidget(self.row(self.button("Run Evaluation", self.run), self.button("Export Report", self.export_report), self.button("Open Report Folder", self.open_report_folder)))
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.content.addWidget(self.output, 1)
        self.last_result = None

    def run(self) -> None:
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        scenario = self.scenario.currentText()
        pairs_csv = self.pairs_csv.path().strip()
        gallery_folder = self.gallery_folder.path().strip()
        probe_folder = self.probe_folder.path().strip()
        gt_csv = self.gt_csv.path().strip()
        if scenario == "KYC / 1:1 Verification" and not pairs_csv:
            self.show_error("Please select a pairs CSV with image1_path,image2_path,label.")
            return
        if scenario == "Access Control / 1:N Identification" and (not gallery_folder or not probe_folder):
            self.show_error("Please select gallery and probe folders.")
            return
        if scenario not in ["KYC / 1:1 Verification", "Access Control / 1:N Identification"]:
            self.output.setPlainText("This scenario is available as a guided placeholder in v1.0. Use the dedicated page for this workflow or choose KYC / 1:N evaluation.")
            return

        def task(progress=None, is_cancelled=None):
            if scenario == "KYC / 1:1 Verification":
                return run_kyc_pairs_evaluation(pairs_csv, self.context.engine, threshold=self.context.config.recognition_threshold, license_status=self.context.config.license_status, progress_callback=progress, cancel_callback=is_cancelled)
            return run_identification_evaluation(gallery_folder, probe_folder, self.context.engine, threshold=self.context.config.recognition_threshold, ground_truth_csv=gt_csv or None, license_status=self.context.config.license_status, progress_callback=progress, cancel_callback=is_cancelled)

        def done(result):
            self.last_result = result
            paths = write_reports(result, self.context.config.report_dir)
            self.context.storage.save_evaluation_run(result.scenario, result.model_name, result.provider, result.threshold, result.dataset_summary, result.metrics, result.latency, paths["markdown"], created_at=result.created_at)
            self.output.setPlainText("\n".join([f"Scenario: {result.scenario}", f"Report: {paths['markdown']}", "Metrics:", *[f"{k}: {v}" for k, v in result.metrics.items()]]))
            self.set_status(f"Evaluation complete. Report exported to {paths['markdown']}")

        self.run_task("Enterprise evaluation", task, done)

    def export_report(self) -> None:
        if self.last_result is None:
            self.show_error("Run an evaluation first.")
            return
        paths = write_reports(self.last_result, self.context.config.report_dir)
        self.set_status(f"Report exported to {paths['markdown']}")

    def open_report_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.context.config.report_dir))))
