"""Single-page enterprise 1:1 and 1:N evaluation workflow."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QEvent, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..core.evaluation import (
    MULTI_FACE_REQUIRE_ONE,
    MULTI_FACE_SKIP,
    MULTI_FACE_USE_CENTERED_LARGEST,
    MULTI_FACE_USE_LARGEST,
    run_identity_identification_evaluation,
    run_identity_verification_evaluation,
    validate_enterprise_dataset,
)
from ..core.i18n import apply_translations, effective_language, tr
from ..core.reporting import write_reports
from ..widgets.drop_input import DropInput
from .base import BasePage


DATASET_RULES_TEXT = """Enterprise Evaluation Dataset Rules

The Enterprise Evaluation page supports local 1:1 verification and 1:N identification evaluation from identity folders. No images, embeddings, or reports are uploaded automatically.

1:1 Verification

Auto Split enabled:

dataset_1v1/
  0001__Alice/
    gallery.jpg
    img002.jpg
    img003.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Each subfolder is one identity. A file whose name contains "gallery" is selected as that identity's gallery image. If no such file exists, the first sorted image is selected as gallery. All other images are probes. Evaluation compares every probe against every identity gallery, so each trial is probe vs gallery.

Auto Split disabled:

dataset_1v1/
  0001__Alice/
    img001.jpg
    img002.jpg
  0002__Bob/
    img001.jpg
    img002.jpg

Every image is treated as a probe. Evaluation runs full pairwise probe-vs-probe comparisons across all identity folders.

1:N Identification

Auto Split enabled:

dataset/
  identities/
    0001__Alice/
      gallery.jpg
      img002.jpg
      img003.jpg
    0002__Bob/
      img001.jpg
      img002.jpg

The identities/ folder is preferred when present. If it is not present, the selected dataset root may contain identity folders directly. Gallery selection follows the same Auto Split rule as 1:1.

Auto Split disabled:

dataset_1n/
  gallery/
    0001__Alice/
      enroll_001.jpg
      enroll_002.jpg
    0002__Bob/
      enroll_001.jpg
  probe/
    0001__Alice/
      test_001.jpg
      test_002.jpg
    0002__Bob/
      test_001.jpg
  unknown/
    unknown_001.jpg
    unknown_002.jpg

gallery/ and probe/ are required. unknown/ is optional. 1:N evaluation always requires gallery images.

Validation Checks

- The selected root exists and matches the selected mode.
- Required gallery and probe folders/images exist.
- Auto Split can create valid gallery/probe sets.
- 1:1 can generate both positive and negative pairs.
- 1:N probe identities have valid gallery coverage.
- Each image can be read.
- Detected face count follows the selected multi-face handling policy.

Multi-face Handling

- Require exactly one face: validation fails on multi-face images.
- Use largest face: evaluation uses the largest detected face.
- Use largest centered face: evaluation uses area - center_distance^2 * 2.0.
- Mark as skip: multi-face images are skipped; skipped gallery images can make validation fail.

Report Outputs

1:1 reports include best cosine threshold accuracy, FAR/FRR, TAR@FAR, threshold recommendations, latency, and raw result examples.

1:N reports include Top1, TAR@FAR, threshold recommendations, latency, and raw result examples.
"""

DATASET_RULES_LOCAL_SUMMARY = {
    "zh": """本页用于本地 1:1 验证和 1:N 识别评测。数据不会自动上传。

1:1 自动切分：每个身份文件夹选择一张 gallery 图，其余图片作为 probe，并执行 probe vs gallery 评测。
1:1 非自动切分：每张图片都作为 probe，生成完整的两两比对。

1:N 自动切分：可使用 identities/<身份文件夹> 或直接使用身份文件夹，每个身份必须能生成 gallery 和 probe。
1:N 非自动切分：必须包含 gallery/<identity> 和 probe/<identity>；unknown/ 可选，用于提供 FAR 的冒名负样本。

核验会检查目录结构、gallery/probe 可用性、正负样本是否足够、图片可读取性，以及多人脸处理策略。""",
    "ja": """このページでは、ローカルの 1:1 検証と 1:N 識別評価を実行します。データは自動アップロードされません。

1:1 自動分割では、各IDから gallery 画像を1枚選び、残りを probe として probe vs gallery 評価を行います。
1:1 自動分割なしでは、すべての画像を probe として総当たり比較を行います。

1:N 自動分割では identities/<IDフォルダ> またはIDフォルダ直下を利用でき、各IDに gallery と probe が必要です。
1:N 構造化形式では gallery/<identity> と probe/<identity> が必須で、unknown/ は FAR 用のインポスター負例として利用できます。

検証では、フォルダ構造、gallery/probe、正負ペア、画像読み込み、複数顔処理ポリシーを確認します。""",
    "ko": """이 페이지는 로컬 1:1 검증 및 1:N 식별 평가를 수행합니다. 데이터는 자동 업로드되지 않습니다.

1:1 자동 분할은 각 ID에서 gallery 이미지 1장을 선택하고 나머지를 probe로 사용해 probe vs gallery 평가를 수행합니다.
1:1 자동 분할 해제 시 모든 이미지를 probe로 보고 전체 쌍 비교를 생성합니다.

1:N 자동 분할은 identities/<ID 폴더> 또는 ID 폴더를 직접 사용할 수 있으며 각 ID에는 gallery와 probe가 필요합니다.
1:N 구조화 형식은 gallery/<identity>와 probe/<identity>가 필수이며 unknown/은 FAR용 impostor 음성 샘플로 사용됩니다.

검증 단계는 폴더 구조, gallery/probe 사용 가능 여부, 양/음성 샘플, 이미지 읽기, 다중 얼굴 처리 정책을 확인합니다.""",
    "es": """Esta página ejecuta evaluaciones locales 1:1 y 1:N. Los datos no se suben automáticamente.

En 1:1 con división automática, se selecciona una imagen gallery por identidad y el resto se usa como probe para evaluar probe vs gallery.
En 1:1 sin división automática, todas las imágenes son probes y se generan comparaciones por pares.

En 1:N con división automática, puede usar identities/<carpetas de identidad> o carpetas de identidad directamente; cada identidad requiere gallery y probe.
En 1:N estructurado, gallery/<identity> y probe/<identity> son obligatorios; unknown/ es opcional para negativos impostores de FAR.

La validación comprueba estructura, disponibilidad de gallery/probe, muestras positivas/negativas, lectura de imágenes y política de varias caras.""",
    "fr": """Cette page exécute des évaluations locales 1:1 et 1:N. Les données ne sont pas téléversées automatiquement.

En 1:1 avec découpage automatique, une image gallery est choisie par identité et les autres servent de probe pour l’évaluation probe vs gallery.
En 1:1 sans découpage automatique, toutes les images sont des probes et des comparaisons par paires sont générées.

En 1:N avec découpage automatique, utilisez identities/<dossiers d’identité> ou les dossiers d’identité directement ; chaque identité doit fournir gallery et probe.
En 1:N structuré, gallery/<identity> et probe/<identity> sont obligatoires ; unknown/ est optionnel pour les négatifs imposteurs FAR.

La validation vérifie la structure, gallery/probe, les échantillons positifs/négatifs, la lecture des images et la politique multi-visage.""",
    "de": """Diese Seite führt lokale 1:1- und 1:N-Evaluierungen aus. Daten werden nicht automatisch hochgeladen.

Bei 1:1 mit automatischer Aufteilung wird pro Identität ein gallery-Bild gewählt; die übrigen Bilder dienen als probe für probe-vs-gallery.
Bei 1:1 ohne automatische Aufteilung werden alle Bilder als probe behandelt und paarweise verglichen.

Bei 1:N mit automatischer Aufteilung können identities/<Identitätsordner> oder direkte Identitätsordner verwendet werden; jede Identität benötigt gallery und probe.
Bei strukturiertem 1:N sind gallery/<identity> und probe/<identity> erforderlich; unknown/ ist optional für FAR-Impostor-Negativbeispiele.

Die Validierung prüft Struktur, gallery/probe, positive/negative Beispiele, Lesbarkeit der Bilder und die Mehrgesichter-Strategie.""",
    "pt": """Esta página executa avaliações locais 1:1 e 1:N. Os dados não são carregados automaticamente.

Em 1:1 com divisão automática, é escolhida uma imagem gallery por identidade e as restantes são probe para avaliação probe vs gallery.
Em 1:1 sem divisão automática, todas as imagens são probes e são geradas comparações par a par.

Em 1:N com divisão automática, use identities/<pastas de identidade> ou pastas de identidade diretamente; cada identidade requer gallery e probe.
Em 1:N estruturado, gallery/<identity> e probe/<identity> são obrigatórios; unknown/ é opcional para negativos impostores de FAR.

A validação verifica estrutura, gallery/probe, amostras positivas/negativas, leitura das imagens e política de múltiplos rostos.""",
    "ru": """Эта страница выполняет локальную оценку 1:1 и 1:N. Данные не загружаются автоматически.

В 1:1 с авторазделением для каждой идентичности выбирается одно изображение gallery, остальные используются как probe для probe vs gallery.
В 1:1 без авторазделения все изображения считаются probe и сравниваются попарно.

В 1:N с авторазделением можно использовать identities/<папки идентичностей> или папки идентичностей напрямую; каждой идентичности нужны gallery и probe.
В структурированном 1:N обязательны gallery/<identity> и probe/<identity>; unknown/ опционален для impostor-негативов FAR.

Валидация проверяет структуру, gallery/probe, наличие положительных и отрицательных примеров, чтение изображений и политику обработки нескольких лиц.""",
}


def dataset_rules_text(language: str | None) -> str:
    summary = DATASET_RULES_LOCAL_SUMMARY.get(effective_language(language))
    return f"{summary}\n\n---\n\n{DATASET_RULES_TEXT}" if summary else DATASET_RULES_TEXT


class DatasetRulesDialog(QDialog):
    def __init__(self, language: str | None = None, parent=None):
        if parent is None and isinstance(language, QWidget):
            parent = language
            language = getattr(getattr(parent, "context", None), "config", None)
            language = getattr(language, "ui_language", None)
        super().__init__(parent)
        self.setWindowTitle("Evaluation Dataset Rules")
        self.resize(820, 680)
        layout = QVBoxLayout(self)
        intro = QLabel("Prepare local identity folders using one of the supported layouts below.")
        intro.setWordWrap(True)
        intro.setProperty("role", "muted")
        layout.addWidget(intro)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(dataset_rules_text(language))
        layout.addWidget(text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        apply_translations(self, language)


class EnterpriseEvalPage(BasePage):
    def __init__(self, context, parent=None):
        super().__init__(
            context,
            "Enterprise Evaluation",
            "Run local 1:1 verification or 1:N identification evaluation from identity folders and export procurement-ready reports.",
            parent,
        )
        self.eval_mode = QComboBox()
        self.eval_mode.setProperty("i18nItems", True)
        self.eval_mode.addItem("1:1 Verification", "1:1 Verification")
        self.eval_mode.addItem("1:N Identification", "1:N Identification")
        self.eval_mode.currentIndexChanged.connect(self._update_instructions)
        self.auto_split = QCheckBox("Auto Split")
        self.auto_split.setChecked(True)
        self.auto_split.setToolTip(
            "Automatically select each identity's gallery image from a file containing 'gallery', or the first sorted image."
        )
        self.auto_split.stateChanged.connect(self._update_instructions)
        self.multi_face_policy = QComboBox()
        self.multi_face_policy.setProperty("i18nItems", True)
        self.multi_face_policy.addItem("Require exactly one face", MULTI_FACE_REQUIRE_ONE)
        self.multi_face_policy.addItem("Use largest face", MULTI_FACE_USE_LARGEST)
        self.multi_face_policy.addItem("Use largest centered face", MULTI_FACE_USE_CENTERED_LARGEST)
        self.multi_face_policy.addItem("Mark as skip", MULTI_FACE_SKIP)
        self.multi_face_policy.setToolTip(
            "Choose how evaluation handles images where more than one face is detected."
        )
        self.multi_face_policy.currentIndexChanged.connect(self._update_instructions)
        self.dataset_root = DropInput("Identity Folders / Dataset Root", mode="folder")
        self.dataset_root.pathsChanged.connect(self._invalidate_validation)
        self.mode_summary = QLabel()
        self.mode_summary.setWordWrap(True)
        self.mode_summary.setProperty("role", "muted")
        self.validation_status = QLabel("Validate the dataset before running an evaluation.")
        self.validation_status.setWordWrap(True)
        self.validation_status.setProperty("role", "muted")

        workspace = QWidget()
        workspace_layout = QGridLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setHorizontalSpacing(14)
        workspace_layout.setVerticalSpacing(12)

        setup_card, setup_layout = self._card("Evaluation Setup")
        setup_layout.addWidget(self._field_row("Evaluation mode", self.eval_mode))
        setup_layout.addWidget(self._field_row("Data split", self.auto_split))
        setup_layout.addWidget(self._field_row("Multi-face handling", self.multi_face_policy))
        self.help_button = self.button("Dataset Rules", self.show_dataset_rules)
        self.help_button.setToolTip("Open the detailed identity folder rules for 1:1, 1:N, and Auto Split.")
        setup_layout.addWidget(self.help_button, alignment=Qt.AlignLeft)
        setup_layout.addStretch(1)

        data_card, data_layout = self._card("Dataset")
        data_card.setAcceptDrops(True)
        data_card.installEventFilter(self)
        self.dataset_drop_card = data_card
        data_layout.addWidget(self.dataset_root)
        self.mode_summary.setAcceptDrops(True)
        self.mode_summary.installEventFilter(self)
        data_layout.addWidget(self.mode_summary)
        data_layout.addStretch(1)

        actions_card, actions_layout = self._card("Run")
        actions_hint = QLabel("Validate the selected dataset and configuration before running evaluation.")
        actions_hint.setWordWrap(True)
        actions_hint.setProperty("role", "muted")
        actions_layout.addWidget(actions_hint)
        actions_layout.addWidget(self.validation_status)
        button_row = QHBoxLayout()
        self.validate_button = self.button("Validate Dataset", self.validate_dataset)
        self.run_button = self.button("Run Evaluation", self.run, enabled=False)
        self.export_pdf_button = self.button("Export PDF", self.export_pdf)
        self.open_reports_button = self.button("Open Report Folder", self.open_report_folder)
        button_row.addWidget(self.validate_button)
        button_row.addWidget(self.run_button)
        button_row.addWidget(self.export_pdf_button)
        button_row.addWidget(self.open_reports_button)
        button_row.addStretch(1)
        actions_layout.addLayout(button_row)
        actions_layout.addStretch(1)

        workspace_layout.addWidget(setup_card, 0, 0)
        workspace_layout.addWidget(data_card, 0, 1)
        workspace_layout.addWidget(actions_card, 1, 0, 1, 2)
        workspace_layout.setColumnStretch(0, 0)
        workspace_layout.setColumnStretch(1, 1)
        self.content.addWidget(workspace)

        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setMinimumHeight(260)
        self.content.addWidget(self.output, 1)
        self.last_result = None
        self.last_report_paths: dict[str, str] = {}
        self.validation_result = None
        self._validated_signature: tuple[str, bool, str, str] | None = None
        self._update_instructions()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        dataset_targets = {
            getattr(self, "dataset_drop_card", None),
            getattr(self, "mode_summary", None),
        }
        if watched in dataset_targets:
            if event.type() in (QEvent.DragEnter, QEvent.DragMove):
                if self._dataset_drop_path(event):
                    self._set_dataset_drag_active(True)
                    event.acceptProposedAction()
                    return True
                event.ignore()
                return True
            if event.type() == QEvent.DragLeave:
                self._set_dataset_drag_active(False)
                return False
            if event.type() == QEvent.Drop:
                path = self._dataset_drop_path(event)
                self._set_dataset_drag_active(False)
                if path:
                    self.dataset_root.set_path(path)
                    self.set_status(f"Dataset root selected: {path}")
                    event.acceptProposedAction()
                    return True
                event.ignore()
                return True
        return super().eventFilter(watched, event)

    def _dataset_drop_path(self, event) -> str:
        mime = event.mimeData()
        if not mime.hasUrls():
            return ""
        for url in mime.urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile()).expanduser()
            if path.is_dir():
                return str(path)
        return ""

    def _set_dataset_drag_active(self, active: bool) -> None:
        if hasattr(self.dataset_root, "_set_property"):
            self.dataset_root._set_property("dragActive", bool(active))

    def _update_instructions(self) -> None:
        language = self.context.config.ui_language
        mode = self._eval_mode()
        auto_split = self.auto_split.isChecked()
        policy = tr(self.multi_face_policy.currentText(), language)
        if mode.startswith("1:1"):
            split_text = (
                "Auto Split selects one gallery image per identity and compares every probe against every identity gallery."
                if auto_split
                else "Auto Split is off, so every image is treated as a probe and full pairwise probe-vs-probe comparisons are generated."
            )
            text = f"1:1 verification expects one identity per subfolder. {split_text}"
        elif auto_split:
            text = (
                "1:N Auto Split can use identities/<identity folders> or identity folders directly. "
                "Each identity must provide a gallery image and at least one probe."
            )
        else:
            text = (
                "1:N structured evaluation requires gallery/<identity> and probe/<identity>. "
                "unknown/ is optional and contributes impostor scores for FAR thresholds."
            )
        self.mode_summary.setText(
            f"{tr(text, language)} {tr('Multi-face handling', language)}: {policy}. "
            f"{tr('Open Dataset Rules for detailed folder examples.', language)}"
        )
        self._invalidate_validation()

    def _card(self, title: str) -> tuple[QFrame, QVBoxLayout]:
        frame = QFrame()
        frame.setObjectName("enterpriseCard")
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)
        layout.setSpacing(10)
        heading = QLabel(title)
        heading.setObjectName("enterpriseCardTitle")
        heading.setStyleSheet("font-size: 15px; font-weight: 700;")
        layout.addWidget(heading)
        return frame, layout

    def _field_row(self, label: str, widget) -> QWidget:
        row = QWidget()
        layout = QGridLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        caption = QLabel(label)
        caption.setMinimumWidth(150)
        caption.setProperty("role", "secondary")
        layout.addWidget(caption, 0, 0, Qt.AlignTop)
        layout.addWidget(widget, 0, 1)
        layout.setColumnStretch(1, 1)
        return row

    def show_dataset_rules(self) -> None:
        dialog = DatasetRulesDialog(self.context.config.ui_language, self)
        dialog.exec()

    def _current_signature(self) -> tuple[str, bool, str, str]:
        return (
            self._eval_mode(),
            self.auto_split.isChecked(),
            self._multi_face_policy(),
            self.dataset_root.path().strip(),
        )

    def _eval_mode(self) -> str:
        return str(self.eval_mode.currentData() or "1:1 Verification")

    def _multi_face_policy(self) -> str:
        return str(self.multi_face_policy.currentData() or MULTI_FACE_REQUIRE_ONE)

    def _invalidate_validation(self, *args) -> None:
        del args
        self._validated_signature = None
        if hasattr(self, "run_button"):
            self.run_button.setEnabled(False)
        if hasattr(self, "validation_status"):
            self.validation_status.setText(
                tr("Dataset validation is required before running an evaluation.", self.context.config.ui_language)
            )

    def validate_dataset(self) -> None:
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        root = self.dataset_root.path().strip()
        if not root:
            self.show_error("Please select identity folders or a dataset root.")
            return
        signature = self._current_signature()
        mode = self._eval_mode()
        auto_split = self.auto_split.isChecked()
        multi_face_policy = self._multi_face_policy()

        def task(progress=None, is_cancelled=None):
            return validate_enterprise_dataset(
                root,
                mode,
                auto_split,
                self.context.engine,
                multi_face_policy=multi_face_policy,
                progress_callback=progress,
                cancel_callback=is_cancelled,
            )

        def done(result):
            self.validation_result = result
            self.output.setPlainText(self._validation_text(result))
            if result.ok:
                self._validated_signature = signature
                self.run_button.setEnabled(True)
                self.validation_status.setText(
                    tr("Dataset validation passed. Evaluation is ready to run.", self.context.config.ui_language)
                )
                self.set_status("Dataset validation passed.")
            else:
                self._validated_signature = None
                self.run_button.setEnabled(False)
                self.validation_status.setText(
                    tr(
                        "Dataset validation failed. Fix the listed issues before running evaluation.",
                        self.context.config.ui_language,
                    )
                )
                self.set_status("Dataset validation failed.")

        self.run_task("Dataset validation", task, done)

    def run(self) -> None:
        if not self.context.engine.is_loaded():
            self.show_error("Model is not loaded. Please open Models.")
            return
        root = self.dataset_root.path().strip()
        if not root:
            self.show_error("Please select identity folders or a dataset root.")
            return
        if self._validated_signature != self._current_signature():
            self.show_error("Please validate this dataset and configuration before running evaluation.")
            return
        mode = self._eval_mode()
        auto_split = self.auto_split.isChecked()
        multi_face_policy = self._multi_face_policy()

        def task(progress=None, is_cancelled=None):
            if mode.startswith("1:1"):
                return run_identity_verification_evaluation(
                    root,
                    self.context.engine,
                    auto_split=auto_split,
                    multi_face_policy=multi_face_policy,
                    license_status=self.context.config.license_status,
                    progress_callback=progress,
                    cancel_callback=is_cancelled,
                )
            return run_identity_identification_evaluation(
                root,
                self.context.engine,
                auto_split=auto_split,
                multi_face_policy=multi_face_policy,
                license_status=self.context.config.license_status,
                progress_callback=progress,
                cancel_callback=is_cancelled,
            )

        def done(result):
            self.last_result = result
            self.last_report_paths = write_reports(result, self.context.config.report_dir)
            report_path = self.last_report_paths.get("markdown", "")
            self.context.storage.save_evaluation_run(
                result.scenario,
                result.model_name,
                result.provider,
                result.threshold,
                result.dataset_summary,
                result.metrics,
                result.latency,
                report_path,
                created_at=result.created_at,
            )
            self.output.setPlainText(self._summary_text(result, self.last_report_paths))
            pdf_path = self.last_report_paths.get("pdf")
            self.set_status(
                f"Evaluation complete. PDF report: {pdf_path}" if pdf_path else f"Evaluation complete. Report: {report_path}"
            )

        self.run_task("Enterprise evaluation", task, done)

    def _validation_text(self, result) -> str:
        lines = [
            "Dataset Validation",
            f"Status: {'PASSED' if result.ok else 'FAILED'}",
            f"Mode: {result.mode}",
            f"Auto Split: {result.auto_split}",
            f"Multi-face policy: {result.multi_face_policy}",
            f"Root: {result.root}",
            "",
            "Summary:",
        ]
        for key, value in result.summary.items():
            lines.append(f"{key}: {value}")
        if result.errors:
            lines.extend(["", f"Errors: {len(result.errors)}"])
            for row in result.errors[:50]:
                lines.append(str(row))
            if len(result.errors) > 50:
                lines.append(f"... {len(result.errors) - 50} more errors")
        if result.warnings:
            lines.extend(["", f"Warnings: {len(result.warnings)}"])
            for row in result.warnings[:50]:
                lines.append(str(row))
            if len(result.warnings) > 50:
                lines.append(f"... {len(result.warnings) - 50} more warnings")
        return "\n".join(lines)

    def _summary_text(self, result, paths: dict[str, str]) -> str:
        lines = [
            f"Scenario: {result.scenario}",
            f"Dataset: {result.dataset_summary}",
            f"Report Markdown: {paths.get('markdown', 'not exported')}",
            f"Report HTML: {paths.get('html', 'not exported')}",
            f"Report PDF: {paths.get('pdf', 'PDF unavailable; install reportlab')}",
            "",
            "Metrics:",
        ]
        for key, value in result.metrics.items():
            if isinstance(value, float):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")
        if result.errors:
            lines.extend(["", f"Errors: {len(result.errors)}", "First errors:"])
            for row in result.errors[:10]:
                lines.append(str(row))
        return "\n".join(lines)

    def export_pdf(self) -> None:
        if self.last_result is None:
            self.show_error("Run an evaluation first.")
            return
        self.last_report_paths = write_reports(self.last_result, self.context.config.report_dir)
        pdf_path = self.last_report_paths.get("pdf")
        if not pdf_path:
            self.show_error("PDF export is unavailable. Please install reportlab.")
            return
        target, _ = QFileDialog.getSaveFileName(
            self,
            "Export PDF Report",
            str(Path(self.context.config.report_dir) / Path(pdf_path).name),
            "PDF (*.pdf)",
        )
        if target:
            shutil.copyfile(pdf_path, target)
            self.set_status(f"PDF report exported to {target}")
        else:
            self.set_status(f"PDF report exported to {pdf_path}")

    def open_report_folder(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self.context.config.report_dir))))
