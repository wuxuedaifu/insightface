import pytest
import os


def test_main_window_smoke(tmp_path):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    PySide6 = pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from insightface.gui.app import StudioContext
    from insightface.gui.core.config import AppConfig
    from insightface.gui.core.face_engine import FaceEngine
    from insightface.gui.core.navigation import AppMode
    from insightface.gui.core.storage import Storage
    from insightface.gui.dialogs.license_dialog import LicenseDialog
    from insightface.gui.dialogs.model_manager_dialog import ModelManagerDialog
    from insightface.gui.dialogs.settings_dialog import SettingsDialog
    from insightface.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication([])
    cfg = AppConfig(workspace_path=str(tmp_path), auto_load_model=False, safe_mode=True)
    storage = Storage(cfg.database_path)
    engine = FaceEngine(model_name=cfg.model_name)
    window = MainWindow(StudioContext(cfg, True, storage, engine, str(tmp_path / "app.log")))
    window.show()
    for mode in AppMode:
        window.change_mode(mode)
        assert window.sidebar_list.count() > 0
        if mode in {AppMode.FACE_VERIFICATION, AppMode.ALBUM_MANAGEMENT, AppMode.FACE_SWAP}:
            assert not window.sidebar.isVisible()
        else:
            assert window.sidebar.isVisible()
        sidebar_titles = [window.sidebar_list.item(i).text() for i in range(window.sidebar_list.count())]
        assert "Settings" not in sidebar_titles
        assert "Model Settings" not in sidebar_titles
        assert "Model Downloads" not in sidebar_titles
        assert "License Center" not in sidebar_titles
    settings_dialog = SettingsDialog(window.context, window)
    assert hasattr(settings_dialog, "theme")
    assert not hasattr(settings_dialog, "workspace")
    assert not hasattr(settings_dialog, "default_mode")
    dialogs = [settings_dialog, ModelManagerDialog(window.context, window), LicenseDialog(window.context, window)]
    for dialog in dialogs:
        dialog.close()
    window.close()
    assert window.windowTitle().startswith("InsightFace Evaluation Studio")
