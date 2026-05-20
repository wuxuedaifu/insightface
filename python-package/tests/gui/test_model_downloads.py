from insightface.gui.core.model_downloads import fallback_model_assets, local_model_status
from insightface.gui.core.paths import default_workspace


def test_default_gui_workspace_path():
    assert str(default_workspace()).endswith(".insightface/gui")


def test_fallback_model_assets_have_github_release_urls(tmp_path):
    assets = fallback_model_assets()
    names = {asset.name for asset in assets}
    assert {"buffalo_l.zip", "buffalo_s.zip", "antelopev2.zip"}.issubset(names)
    for asset in assets:
        assert asset.tag_name == "v0.7"
        assert asset.browser_download_url.startswith(
            "https://github.com/deepinsight/insightface/releases/download/v0.7/"
        )
    assert local_model_status(assets[0], tmp_path) == "not installed"
