from insightface.model_zoo.retinaface import RetinaFace
from insightface.model_zoo.scrfd import SCRFD
from insightface.gui.core.face_engine import FaceEngine


def test_scrfd_accepts_multi_det_size_config():
    assert SCRFD._normalize_input_sizes([(128, 128), (640, 640)]) == [(128, 128), (640, 640)]
    assert SCRFD._normalize_input_sizes((320, 320)) == [(320, 320)]


def test_retinaface_accepts_multi_det_size_config():
    assert RetinaFace._normalize_input_sizes([(128, 128), (640, 640)]) == [(128, 128), (640, 640)]
    assert RetinaFace._normalize_input_sizes([1024, 1024]) == [(1024, 1024)]


def test_gui_routes_detection_outputs_to_scrfd():
    assert FaceEngine._is_scrfd_detection_outputs([object()] * 6) is True
    assert FaceEngine._is_scrfd_detection_outputs([object()] * 9) is True
    assert FaceEngine._is_scrfd_detection_outputs([object()] * 10) is True
    assert FaceEngine._is_scrfd_detection_outputs([object()] * 15) is True
    assert FaceEngine._is_scrfd_detection_outputs([object()]) is False


def test_gui_auto_detection_size_passes_multi_scale_to_detector():
    engine = FaceEngine(det_size=(0, 0))

    assert engine._detector_input_size() == [(128, 128), (640, 640)]
