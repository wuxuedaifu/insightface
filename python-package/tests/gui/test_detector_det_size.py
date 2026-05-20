from insightface.model_zoo.retinaface import RetinaFace
from insightface.model_zoo.scrfd import SCRFD


def test_scrfd_accepts_multi_det_size_config():
    assert SCRFD._normalize_input_sizes([(128, 128), (640, 640)]) == [(128, 128), (640, 640)]
    assert SCRFD._normalize_input_sizes((320, 320)) == [(320, 320)]


def test_retinaface_accepts_multi_det_size_config():
    assert RetinaFace._normalize_input_sizes([(128, 128), (640, 640)]) == [(128, 128), (640, 640)]
    assert RetinaFace._normalize_input_sizes([1024, 1024]) == [(1024, 1024)]
