import numpy as np

from src.preprocessing import PreprocessConfig, preprocess_for_inference


def test_preprocess_for_inference_returns_expected_shape(tmp_path):
    dummy_image = np.random.randint(0, 255, size=(320, 320, 3), dtype=np.uint8)
    config = PreprocessConfig(target_size=(640, 640))
    processed = preprocess_for_inference(dummy_image, config)
    assert processed.shape == (640, 640, 3)
    assert processed.dtype == np.float32
