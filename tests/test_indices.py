import numpy as np

from src.m03_indices.indices import calc_ndvi


def test_ndvi_basic():
    # simple 2x2 image with known values
    # pixel: R800=0.8, R670=0.2 => NDVI=(0.8-0.2)/(0.8+0.2)=0.6
    data = np.zeros((2, 2, 3), dtype=float)
    # wavelengths order: [670, 800, 550] chosen for test convenience
    wl = [670.0, 800.0, 550.0]
    data[:, :, 0] = 0.2  # 670
    data[:, :, 1] = 0.8  # 800
    ndvi = calc_ndvi(data, wl)
    assert np.allclose(ndvi, 0.6)
