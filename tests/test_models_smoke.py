import os

import numpy as np

from ml.models import load_single_model


def test_diabetes_model_predict():
    m = load_single_model(os.getcwd(), "diabetes")
    assert m is not None
    X = np.array([[0, 120, 70, 20, 80, 24.0, 0.5, 30]], dtype=float)
    y = m.predict(X)
    assert y.shape == (1,)


def test_heart_model_predict():
    m = load_single_model(os.getcwd(), "heart")
    assert m is not None
    X = np.array([[40, 1, 0, 120, 200, 0, 0, 150, 0, 1.0, 1, 0, 0]], dtype=float)
    y = m.predict(X)
    assert y.shape == (1,)


def test_parkinsons_model_predict():
    m = load_single_model(os.getcwd(), "parkinsons")
    assert m is not None
    X = np.array([[
        119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037, 0.00554, 0.01109,
        0.04374, 0.426, 0.02182, 0.0313, 0.02971, 0.06545, 0.02211, 21.033,
        0.414783, 0.815285, -4.813031, 0.266, 2.301442, 0.284654
    ]], dtype=float)
    y = m.predict(X)
    assert y.shape == (1,)
