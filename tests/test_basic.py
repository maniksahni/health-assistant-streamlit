import os
import re


def test_runtime_pin_exists():
    assert os.path.exists("runtime.txt")
    with open("runtime.txt") as f:
        content = f.read().strip()
    assert content.startswith("3.11") or content.startswith("python-3.11")


def test_requirements_versions():
    assert os.path.exists("requirements.txt")
    txt = open("requirements.txt").read()
    assert "numpy==1.26.4" in txt
    assert re.search(r"scikit-learn==1\.[45]", txt)


def test_model_files_exist():
    for name in ["diabetes_model.sav", "heart_disease_model.sav", "parkinsons_model.sav"]:
        assert os.path.exists(os.path.join("saved_models", name))
