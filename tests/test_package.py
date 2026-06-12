import importlib


def test_package_imports() -> None:
    importlib.import_module("credit_risk_training")
