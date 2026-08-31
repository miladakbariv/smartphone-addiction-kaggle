from src import add_crossfit_exact_te, add_structural_features


def test_public_api_exports_core_transformations():
    assert callable(add_structural_features)
    assert callable(add_crossfit_exact_te)
