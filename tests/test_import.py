def test_package_import() -> None:
    import conditioned_bo

    assert conditioned_bo.__version__ == "0.1.0"
