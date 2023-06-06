import pupil_labs.aois_module as this_project


def test_package_metadata() -> None:
    assert hasattr(this_project, "__version__")
