"""Shared fixtures; importing main loads the SentenceTransformer once per test session."""

import pytest


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    import main as app_main

    return TestClient(app_main.app)
