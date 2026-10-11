import pytest
from fastapi.testclient import TestClient

from spam_filter.api import app


@pytest.fixture(scope="session")
def client():
    # Using the client in a "with" block runs the startup code, which loads the model
    with TestClient(app) as test_client:
        yield test_client
