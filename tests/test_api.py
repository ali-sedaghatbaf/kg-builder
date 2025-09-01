from fastapi import Response
from fastapi.testclient import TestClient
import pytest


@pytest.fixture(scope="module")
def client():
    from app.api import app

    with TestClient(app) as client:
        yield client


def test_read_main(client):
    response: Response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"Hello": "World"}


def test_file_upload(client):
    file_content = b"Hello World"
    response = client.post(
        "/upload/",
        files={"file": ("test.txt", file_content, "text/plain")},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "test.txt"
    assert data["content_type"] == "text/plain"
