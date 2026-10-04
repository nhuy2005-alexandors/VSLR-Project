from __future__ import annotations

import io
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from prototype_3_gestures.web_server import RealtimeVSLRPipeline, create_app


@pytest.fixture(scope="module")
def app_client():
    model_path = Path("artifacts/v3-realtime-test-candidate/gesture_lstm.pt")
    if not model_path.is_file():
        pytest.skip(f"Model checkpoint not found at {model_path}")

    pipeline = RealtimeVSLRPipeline(
        model_path=model_path,
        camera_id=0,
        no_record=True,
        allow_uncalibrated=True,
    )
    app = create_app(pipeline, web_dir="vslr-web")
    with TestClient(app) as client:
        yield client, pipeline
    pipeline.stop()


def test_recognize_video_invalid_extension(app_client):
    client, _ = app_client
    fake_txt = io.BytesIO(b"this is not a video file")
    response = client.post(
        "/api/recognize_video",
        files={"file": ("notes.txt", fake_txt, "text/plain")},
    )
    assert response.status_code == 400
    data = response.json()
    assert data["status"] == "error"
    assert "Định dạng video không được hỗ trợ" in data["detail"]


def test_recognize_video_success(app_client):
    client, pipeline = app_client
    video_path = Path("vslr-web/tutorials/xin_chao.mp4")
    if not video_path.is_file():
        pytest.skip(f"Sample video not found at {video_path}")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    response = client.post(
        "/api/recognize_video",
        files={"file": ("xin_chao.mp4", io.BytesIO(video_bytes), "video/mp4")},
        data={"signer": "TestSigner", "session_id": "test_sess_1", "add_to_sentence": "true"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["label"] == "Xin chào"
    assert data["confidence"] >= 70.0
    assert data["accepted"] is True
    assert data["signer"] == "TestSigner"
    assert "Xin chào" in data["sentence"]
    assert "meta" in data
    assert data["meta"]["sampled_frames"] > 0
    assert len(data["top_k"]) == 5
    assert data["top_k"][0]["label"] == "Xin chào"
    assert data["top_k"][0]["confidence"] >= data["top_k"][1]["confidence"]


def test_convert_video_preview(app_client):
    client, _ = app_client
    video_path = Path("vslr-web/tutorials/xin_chao.mp4")
    if not video_path.is_file():
        pytest.skip(f"Sample video not found at {video_path}")

    with open(video_path, "rb") as f:
        video_bytes = f.read()

    response = client.post(
        "/api/convert_video_preview",
        files={"file": ("raw_test.mp4", io.BytesIO(video_bytes), "video/mp4")},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "video/mp4"
    assert len(response.content) > 1000
