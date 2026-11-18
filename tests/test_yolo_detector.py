"""Pure unit tests for the adapter: no Ultralytics installation needed."""
from io import BytesIO

import pytest
from PIL import Image

from app.services.inference_service import InferenceService
from app.services.yolo_detector import YOLODetector


class FakeTensor:
    """Only the .cpu().tolist() subset of a torch.Tensor we use."""

    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self, boxes, scores, classes):
        self.xyxy = FakeTensor(boxes)
        self.conf = FakeTensor(scores)
        self.cls = FakeTensor(classes)


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes
        self.names = {0: "person", 2: "car"}


class FakeYOLOModel:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return self.results


def make_fake_yolo():
    return FakeYOLOModel([
        FakeResult(FakeBoxes(
            boxes=[[10.2, 20.8, 100.1, 150.9], [40, 50, 200, 250]],
            scores=[0.92, 0.81],
            classes=[0.0, 2.0],
        )),
    ])


def test_adapter_converts_boxes_and_passes_yolo_arguments():
    model = make_fake_yolo()
    detector = YOLODetector(model=model, device="cpu")
    image = Image.new("RGB", (320, 260))

    detections = detector.predict(image, confidence_threshold=0.65)

    assert len(detections) == 2
    assert detections[0].class_id == 0
    assert detections[0].class_name == "person"
    assert detections[0].confidence == pytest.approx(0.92)
    assert detections[0].bbox.model_dump() == {
        "x1": 10, "y1": 21, "x2": 100, "y2": 151
    }
    assert detections[1].class_name == "car"
    assert model.calls == [{
        "source": image,
        "conf": 0.65,
        "device": "cpu",
        "verbose": False,
    }]


def test_adapter_handles_empty_detections():
    model = FakeYOLOModel([FakeResult(FakeBoxes([], [], []))])
    assert YOLODetector(model=model).predict(Image.new("RGB", (30, 30)), 0.5) == []


def test_adapter_handles_missing_boxes():
    model = FakeYOLOModel([FakeResult(None)])
    assert YOLODetector(model=model).predict(Image.new("RGB", (30, 30)), 0.5) == []


def test_inference_service_accepts_yolo_adapter():
    model = make_fake_yolo()
    service = InferenceService(YOLODetector(model=model))
    buffer = BytesIO()
    Image.new("L", (320, 260)).save(buffer, format="PNG")

    result = service.infer(buffer.getvalue(), confidence_threshold=0.7)

    assert result.format == "PNG"
    assert (result.width, result.height) == (320, 260)
    assert len(result.detections) == 2
    assert model.calls[0]["source"].mode == "RGB"
    assert model.calls[0]["conf"] == 0.7


def test_adapter_requires_a_model_or_model_path():
    with pytest.raises(ValueError, match="model_path is required"):
        YOLODetector()
