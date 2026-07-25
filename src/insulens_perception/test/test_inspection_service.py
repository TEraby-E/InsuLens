import csv
import json

import cv2
import imageio_ffmpeg
import numpy as np

from insulens_perception.inspection_service import (
    Detection,
    InspectionRunner,
    InsulatorDetector,
    IoUTracker,
    deduplicate,
    infer_report_category,
)


def test_deduplicate_keeps_highest_confidence_per_overlapping_class():
    kept = deduplicate([
        Detection((0, 0, 20, 20), "normal", 0.7),
        Detection((1, 1, 19, 19), "normal", 0.9),
        Detection((0, 0, 20, 20), "broken", 0.8),
    ])
    assert len(kept) == 2
    assert kept[0].confidence == 0.9


def test_tracker_preserves_id_for_same_insulator():
    tracker = IoUTracker()
    first = tracker.update([Detection((10, 10, 30, 50), "normal", 0.8)])[0]
    second = tracker.update([Detection((12, 11, 32, 51), "normal", 0.8)])[0]
    assert first.track_id == second.track_id


def test_detector_accepts_arbitrary_model_class_schema_for_web_inspection():
    detector = InsulatorDetector()
    detector.model = object()
    detector.model_task = "detect"
    detector.model_classes = {0: "insulator"}
    assert detector.ready_for_inspection
    assert detector.class_names == ("insulator",)

    detector.model_classes = {0: "person", 1: "helmet", 2: "vehicle"}
    assert detector.ready_for_inspection
    assert [item["name"] for item in detector.category_schema()] == [
        "person", "helmet", "vehicle"
    ]


def test_image_inspection_writes_model_driven_category_reports(tmp_path):
    image = np.zeros((160, 160, 3), dtype=np.uint8)
    cv2.rectangle(image, (40, 20), (65, 140), (255, 255, 255), -1)
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), image)
    report = InspectionRunner(InsulatorDetector(), tmp_path / "results").inspect_image(source)
    job = tmp_path / "results" / report["job_id"]
    json_report = json.loads((job / report["report_json"]).read_text(encoding="utf-8"))
    with (job / report["report_csv"]).open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.reader(handle))
    assert json_report["category_counts"] == {"object": report["detection_total"]}
    assert json_report["class_schema"][0]["name"] == "object"
    assert report["detection_total"] >= 1
    assert ["检测数量", str(report["detection_total"])] in csv_rows


def test_report_only_inference_is_optional_and_model_agnostic():
    classes = ("normal", "broken")
    assert infer_report_category("normal", 0.15, 0.30, classes, "flashover") == (
        "flashover", True
    )
    assert infer_report_category("broken", 0.85, 0.30, classes, "flashover") == (
        "broken", False
    )
    assert infer_report_category("normal", 0.15, None, classes, None) == (
        "normal", False
    )


def test_aliases_and_display_labels_are_optional_metadata():
    detector = InsulatorDetector(
        class_aliases={"missing_disc": "missing"},
        class_labels={"missing": "缺片"},
    )
    detector.model = object()
    detector.model_task = "detect"
    detector.model_classes = {0: "insulator", 1: "missing_disc"}

    schema = detector.category_schema()
    assert [item["name"] for item in schema] == ["insulator", "missing"]
    assert schema[1]["display_name"] == "缺片"


def test_video_inspection_writes_browser_compatible_h264(tmp_path):
    source = tmp_path / "source.avi"
    source_writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (96, 64)
    )
    assert source_writer.isOpened()
    for offset in range(4):
        frame = np.zeros((64, 96, 3), dtype=np.uint8)
        cv2.rectangle(frame, (12 + offset, 20), (72 + offset, 34), (255, 255, 255), -1)
        source_writer.write(frame)
    source_writer.release()

    report = InspectionRunner(InsulatorDetector(), tmp_path / "results").inspect_video(source)
    output = tmp_path / "results" / report["job_id"] / report["output_media"]
    reader = imageio_ffmpeg.read_frames(str(output), pix_fmt="rgb24")
    try:
        metadata = next(reader)
    finally:
        reader.close()

    assert metadata["codec"] == "h264"
    assert metadata["size"] == (96, 64)
    assert report["frames"] == 4
