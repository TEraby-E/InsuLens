"""Regression tests for small-object coordinate and clustering primitives."""

import numpy as np

from insulens_perception.small_object import (
    generate_tiles,
    kmeans_iou,
    weighted_box_fusion,
)


def test_tiles_cover_right_and_bottom_edges():
    tiles = generate_tiles(1000, 700, tile_size=512, overlap=0.25)
    assert any(tile.x + tile.width == 1000 for tile in tiles)
    assert any(tile.y + tile.height == 700 for tile in tiles)


def test_kmeans_returns_a_cluster_for_every_box():
    boxes = np.asarray([[8.0, 10.0], [9.0, 11.0], [100.0, 90.0], [110.0, 95.0]])
    centroids, assignment = kmeans_iou(boxes, clusters=2, seed=7)
    assert centroids.shape == (2, 2)
    assert assignment.shape == (4,)


def test_weighted_fusion_keeps_classes_and_merges_overlap():
    detections = [
        {"class_id": 1, "confidence": 0.9, "bbox_xyxy": [10, 10, 30, 30]},
        {"class_id": 1, "confidence": 0.8, "bbox_xyxy": [11, 10, 31, 30]},
        {"class_id": 2, "confidence": 0.7, "bbox_xyxy": [10, 10, 30, 30]},
    ]
    fused = weighted_box_fusion(detections, iou_threshold=0.5)
    assert len(fused) == 2
    assert fused[0]["tile_observations"] == 2
