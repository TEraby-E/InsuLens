"""Build a synthetic UAV inspection clip from labelled missing-disc samples.

The output is intentionally marked as a synthetic sequence because its shots
come from different training images rather than one continuous drone flight.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "insulator_six_class"
OUTPUT = ROOT / "reports" / "insulens_missing_disc_uav_demo.mp4"
MANIFEST = ROOT / "reports" / "insulens_missing_disc_uav_demo_sources.json"

WIDTH = 1280
HEIGHT = 720
FPS = 25
FRAMES_PER_SHOT = 70
TRANSITION_FRAMES = 10
MISSING_CLASS_ID = 4

# Ordered from wide views to closer inspection views.
SOURCE_NAMES = [
    "cplid_defective_075.jpg",
    "cplid_defective_083.jpg",
    "cplid_defective_109.jpg",
    "cplid_defective_031.jpg",
    "cplid_defective_114.jpg",
    "cplid_defective_100.jpg",
]

FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size=size)


FONTS = {
    "tiny": font(14),
    "small": font(17),
    "small_bold": font(17, True),
    "body": font(20),
    "body_bold": font(20, True),
    "label": font(22, True),
    "title": font(36, True),
}


def smoothstep(value: float) -> float:
    value = max(0.0, min(1.0, value))
    return value * value * (3.0 - 2.0 * value)


def load_sample(name: str) -> dict:
    image_path = DATASET / "images" / "train" / name
    label_path = DATASET / "labels" / "train" / f"{Path(name).stem}.txt"
    if not image_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing training sample or label: {name}")

    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Unable to decode {image_path}")
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.split()
        if len(values) != 5 or int(float(values[0])) != MISSING_CLASS_ID:
            continue
        xc, yc, bw, bh = map(float, values[1:])
        boxes.append(
            {
                "xyxy": [
                    (xc - bw / 2.0) * width,
                    (yc - bh / 2.0) * height,
                    (xc + bw / 2.0) * width,
                    (yc + bh / 2.0) * height,
                ],
                "normalized_xywh": [xc, yc, bw, bh],
                "area": bw * bh,
            }
        )
    if not boxes:
        raise ValueError(f"No class-{MISSING_CLASS_ID} box in {label_path}")
    target = max(boxes, key=lambda item: item["area"])
    return {
        "name": name,
        "image_path": image_path,
        "label_path": label_path,
        "image": image,
        "width": width,
        "height": height,
        "target": target,
    }


def crop_frame(sample: dict, progress: float, shot_index: int) -> tuple[np.ndarray, list[float]]:
    image = sample["image"]
    source_height, source_width = image.shape[:2]
    x1, y1, x2, y2 = sample["target"]["xyxy"]
    target_x = (x1 + x2) / 2.0
    target_y = (y1 + y2) / 2.0

    eased = smoothstep(progress)
    zoom_start = 1.02 + (shot_index % 2) * 0.025
    zoom_end = 1.23 + (shot_index % 3) * 0.025
    zoom = zoom_start + (zoom_end - zoom_start) * eased

    crop_width = source_width / zoom
    crop_height = crop_width / (WIDTH / HEIGHT)
    if crop_height > source_height:
        crop_height = source_height / zoom
        crop_width = crop_height * (WIDTH / HEIGHT)

    direction = -1.0 if shot_index % 2 else 1.0
    start_x = source_width * (0.50 - 0.055 * direction)
    start_y = source_height * (0.50 + 0.025 * direction)
    end_x = target_x * 0.82 + source_width * 0.50 * 0.18
    end_y = target_y * 0.82 + source_height * 0.50 * 0.18
    drift_x = math.sin(progress * math.pi * 2.0 + shot_index) * 5.0
    drift_y = math.sin(progress * math.pi * 1.4 + shot_index * 0.7) * 3.5
    center_x = start_x + (end_x - start_x) * eased + drift_x
    center_y = start_y + (end_y - start_y) * eased + drift_y

    left = max(0.0, min(source_width - crop_width, center_x - crop_width / 2.0))
    top = max(0.0, min(source_height - crop_height, center_y - crop_height / 2.0))
    right = left + crop_width
    bottom = top + crop_height

    cropped = image[
        int(round(top)) : int(round(bottom)),
        int(round(left)) : int(round(right)),
    ]
    resized = cv2.resize(cropped, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    output_box = [
        (x1 - left) * WIDTH / crop_width,
        (y1 - top) * HEIGHT / crop_height,
        (x2 - left) * WIDTH / crop_width,
        (y2 - top) * HEIGHT / crop_height,
    ]
    return resized, output_box


def rgba_rectangle(draw: ImageDraw.ImageDraw, box, fill, outline=None, width: int = 1, radius: int = 0):
    if radius:
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    else:
        draw.rectangle(box, fill=fill, outline=outline, width=width)


def corner_box(draw: ImageDraw.ImageDraw, box: list[float], color, alpha: int) -> None:
    x1, y1, x2, y2 = box
    x1 = max(18, min(WIDTH - 18, int(x1)))
    y1 = max(76, min(HEIGHT - 50, int(y1)))
    x2 = max(18, min(WIDTH - 18, int(x2)))
    y2 = max(76, min(HEIGHT - 50, int(y2)))
    if x2 <= x1 or y2 <= y1:
        return
    pad = 9
    x1 -= pad
    y1 -= pad
    x2 += pad
    y2 += pad
    segment = max(15, min(34, int(min(x2 - x1, y2 - y1) * 0.35)))
    width = 4
    for points in [
        [(x1, y1 + segment), (x1, y1), (x1 + segment, y1)],
        [(x2 - segment, y1), (x2, y1), (x2, y1 + segment)],
        [(x1, y2 - segment), (x1, y2), (x1 + segment, y2)],
        [(x2 - segment, y2), (x2, y2), (x2, y2 - segment)],
    ]:
        draw.line(points, fill=(*color, alpha), width=width, joint="curve")


def overlay_hud(
    frame: np.ndarray,
    box: list[float],
    shot_index: int,
    progress: float,
    timeline_seconds: float,
    total_shots: int,
) -> np.ndarray:
    base = Image.fromarray(frame, mode="RGB").convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    # Letterbox-like telemetry strips.
    rgba_rectangle(draw, (0, 0, WIDTH, 64), fill=(5, 23, 39, 190))
    rgba_rectangle(draw, (0, HEIGHT - 34, WIDTH, HEIGHT), fill=(5, 23, 39, 175))
    draw.rectangle((0, 0, 7, HEIGHT), fill=(20, 125, 223, 235))
    draw.text((28, 16), "INSULENS UAV  //  绝缘子自主巡检", font=FONTS["body_bold"], fill=(255, 255, 255, 240))
    draw.ellipse((1008, 22, 1020, 34), fill=(239, 67, 67, 255))
    draw.text((1028, 14), "REC", font=FONTS["small_bold"], fill=(255, 255, 255, 235))
    simulated_altitude = 42.0 - shot_index * 1.8 - progress * 0.6
    draw.text((1086, 14), f"SIM ALT {simulated_altitude:04.1f} m", font=FONTS["small"], fill=(179, 205, 222, 235))

    # Crosshair and animated scan line.
    center_x, center_y = WIDTH // 2, HEIGHT // 2
    cross = (25, 198, 212, 110)
    draw.line((center_x - 22, center_y, center_x - 7, center_y), fill=cross, width=2)
    draw.line((center_x + 7, center_y, center_x + 22, center_y), fill=cross, width=2)
    draw.line((center_x, center_y - 22, center_x, center_y - 7), fill=cross, width=2)
    draw.line((center_x, center_y + 7, center_x, center_y + 22), fill=cross, width=2)
    draw.ellipse((center_x - 4, center_y - 4, center_x + 4, center_y + 4), outline=cross, width=1)
    if progress < 0.34:
        scan_y = int(92 + (HEIGHT - 150) * (progress / 0.34))
        draw.line((26, scan_y, WIDTH - 26, scan_y), fill=(25, 198, 212, 95), width=2)
        draw.text((30, scan_y - 23), "SCANNING", font=FONTS["tiny"], fill=(25, 198, 212, 210))

    # Ground-truth target overlay fades in after the scanning phase.
    acquire = smoothstep((progress - 0.23) / 0.16)
    if acquire > 0:
        alpha = int(255 * acquire)
        target_color = (239, 70, 70)
        corner_box(draw, box, target_color, alpha)
        x1, y1, x2, _ = box
        label_x = int(max(22, min(WIDTH - 250, x1 - 8)))
        label_y = int(max(76, min(HEIGHT - 92, y1 - 47)))
        rgba_rectangle(
            draw,
            (label_x, label_y, label_x + 235, label_y + 38),
            fill=(151, 29, 36, int(220 * acquire)),
            radius=4,
        )
        draw.text(
            (label_x + 10, label_y + 7),
            "MISSING_DISC · 缺片样例",
            font=FONTS["small_bold"],
            fill=(255, 255, 255, alpha),
        )
        panel_x = WIDTH - 292
        panel_y = 92
        rgba_rectangle(draw, (panel_x, panel_y, WIDTH - 28, panel_y + 114), fill=(5, 23, 39, int(186 * acquire)), radius=8)
        draw.text((panel_x + 16, panel_y + 13), f"TARGET {shot_index + 1:02d}", font=FONTS["small_bold"], fill=(25, 198, 212, alpha))
        draw.text((panel_x + 16, panel_y + 43), "状态：需要人工复核", font=FONTS["body_bold"], fill=(255, 255, 255, alpha))
        draw.text((panel_x + 16, panel_y + 78), "SOURCE  TRAIN / LABEL", font=FONTS["tiny"], fill=(179, 205, 222, alpha))

    draw.text((24, HEIGHT - 28), "合成巡检演示｜素材来自训练集，不代表连续实拍", font=FONTS["tiny"], fill=(215, 229, 238, 235))
    draw.text((WIDTH - 232, HEIGHT - 28), f"T+{timeline_seconds:05.1f}s   {shot_index + 1:02d}/{total_shots:02d}", font=FONTS["tiny"], fill=(215, 229, 238, 235))

    # Opening and closing narration cards are embedded in the video itself.
    if shot_index == 0 and progress < 0.46:
        fade = 1.0 if progress < 0.28 else max(0.0, (0.46 - progress) / 0.18)
        rgba_rectangle(draw, (66, 115, 700, 292), fill=(5, 23, 39, int(205 * fade)), radius=14)
        draw.text((94, 143), "绝缘子缺片无人机巡检", font=FONTS["title"], fill=(255, 255, 255, int(255 * fade)))
        draw.text((96, 203), "SYNTHETIC UAV INSPECTION SEQUENCE", font=FONTS["small_bold"], fill=(25, 198, 212, int(255 * fade)))
        draw.text((96, 240), "6 个训练集缺片样例 · 720p / 25 FPS", font=FONTS["body"], fill=(179, 205, 222, int(255 * fade)))

    if shot_index == total_shots - 1 and progress > 0.67:
        fade = smoothstep((progress - 0.67) / 0.15)
        rgba_rectangle(draw, (70, 524, 616, 642), fill=(5, 23, 39, int(210 * fade)), radius=12)
        draw.text((94, 544), "巡检样例序列完成", font=FONTS["label"], fill=(255, 255, 255, int(255 * fade)))
        draw.text((94, 586), "6 / 6 缺片样例已记录 · 建议人工复核", font=FONTS["small"], fill=(25, 198, 212, int(255 * fade)))

    composed = Image.alpha_composite(base, layer).convert("RGB")
    return np.asarray(composed)


def render_shot(sample: dict, shot_index: int, total_shots: int, base_frame_index: int) -> list[np.ndarray]:
    frames = []
    for local_index in range(FRAMES_PER_SHOT):
        progress = local_index / (FRAMES_PER_SHOT - 1)
        frame, output_box = crop_frame(sample, progress, shot_index)
        timeline_seconds = (base_frame_index + local_index) / FPS
        frame = overlay_hud(frame, output_box, shot_index, progress, timeline_seconds, total_shots)
        frames.append(np.ascontiguousarray(frame))
    return frames


def encode(samples: list[dict]) -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(OUTPUT),
        (WIDTH, HEIGHT),
        fps=FPS,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "19", "-preset", "medium", "-movflags", "+faststart"],
    )
    writer.send(None)
    written = 0
    previous_tail: list[np.ndarray] | None = None
    try:
        for shot_index, sample in enumerate(samples):
            base_index = shot_index * (FRAMES_PER_SHOT - TRANSITION_FRAMES)
            frames = render_shot(sample, shot_index, len(samples), base_index)
            if previous_tail is not None:
                for transition_index in range(TRANSITION_FRAMES):
                    blend = smoothstep((transition_index + 1) / (TRANSITION_FRAMES + 1))
                    mixed = cv2.addWeighted(
                        previous_tail[transition_index],
                        1.0 - blend,
                        frames[transition_index],
                        blend,
                        0.0,
                    )
                    writer.send(np.ascontiguousarray(mixed).tobytes())
                    written += 1
                start = TRANSITION_FRAMES
            else:
                start = 0

            end = len(frames) - TRANSITION_FRAMES
            for frame in frames[start:end]:
                writer.send(frame.tobytes())
                written += 1
            previous_tail = frames[end:]

        if previous_tail:
            for frame in previous_tail:
                writer.send(frame.tobytes())
                written += 1
    finally:
        writer.close()
    return written


def main() -> None:
    samples = [load_sample(name) for name in SOURCE_NAMES]
    frame_count = encode(samples)
    duration = frame_count / FPS
    manifest = {
        "output": str(OUTPUT),
        "synthetic_sequence": True,
        "continuous_real_flight": False,
        "dataset": str(DATASET / "images" / "train"),
        "class_id": MISSING_CLASS_ID,
        "class_name": "missing",
        "resolution": [WIDTH, HEIGHT],
        "fps": FPS,
        "frames": frame_count,
        "duration_seconds": round(duration, 3),
        "sources": [
            {
                "image": str(sample["image_path"]),
                "label": str(sample["label_path"]),
                "normalized_xywh": sample["target"]["normalized_xywh"],
            }
            for sample in samples
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
    print(f"Wrote {MANIFEST}")
    print(f"Frames: {frame_count}; duration: {duration:.2f}s")


if __name__ == "__main__":
    main()
