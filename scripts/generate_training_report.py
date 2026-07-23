#!/usr/bin/env python3
"""Build a self-contained, presentation-friendly YOLO training report."""

from __future__ import annotations

import argparse
import base64
import csv
from datetime import datetime, timezone
import html
import mimetypes
from pathlib import Path
from typing import Iterable


PROJECT_DIR = Path(__file__).resolve().parents[1]

RUNS = (
    {
        "slug": "simulation",
        "title": "阶段一：仿真绝缘子定位",
        "subtitle": "用 Gazebo 自动标注数据学习绝缘子的位置、尺度和姿态变化",
        "run_dir": PROJECT_DIR / "runs" / "insulator_yolov10s",
        "model": PROJECT_DIR / "models" / "insulator_yolov10s.pt",
        "dataset": PROJECT_DIR / "datasets" / "insulator_sim",
        "accent": "#38bdf8",
        "interpretation": (
            "仿真模型的 mAP50 已接近饱和，说明它能够稳定完成绝缘子定位。"
            "mAP50-95 更严格，能反映边界框贴合程度，适合用来观察后续精细定位空间。"
        ),
    },
    {
        "slug": "defect",
        "title": "阶段二：真实场景缺陷识别",
        "subtitle": "在 CPLID 航拍图像上微调，识别绝缘子并定位 missing_disc 缺陷",
        "run_dir": PROJECT_DIR / "runs" / "insulator_defect_yolov10s",
        "model": PROJECT_DIR / "models" / "insulator_defect_yolov10s.pt",
        "dataset": PROJECT_DIR / "datasets" / "cplid_yolo",
        "accent": "#34d399",
        "interpretation": (
            "真实航拍图像包含远距离小目标、复杂背景和遮挡，因此指标低于仿真域是正常现象。"
            "验证样例显示模型已经能够同时框出绝缘子主体和缺失伞盘区域。"
        ),
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_DIR / "reports" / "training_report.html",
        help="Output HTML path (default: reports/training_report.html)",
    )
    return parser.parse_args()


def read_results(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def metric(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except (TypeError, ValueError):
        return 0.0


def count_files(directory: Path, suffixes: Iterable[str]) -> int:
    if not directory.is_dir():
        return 0
    allowed = {suffix.lower() for suffix in suffixes}
    return sum(
        1 for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in allowed
    )


def format_duration(seconds: float) -> str:
    minutes, seconds = divmod(int(round(seconds)), 60)
    if minutes >= 60:
        hours, minutes = divmod(minutes, 60)
        return f"{hours} 小时 {minutes} 分"
    return f"{minutes} 分 {seconds} 秒"


def format_size(path: Path) -> str:
    if not path.is_file():
        return "未找到"
    size = path.stat().st_size / (1024 * 1024)
    return f"{size:.1f} MB"


def data_uri(path: Path) -> str:
    if not path.is_file():
        return ""
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def image_panel(path: Path, title: str, note: str) -> str:
    uri = data_uri(path)
    if not uri:
        return (
            '<figure class="visual missing">'
            f"<figcaption><strong>{html.escape(title)}</strong><span>{html.escape(note)}</span>"
            "</figcaption><div>训练产物尚未生成</div></figure>"
        )
    return (
        '<figure class="visual">'
        f'<img class="zoomable" src="{uri}" alt="{html.escape(title)}" loading="lazy">'
        f"<figcaption><strong>{html.escape(title)}</strong><span>{html.escape(note)}</span>"
        "</figcaption></figure>"
    )


def metric_card(label: str, value: str, description: str, accent: str = "") -> str:
    style = f' style="--card-accent:{accent}"' if accent else ""
    return (
        f'<article class="metric"{style}><span>{html.escape(label)}</span>'
        f"<strong>{html.escape(value)}</strong><small>{html.escape(description)}</small></article>"
    )


def prepare_run(config: dict[str, object]) -> dict[str, object]:
    run_dir = Path(config["run_dir"])
    dataset = Path(config["dataset"])
    rows = read_results(run_dir / "results.csv")
    final = rows[-1] if rows else {}
    best_map50 = max(rows, key=lambda row: metric(row, "metrics/mAP50(B)"), default={})
    best_map5095 = max(
        rows, key=lambda row: metric(row, "metrics/mAP50-95(B)"), default={}
    )
    train_images = count_files(dataset / "images" / "train", (".jpg", ".jpeg", ".png"))
    val_images = count_files(dataset / "images" / "val", (".jpg", ".jpeg", ".png"))
    return {
        **config,
        "rows": rows,
        "final": final,
        "best_map50": best_map50,
        "best_map5095": best_map5095,
        "epochs": len(rows),
        "duration": metric(final, "time"),
        "train_images": train_images,
        "val_images": val_images,
        "total_images": train_images + val_images,
    }


def percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def build_run_section(run: dict[str, object]) -> str:
    run_dir = Path(run["run_dir"])
    final = run["final"]
    best_map50 = run["best_map50"]
    best_map5095 = run["best_map5095"]
    accent = str(run["accent"])
    slug = str(run["slug"])
    cards = "".join(
        (
            metric_card(
                "最终 Precision",
                percent(metric(final, "metrics/precision(B)")),
                "模型给出的检测框中，有多少是正确的",
                accent,
            ),
            metric_card(
                "最终 Recall",
                percent(metric(final, "metrics/recall(B)")),
                "所有真实目标中，有多少被模型成功找到",
                accent,
            ),
            metric_card(
                "峰值 mAP50",
                percent(metric(best_map50, "metrics/mAP50(B)")),
                f"第 {best_map50.get('epoch', '-')} 轮；宽松定位综合得分",
                accent,
            ),
            metric_card(
                "峰值 mAP50-95",
                percent(metric(best_map5095, "metrics/mAP50-95(B)")),
                f"第 {best_map5095.get('epoch', '-')} 轮；严格定位综合得分",
                accent,
            ),
            metric_card(
                "数据规模",
                f"{run['total_images']:,}",
                f"训练 {run['train_images']:,} / 验证 {run['val_images']:,}",
                accent,
            ),
            metric_card(
                "训练成本",
                format_duration(float(run["duration"])),
                f"{run['epochs']} 轮；模型文件 {format_size(Path(run['model']))}",
                accent,
            ),
        )
    )
    prediction_comparison = (
        '<div class="comparison">'
        + image_panel(
            run_dir / "val_batch0_labels.jpg",
            "人工标注（Ground Truth）",
            "模型应该学到的正确答案",
        )
        + image_panel(
            run_dir / "val_batch0_pred.jpg",
            "模型预测（Prediction）",
            "框、类别和置信度越接近标注越好",
        )
        + "</div>"
    )
    diagnostics = (
        '<div class="visual-grid">'
        + image_panel(
            run_dir / "results.png",
            "训练过程总览",
            "损失应总体下降，Precision、Recall 和 mAP 应总体上升",
        )
        + image_panel(
            run_dir / "BoxPR_curve.png",
            "Precision–Recall 曲线",
            "曲线越靠近右上角，代表漏检和误检同时更少",
        )
        + image_panel(
            run_dir / "confusion_matrix_normalized.png",
            "归一化混淆矩阵",
            "对角线越深越好；background 反映漏检与误检",
        )
        + "</div>"
    )
    return f"""
    <section id="{slug}" class="run-section" style="--accent:{accent}">
      <div class="section-heading">
        <span class="stage">{html.escape(str(run['title']))}</span>
        <h2>{html.escape(str(run['subtitle']))}</h2>
        <p>{html.escape(str(run['interpretation']))}</p>
      </div>
      <div class="metric-grid">{cards}</div>
      <h3>最直观的效果对比</h3>
      <p class="lead">左侧是验证集人工标注，右侧是模型在未参与梯度更新的验证图像上的预测。</p>
      {prediction_comparison}
      <h3>训练是否可靠</h3>
      <p class="lead">下面三组证据分别回答：模型有没有收敛、查准率与查全率如何平衡、错误主要发生在哪里。</p>
      {diagnostics}
    </section>
    """


def build_html(runs: list[dict[str, object]]) -> str:
    simulation, defect = runs
    sim_final = simulation["final"]
    defect_final = defect["final"]
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    hero_cards = "".join(
        (
            metric_card(
                "仿真定位 mAP50",
                percent(metric(sim_final, "metrics/mAP50(B)")),
                "验证模型是否学会稳定定位绝缘子",
                str(simulation["accent"]),
            ),
            metric_card(
                "真实缺陷 mAP50",
                percent(metric(defect_final, "metrics/mAP50(B)")),
                "复杂航拍背景下的综合检测能力",
                str(defect["accent"]),
            ),
            metric_card(
                "仿真数据",
                f"{simulation['total_images']:,}",
                "Gazebo 自动生成并自动标注",
                str(simulation["accent"]),
            ),
            metric_card(
                "真实数据",
                f"{defect['total_images']:,}",
                "CPLID 正常与缺陷绝缘子图像",
                str(defect["accent"]),
            ),
        )
    )
    sections = "".join(build_run_section(run) for run in runs)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>InsuLens YOLOv10 训练成果报告</title>
  <style>
    :root {{ color-scheme: dark; --bg:#07111f; --panel:#0d1b2d; --text:#e5eefb; --muted:#9fb0c8; --line:#20324a; }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{ margin:0; font-family:Inter,"Noto Sans SC","Microsoft YaHei",system-ui,sans-serif; background:radial-gradient(circle at 20% 0%,#123152 0,#07111f 38rem); color:var(--text); line-height:1.65; }}
    a {{ color:inherit; }}
    .page {{ width:min(1320px,calc(100% - 32px)); margin:auto; }}
    nav {{ position:sticky; top:0; z-index:10; border-bottom:1px solid rgba(255,255,255,.08); background:rgba(7,17,31,.86); backdrop-filter:blur(16px); }}
    nav .page {{ display:flex; align-items:center; justify-content:space-between; min-height:58px; gap:16px; }}
    nav strong {{ letter-spacing:.04em; }}
    nav div {{ display:flex; gap:18px; color:var(--muted); font-size:14px; }}
    nav a {{ text-decoration:none; }}
    header {{ padding:82px 0 54px; }}
    .eyebrow,.stage {{ display:inline-flex; padding:5px 11px; border:1px solid #2b5578; border-radius:99px; color:#7dd3fc; background:#0a2238; font-size:13px; letter-spacing:.08em; }}
    h1 {{ max-width:900px; margin:20px 0 14px; font-size:clamp(38px,7vw,76px); line-height:1.02; letter-spacing:-.04em; }}
    header p {{ max-width:820px; margin:0; color:var(--muted); font-size:18px; }}
    .hero-grid,.metric-grid {{ display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin-top:34px; }}
    .metric {{ position:relative; overflow:hidden; min-height:150px; padding:22px; border:1px solid var(--line); border-radius:18px; background:linear-gradient(145deg,rgba(19,40,65,.92),rgba(10,24,42,.95)); box-shadow:0 18px 50px rgba(0,0,0,.15); }}
    .metric:before {{ content:""; position:absolute; inset:0 auto 0 0; width:3px; background:var(--card-accent,#38bdf8); }}
    .metric span,.metric small {{ display:block; color:var(--muted); }}
    .metric strong {{ display:block; margin:7px 0 4px; font-size:clamp(25px,3vw,38px); line-height:1.1; }}
    .metric small {{ font-size:13px; line-height:1.45; }}
    .story {{ padding:34px; border:1px solid var(--line); border-radius:22px; background:rgba(13,27,45,.75); }}
    .story h2 {{ margin:0 0 22px; font-size:30px; }}
    .pipeline {{ display:grid; grid-template-columns:1fr auto 1fr auto 1fr auto 1fr; align-items:center; gap:12px; }}
    .pipeline div {{ min-height:126px; padding:20px; border:1px solid var(--line); border-radius:16px; background:#0b192b; }}
    .pipeline strong,.pipeline span {{ display:block; }}
    .pipeline strong {{ font-size:18px; }}
    .pipeline span {{ color:var(--muted); font-size:14px; }}
    .arrow {{ color:#60a5fa; font-size:28px; }}
    .run-section {{ margin:74px 0; scroll-margin-top:80px; }}
    .section-heading {{ max-width:920px; }}
    .section-heading .stage {{ color:var(--accent); border-color:color-mix(in srgb,var(--accent) 45%,transparent); background:color-mix(in srgb,var(--accent) 12%,transparent); }}
    h2 {{ margin:15px 0 10px; font-size:clamp(28px,4vw,48px); line-height:1.12; }}
    h3 {{ margin:48px 0 4px; font-size:26px; }}
    .section-heading p,.lead {{ color:var(--muted); }}
    .metric-grid {{ grid-template-columns:repeat(3,1fr); margin-top:28px; }}
    .comparison {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    .visual-grid {{ display:grid; grid-template-columns:2fr 1fr 1fr; gap:18px; align-items:start; }}
    figure {{ margin:0; }}
    .visual {{ overflow:hidden; border:1px solid var(--line); border-radius:18px; background:#0b192b; }}
    .visual img {{ display:block; width:100%; height:auto; cursor:zoom-in; background:white; }}
    .visual figcaption {{ display:flex; flex-direction:column; gap:2px; padding:14px 16px; }}
    .visual figcaption span {{ color:var(--muted); font-size:13px; }}
    .missing div {{ display:grid; min-height:250px; place-items:center; color:var(--muted); }}
    .takeaways {{ margin:70px 0; padding:38px; border:1px solid #28587b; border-radius:24px; background:linear-gradient(135deg,#0b243b,#0b1c31); }}
    .takeaways h2 {{ margin-top:0; }}
    .takeaways li {{ margin:10px 0; }}
    .commands {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
    pre {{ overflow:auto; margin:0; padding:20px; border:1px solid var(--line); border-radius:14px; background:#050c16; color:#bfdbfe; font-size:13px; }}
    footer {{ padding:28px 0 48px; border-top:1px solid var(--line); color:var(--muted); font-size:13px; }}
    dialog {{ width:min(96vw,1500px); max-height:94vh; padding:12px; border:1px solid #36516e; border-radius:16px; background:#07111f; }}
    dialog::backdrop {{ background:rgba(0,0,0,.82); }}
    dialog img {{ display:block; max-width:100%; max-height:88vh; margin:auto; }}
    dialog button {{ position:absolute; top:18px; right:18px; width:38px; height:38px; border:0; border-radius:50%; background:#07111f; color:white; cursor:pointer; font-size:22px; }}
    @media (max-width:980px) {{
      .hero-grid,.metric-grid {{ grid-template-columns:repeat(2,1fr); }}
      .pipeline {{ grid-template-columns:1fr; }} .arrow {{ transform:rotate(90deg); text-align:center; }}
      .visual-grid,.commands {{ grid-template-columns:1fr; }}
    }}
    @media (max-width:650px) {{
      nav div {{ display:none; }} header {{ padding-top:54px; }}
      .hero-grid,.metric-grid,.comparison {{ grid-template-columns:1fr; }}
      .story,.takeaways {{ padding:24px; }}
    }}
    @media print {{ nav,dialog {{ display:none; }} body {{ background:white; color:#152238; }} .metric,.story,.visual,.takeaways {{ break-inside:avoid; box-shadow:none; }} }}
  </style>
</head>
<body>
  <nav><div class="page"><strong>INSULENS · TRAINING REPORT</strong><div><a href="#overview">训练路线</a><a href="#simulation">仿真定位</a><a href="#defect">缺陷识别</a><a href="#conclusion">结论</a></div></div></nav>
  <header class="page">
    <span class="eyebrow">YOLOv10 · ROS 2 · GAZEBO</span>
    <h1>从仿真训练到真实缺陷识别</h1>
    <p>这份报告把训练日志转成可直接讲解的证据链：数据从哪里来、模型怎样收敛、验证集表现如何，以及最终能够看见什么。</p>
    <div class="hero-grid">{hero_cards}</div>
  </header>
  <main class="page">
    <section id="overview" class="story">
      <h2>两阶段训练路线</h2>
      <div class="pipeline">
        <div><strong>① Gazebo 数据</strong><span>{simulation['total_images']:,} 张自动标注图像，覆盖位置、尺度、旋转和光照变化。</span></div>
        <span class="arrow">→</span>
        <div><strong>② 仿真预训练</strong><span>先学会“绝缘子在哪里”，降低真实数据训练的冷启动难度。</span></div>
        <span class="arrow">→</span>
        <div><strong>③ CPLID 微调</strong><span>{defect['total_images']:,} 张真实航拍图像，引入复杂背景与 missing_disc 类别。</span></div>
        <span class="arrow">→</span>
        <div><strong>④ 巡检输出</strong><span>输出检测框、类别、置信度，并可触发缺陷告警与证据保存。</span></div>
      </div>
    </section>
    {sections}
    <section id="conclusion" class="takeaways">
      <h2>如何向别人解释结果</h2>
      <ul>
        <li><strong>先看预测图：</strong>它最直观地证明模型已经能够在未参与训练的验证图片上找到目标。</li>
        <li><strong>再看 mAP50：</strong>仿真定位达到 {percent(metric(sim_final, 'metrics/mAP50(B)'))}，真实缺陷任务达到 {percent(metric(defect_final, 'metrics/mAP50(B)'))}。</li>
        <li><strong>最后看曲线与混淆矩阵：</strong>曲线说明训练在收敛，混淆矩阵说明错误集中在哪些类别或背景。</li>
        <li><strong>不要直接横向比较两阶段：</strong>真实航拍任务更难、类别更多，两组数字对应不同数据集和不同目标。</li>
      </ul>
      <div class="commands">
        <pre><code># 重新生成本报告
python scripts/generate_training_report.py</code></pre>
        <pre><code># 启动仿真巡检可视化
./scripts/run_sim.sh</code></pre>
      </div>
    </section>
  </main>
  <footer><div class="page">生成时间：{generated_at} · 数据来源：Ultralytics 训练产物与 InsuLens 数据目录 · 点击任意图表可放大</div></footer>
  <dialog id="lightbox"><button aria-label="关闭">×</button><img alt="放大图"></dialog>
  <script>
    const box=document.querySelector('#lightbox'), large=box.querySelector('img');
    document.querySelectorAll('.zoomable').forEach(img=>img.addEventListener('click',()=>{{large.src=img.src;box.showModal();}}));
    box.querySelector('button').addEventListener('click',()=>box.close());
    box.addEventListener('click',event=>{{if(event.target===box)box.close();}});
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    runs = [prepare_run(dict(config)) for config in RUNS]
    missing = [str(Path(run["run_dir"]) / "results.csv") for run in runs if not run["rows"]]
    if missing:
        raise SystemExit("Missing training results: " + ", ".join(missing))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(runs), encoding="utf-8")
    print(f"Training report written to: {output}")


if __name__ == "__main__":
    main()
