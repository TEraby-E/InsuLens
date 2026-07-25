const $ = (id) => document.getElementById(id);
const classMetadata = new Map();
let sourceObjectUrl = null;

function isVideoFile(file) {
  return file.type.startsWith("video/") || /\.(mp4|avi|mov|mkv)$/i.test(file.name);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const unit = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
  return `${(bytes / 1024 ** unit).toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

function clearSourceObjectUrl() {
  if (sourceObjectUrl) URL.revokeObjectURL(sourceObjectUrl);
  sourceObjectUrl = null;
}

function renderSourcePreview(file) {
  clearSourceObjectUrl();
  if (!file) {
    $("source-preview").classList.add("hidden");
    $("source-media").replaceChildren();
    return;
  }

  sourceObjectUrl = URL.createObjectURL(file);
  const video = isVideoFile(file);
  const element = document.createElement(video ? "video" : "img");
  element.src = sourceObjectUrl;
  if (video) {
    element.controls = true;
    element.playsInline = true;
    element.preload = "metadata";
    element.setAttribute("aria-label", `上传视频预览：${file.name}`);
    element.addEventListener("loadedmetadata", () => {
      const duration = Number.isFinite(element.duration) ? ` · ${element.duration.toFixed(1)} 秒` : "";
      $("source-meta").textContent = `${file.name} · ${formatBytes(file.size)}${duration}`;
      $("source-message").textContent = "原始视频已就绪，可直接播放；提交巡检不会中断当前预览。";
    });
    element.addEventListener("error", () => {
      $("source-message").textContent = "浏览器无法解码该原始格式，但仍可提交；建议上传 H.264 编码的 MP4。";
    });
  } else {
    element.alt = `上传图片预览：${file.name}`;
  }
  $("source-meta").textContent = `${file.name} · ${formatBytes(file.size)}`;
  $("source-media").replaceChildren(element);
  $("source-preview").classList.remove("hidden");
}

function setClassSchema(schema = []) {
  if (!Array.isArray(schema) || schema.length === 0) return;
  classMetadata.clear();
  schema.forEach((item, index) => {
    if (!item || typeof item.name !== "string") return;
    classMetadata.set(item.name, {
      displayName: item.display_name || item.name,
      color: item.color || `hsl(${(index * 67) % 360} 70% 58%)`,
    });
  });
}

function classInfo(name, index = 0) {
  return classMetadata.get(name) || {
    displayName: name,
    color: `hsl(${(index * 67) % 360} 70% 58%)`,
  };
}

async function health() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    setClassSchema(data.class_schema);
    if (data.ready_for_inspection) {
      const classCount = data.class_schema?.length ?? data.class_names?.length ?? 0;
      const task = data.model_task ? ` · ${data.model_task}` : "";
      $("status").textContent = `${data.backend}${task} · ${classCount} 类模型已就绪`;
      $("model-info").textContent = `类别由模型自动读取：${(data.class_schema || []).map((item) => item.display_name || item.name).join("、") || "未提供类别名"}`;
    } else {
      $("status").textContent = "检测模型未就绪";
      $("progress").textContent = `模型不可用：${data.compatibility_error || data.load_error || "请通过 INSULENS_WEB_MODEL 指定兼容的目标检测权重"}`;
    }
  } catch {
    $("status").textContent = "服务未连接";
  }
}

function artifact(job, name) {
  return `/api/jobs/${encodeURIComponent(job)}/${encodeURIComponent(name)}`;
}

function link(label, url) {
  const anchor = document.createElement("a");
  anchor.textContent = label;
  anchor.href = url;
  anchor.target = "_blank";
  anchor.rel = "noopener";
  return anchor;
}

function renderCounts(counts = {}) {
  const entries = Object.entries(counts);
  const nodes = entries.map(([name, count], index) => {
    const info = classInfo(name, index);
    const node = document.createElement("div");
    node.className = "count-row";
    node.style.setProperty("--class-color", info.color);
    const label = document.createElement("span");
    label.textContent = info.displayName;
    const value = document.createElement("strong");
    value.textContent = String(count);
    node.replaceChildren(label, value);
    return node;
  });
  if (nodes.length === 0) {
    const empty = document.createElement("p");
    empty.className = "muted";
    empty.textContent = "当前模型尚未返回类别统计。";
    nodes.push(empty);
  }
  $("counts").replaceChildren(...nodes);
}

function renderMetrics(result) {
  setClassSchema(result.class_schema);
  $("result").classList.remove("hidden");
  const counts = result.category_counts || {};
  $("total").textContent = result.detection_total ?? Object.values(counts).reduce((sum, count) => sum + Number(count || 0), 0);
  $("fps").textContent = result.fps ?? "—";
  $("confidence").textContent = result.average_confidence == null ? "—" : `${(result.average_confidence * 100).toFixed(1)}%`;
  $("frames").textContent = result.frames ?? result.frames_processed ?? 0;
  renderCounts(counts);
}

function renderPreview(result, media) {
  const element = document.createElement(result.media_type === "video" ? "video" : "img");
  element.src = media;
  if (result.media_type === "video") {
    element.controls = true;
    element.playsInline = true;
    element.preload = "metadata";
  }
  else element.alt = "标注检测结果";
  $("preview").replaceChildren(element);
}

function renderCompletedReport(result) {
  renderMetrics(result);
  const media = artifact(result.job_id, result.output_media);
  renderPreview(result, media);
  const countText = Object.entries(result.category_counts || {})
    .map(([name, value], index) => `${classInfo(name, index).displayName} ${value}`)
    .join("，");
  $("report-text").textContent = `检测数量 ${result.detection_total ?? 0}${countText ? `；${countText}` : ""}。`;
  $("downloads").replaceChildren(
    link("导出 JSON 报告", artifact(result.job_id, result.report_json)),
    link("导出 CSV 报告", artifact(result.job_id, result.report_csv)),
    link("下载标注媒体", media),
  );
}

async function pollVideoJob(jobId) {
  const response = await fetch(`/api/jobs/${encodeURIComponent(jobId)}/status`);
  const state = await response.json();
  if (!response.ok) throw new Error(state.detail || "无法读取任务状态");
  if (state.status === "failed") throw new Error(state.detail || "视频处理失败");
  if (state.status === "completed") {
    renderCompletedReport(state.report);
    $("progress").textContent = `已完成：使用 ${state.report.backend}，耗时 ${state.report.processing_seconds} 秒。`;
    return;
  }
  renderMetrics(state);
  const total = state.frames_total ? `/${state.frames_total}` : "";
  const current = state.current_frame?.detections?.length ?? 0;
  $("progress").textContent = `正在逐帧推理：${state.frames_processed || 0}${total} 帧，当前帧检测 ${current} 个目标。`;
  await new Promise((resolve) => setTimeout(resolve, 400));
  return pollVideoJob(jobId);
}

$("upload-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = $("file").files[0];
  if (!file) return;
  $("progress").textContent = `正在上传 ${file.name}…`;
  const form = new FormData();
  form.append("upload", file);
  $("submit-button").disabled = true;
  try {
    const response = await fetch("/api/inspect", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(result.detail || "巡检失败");
    setClassSchema(result.class_schema);
    if (result.status === "queued") await pollVideoJob(result.job_id);
    else {
      renderCompletedReport(result);
      $("progress").textContent = `已完成：使用 ${result.backend}，耗时 ${result.processing_seconds} 秒。`;
    }
  } catch (error) {
    $("progress").textContent = `处理失败：${error.message}`;
  } finally {
    $("submit-button").disabled = false;
  }
});

$("file").addEventListener("change", (event) => renderSourcePreview(event.target.files?.[0]));
window.addEventListener("beforeunload", clearSourceObjectUrl);

health();
