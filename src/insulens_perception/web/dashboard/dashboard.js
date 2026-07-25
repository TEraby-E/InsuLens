/* global ROSLIB */
const state = { frames: 0, objects: 0, alerts: 0, latencies: [], ros: null, topics: [] };
const $ = (selector) => document.querySelector(selector);
const status = $("#connection-status");
const canvas = $("#detection-canvas");
const context = canvas.getContext("2d");

function setStatus(text, className) { status.textContent = text; status.className = `status ${className}`; }
function metric(id, value) { $(id).textContent = value; }
function renderMetrics() {
  metric("#frame-count", state.frames); metric("#object-count", state.objects); metric("#alert-count", state.alerts);
  const avg = state.latencies.length ? state.latencies.reduce((a, b) => a + b, 0) / state.latencies.length : 0;
  metric("#latency", `${avg.toFixed(1)} ms`);
}
function safeJson(value) { try { return JSON.parse(value); } catch (_) { return null; } }
function renderDetections(detections) {
  const body = $("#detections"); body.replaceChildren();
  if (!detections.length) { body.innerHTML = '<tr><td colspan="5" class="empty">当前帧未检测到目标。</td></tr>'; return; }
  detections.forEach((item) => {
    const row = document.createElement("tr"); const box = item.bbox_xyxy.map((value) => Number(value).toFixed(1)).join(", ");
    row.innerHTML = `<td>${item.class_name}</td><td>${(item.confidence * 100).toFixed(1)}%</td><td class="${item.is_defect ? "defect" : "normal"}">${item.is_defect ? "缺陷" : "正常"}</td><td>${box}</td><td>${item.tile_observations || 1}</td>`;
    body.append(row);
  });
}
function onDetections(message) {
  const payload = safeJson(message.data); if (!payload || !Array.isArray(payload.detections)) return;
  state.frames += 1; state.objects += payload.detections.length; state.latencies.push(Number(payload.inference_ms) || 0);
  if (state.latencies.length > 30) state.latencies.shift(); renderMetrics(); renderDetections(payload.detections);
  $("#message-time").textContent = `最近更新：${new Date().toLocaleTimeString()}`;
}
function onAlert(message) {
  const payload = safeJson(message.data); if (!payload) return; state.alerts += 1; renderMetrics();
  const list = $("#alerts"); list.querySelector(".empty")?.remove(); const classes = (payload.detections || []).map((item) => item.class_name).join("、") || "未知缺陷";
  const entry = document.createElement("li"); entry.className = "alert"; entry.innerHTML = `<strong>${classes}</strong><span>本帧 ${payload.detections?.length || 0} 个缺陷目标</span><time>${new Date().toLocaleString()}</time>`;
  list.prepend(entry); while (list.children.length > 20) list.lastElementChild.remove();
}
function onImage(message) {
  const bytes = Uint8Array.from(atob(message.data), (char) => char.charCodeAt(0)); const width = message.width; const height = message.height; const step = message.step;
  if (!width || !height || !bytes.length) return; canvas.width = width; canvas.height = height; const image = context.createImageData(width, height);
  for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) { const source = y * step + x * 3; const target = (y * width + x) * 4; image.data[target] = bytes[source + 2]; image.data[target + 1] = bytes[source + 1]; image.data[target + 2] = bytes[source]; image.data[target + 3] = 255; }
  context.putImageData(image, 0, 0); $("#image-status").textContent = `${width} × ${height}`;
}
function connect() {
  state.topics.forEach((topic) => topic.unsubscribe()); state.topics = []; state.ros?.close(); setStatus("连接中", "connecting");
  const ros = new ROSLIB.Ros({ url: $("#ros-url").value.trim() }); state.ros = ros;
  ros.on("connection", () => { setStatus("已连接", "online"); state.topics = [
    new ROSLIB.Topic({ ros, name: "/insulens/detections", messageType: "std_msgs/String" }),
    new ROSLIB.Topic({ ros, name: "/insulens/defect_alerts", messageType: "std_msgs/String" }),
    new ROSLIB.Topic({ ros, name: "/insulens/detection_image", messageType: "sensor_msgs/Image" }),
  ]; state.topics[0].subscribe(onDetections); state.topics[1].subscribe(onAlert); state.topics[2].subscribe(onImage); });
  ros.on("error", () => setStatus("连接失败", "offline")); ros.on("close", () => setStatus("连接已关闭", "offline"));
}
$("#connect").addEventListener("click", connect); $("#clear-alerts").addEventListener("click", () => { $("#alerts").innerHTML = '<li class="empty">尚未收到缺陷告警。</li>'; });
