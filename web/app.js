const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const debugCanvas = document.getElementById("debugCanvas");
const dctx = debugCanvas.getContext("2d");
const result = document.getElementById("result");
const logitsOutput = document.getElementById("logits");
const modelStatus = document.getElementById("modelStatus");
const labelSelect = document.getElementById("labelSelect");
const sampleCount = document.getElementById("sampleCount");

const MODEL_SIZE = 64;
const MODEL_LINE_WIDTH = 2;
const MODEL_PADDING_RATIO = 0.08;
const STORAGE_KEY = "shape_classifier_samples";

const labels = [
  "ellipse",
  "line",
  "triangle",
  "rectangle",
  "pentagon",
  "hexagon",
  "star",
  "zigzag",
  "arc",
  "heart",
  "diamond",
  "arrow",
  "double_arrow",
  "cloud",
  "message",
  "parallelogram",
];

let drawing = false;
let lastX = null;
let lastY = null;
let currentStroke = [];
let strokePoints = [];
let session = null;
let shouldClearOnNextStroke = false;
let predictTimeoutId = null;

function fillWhite(targetCtx, width, height) {
  targetCtx.fillStyle = "white";
  targetCtx.fillRect(0, 0, width, height);
}

function resetDrawingState() {
  fillWhite(ctx, canvas.width, canvas.height);
  fillWhite(dctx, debugCanvas.width, debugCanvas.height);
  currentStroke = [];
  strokePoints = [];
  lastX = null;
  lastY = null;
  logitsOutput.textContent = "Run a prediction to inspect logits.";
  setResult("Waiting for input");
}

function updateSampleCount(value) {
  sampleCount.textContent = String(value);
}

function loadSavedSamples() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch (error) {
    console.error("Failed to load saved samples", error);
    return [];
  }
}

function persistSamples(samples) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
  updateSampleCount(samples.length);
}

function populateLabels() {
  for (const label of labels) {
    const option = document.createElement("option");
    option.value = label;
    option.textContent = label;
    labelSelect.appendChild(option);
  }
}

function setResult(message) {
  result.textContent = message;
}

function finishStroke() {
  drawing = false;
  lastX = null;
  lastY = null;

  if (currentStroke.length > 0) {
    strokePoints = strokePoints.concat(currentStroke);
    currentStroke = [];
    scheduleAutoPredict();
  }
}

function scheduleAutoPredict() {
  if (predictTimeoutId) {
    window.clearTimeout(predictTimeoutId);
  }

  predictTimeoutId = window.setTimeout(() => {
    predictTimeoutId = null;
    predict({ silent: true });
  }, 120);
}

function draw(event) {
  if (!drawing) return;

  const { x, y } = getCanvasPoint(event);

  currentStroke.push([x, y]);

  if (lastX !== null) {
    ctx.strokeStyle = "black";
    ctx.lineWidth = 8;
    ctx.lineCap = "round";

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
  }

  lastX = x;
  lastY = y;
}

function getCanvasPoint(event) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  const x = (event.clientX - rect.left) * scaleX;
  const y = (event.clientY - rect.top) * scaleY;

  return {
    x: Math.max(0, Math.min(canvas.width, x)),
    y: Math.max(0, Math.min(canvas.height, y)),
  };
}

function startStroke(event) {
  event.preventDefault();
  if (shouldClearOnNextStroke) {
    resetDrawingState();
    shouldClearOnNextStroke = false;
  }
  drawing = true;
  draw(event);
}

function getTouchPosition(event) {
  const touch = event.touches[0] || event.changedTouches[0];
  return {
    clientX: touch.clientX,
    clientY: touch.clientY,
  };
}

function handleTouchStart(event) {
  const touch = getTouchPosition(event);
  startStroke({
    preventDefault: () => event.preventDefault(),
    clientX: touch.clientX,
    clientY: touch.clientY,
  });
}

function handleTouchMove(event) {
  const touch = getTouchPosition(event);
  draw({
    clientX: touch.clientX,
    clientY: touch.clientY,
  });
  event.preventDefault();
}

function getAllPoints() {
  return strokePoints.concat(currentStroke);
}

function paintSquare(pixels, size, x, y, radius) {
  for (let dy = -radius; dy <= radius; dy++) {
    for (let dx = -radius; dx <= radius; dx++) {
      const px = x + dx;
      const py = y + dy;
      if (px < 0 || px >= size || py < 0 || py >= size) continue;
      pixels[py * size + px] = 255;
    }
  }
}

function softmax(values) {
  const maxValue = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - maxValue));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

function drawRasterLine(pixels, size, x0, y0, x1, y1, lineWidth) {
  let x = x0;
  let y = y0;
  const dx = Math.abs(x1 - x0);
  const dy = Math.abs(y1 - y0);
  const sx = x0 < x1 ? 1 : -1;
  const sy = y0 < y1 ? 1 : -1;
  let err = dx - dy;
  const radius = Math.max(0, Math.floor((lineWidth - 1) / 2));

  while (true) {
    paintSquare(pixels, size, x, y, radius);
    if (x === x1 && y === y1) break;
    const e2 = 2 * err;
    if (e2 > -dy) {
      err -= dy;
      x += sx;
    }
    if (e2 < dx) {
      err += dx;
      y += sy;
    }
  }
}

function pointsToModelInput(points, size = MODEL_SIZE) {
  const validPoints = points.filter(
    (point) =>
      Array.isArray(point) &&
      point.length === 2 &&
      Number.isFinite(point[0]) &&
      Number.isFinite(point[1]),
  );

  if (validPoints.length < 2) {
    return null;
  }

  let minX = validPoints[0][0];
  let minY = validPoints[0][1];
  let maxX = validPoints[0][0];
  let maxY = validPoints[0][1];

  for (const [x, y] of validPoints) {
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  const rangeX = maxX - minX;
  const rangeY = maxY - minY;
  let maxRange = Math.max(rangeX, rangeY);
  if (maxRange === 0) maxRange = 1;

  const innerScale = Math.max(1.0 - 2.0 * MODEL_PADDING_RATIO, 1e-6);
  const offsetX = (maxRange - rangeX) / 2;
  const offsetY = (maxRange - rangeY) / 2;
  const rasterPoints = [];

  validPoints.forEach(([x, y]) => {
    const nx = ((x - minX + offsetX) / maxRange) * innerScale + MODEL_PADDING_RATIO;
    const ny = ((y - minY + offsetY) / maxRange) * innerScale + MODEL_PADDING_RATIO;
    const px = Math.min(Math.max(nx, 0), 1) * (size - 1);
    const py = Math.min(Math.max(ny, 0), 1) * (size - 1);
    rasterPoints.push([Math.round(px), Math.round(py)]);
  });

  const pixels = new Uint8Array(size * size);
  for (let i = 0; i < rasterPoints.length - 1; i++) {
    const [x0, y0] = rasterPoints[i];
    const [x1, y1] = rasterPoints[i + 1];
    drawRasterLine(pixels, size, x0, y0, x1, y1, MODEL_LINE_WIDTH);
  }

  const input = new Float32Array(size * size);
  const debugImage = dctx.createImageData(size, size);

  for (let i = 0; i < size * size; i++) {
    const val = pixels[i];
    const normalized = 1.0 - val / 255.0;
    input[i] = normalized;

    debugImage.data[i * 4] = val;
    debugImage.data[i * 4 + 1] = val;
    debugImage.data[i * 4 + 2] = val;
    debugImage.data[i * 4 + 3] = 255;
  }

  const debugTempCanvas = document.createElement("canvas");
  debugTempCanvas.width = size;
  debugTempCanvas.height = size;
  debugTempCanvas.getContext("2d").putImageData(debugImage, 0, 0);

  dctx.imageSmoothingEnabled = false;
  dctx.clearRect(0, 0, debugCanvas.width, debugCanvas.height);
  dctx.drawImage(debugTempCanvas, 0, 0, debugCanvas.width, debugCanvas.height);

  return input;
}

async function loadModel() {
  const modelUrl = `./model.onnx?v=${Date.now()}`;
  modelStatus.textContent = "Loading...";
  try {
    session = await ort.InferenceSession.create(modelUrl);
    modelStatus.textContent = "Ready";
  } catch (error) {
    console.error("Failed to load ONNX model", error);
    modelStatus.textContent = `Failed to load (${error.message || error})`;
  }
}

async function predict(options = {}) {
  const { silent = false } = options;

  if (!session) {
    if (!silent) {
      alert("Model not loaded yet");
    }
    return;
  }

  const inputData = pointsToModelInput(getAllPoints());
  if (!inputData) {
    if (!silent) {
      alert("Draw something first");
    }
    return;
  }

  const tensor = new ort.Tensor("float32", inputData, [1, 1, MODEL_SIZE, MODEL_SIZE]);
  const outputs = await session.run({ input: tensor });
  const logits = outputs.output.data;
  const probabilities = softmax(Array.from(logits));

  logitsOutput.textContent =
    "Probabilities:\n" +
    probabilities
      .map((value, index) => ({ label: labels[index], value }))
      .sort((a, b) => b.value - a.value)
      .map(({ label, value }) => `${label}: ${(value * 100).toFixed(2)}%`)
      .join("\n");

  let maxIdx = 0;
  let maxVal = logits[0];
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > maxVal) {
      maxVal = logits[i];
      maxIdx = i;
    }
  }

  setResult(labels[maxIdx]);
  shouldClearOnNextStroke = true;
}

function clearCanvas() {
  resetDrawingState();
  shouldClearOnNextStroke = false;
}

function saveSample() {
  const points = getAllPoints();
  if (points.length < 2) {
    alert("Draw something first");
    return;
  }

  let label = labelSelect.value;
  if (label === "square") label = "rectangle";
  if (label === "circle") label = "ellipse";

  const sample = {
    label,
    points,
    source: "browser",
    created_at: new Date().toISOString(),
  };

  const samples = loadSavedSamples();
  samples.push(sample);
  persistSamples(samples);
  clearCanvas();
  setResult(`Saved sample for ${sample.label}`);
  shouldClearOnNextStroke = true;
}

function downloadSamples() {
  const samples = loadSavedSamples();
  if (samples.length === 0) {
    alert("No saved samples yet");
    return;
  }

  const blob = new Blob([JSON.stringify(samples, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = `browser-samples-${Date.now()}.json`;
  anchor.click();
  URL.revokeObjectURL(url);
}

function clearSavedSamples() {
  const samples = loadSavedSamples();
  if (samples.length > 0 && !window.confirm(`Delete all ${samples.length} saved samples?`)) {
    return;
  }
  persistSamples([]);
}

function bindEvents() {
  canvas.addEventListener("mousedown", startStroke);
  canvas.addEventListener("mouseup", finishStroke);
  canvas.addEventListener("mouseleave", finishStroke);
  canvas.addEventListener("mousemove", draw);
  canvas.addEventListener("touchstart", handleTouchStart, { passive: false });
  canvas.addEventListener("touchmove", handleTouchMove, { passive: false });
  canvas.addEventListener("touchend", finishStroke);
  canvas.addEventListener("touchcancel", finishStroke);

  document.getElementById("clearBtn").addEventListener("click", clearCanvas);
  document.getElementById("predictBtn").addEventListener("click", predict);
  document.getElementById("saveBtn").addEventListener("click", saveSample);
  document.getElementById("downloadBtn").addEventListener("click", downloadSamples);
  document.getElementById("clearSavedBtn").addEventListener("click", clearSavedSamples);
}

function init() {
  fillWhite(ctx, canvas.width, canvas.height);
  fillWhite(dctx, debugCanvas.width, debugCanvas.height);
  populateLabels();
  persistSamples(loadSavedSamples());
  bindEvents();
  loadModel();
}

init();
