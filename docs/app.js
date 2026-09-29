/**
 * Case Study 46: Handwritten Digit Recognition
 * Real-time In-Browser Client-Side Machine Learning with PCA
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const canvas = document.getElementById('digitCanvas');
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const preview8x8 = document.getElementById('preview8x8');
  const ctx8x8 = preview8x8.getContext('2d');
  const previewPca = document.getElementById('previewPca');
  const ctxPca = previewPca.getContext('2d');

  const btnClear = document.getElementById('btnClear');
  const imageUpload = document.getElementById('imageUpload');
  const btnSample = document.getElementById('btnSample');
  const strokeWidthInput = document.getElementById('strokeWidth');
  const modelSelect = document.getElementById('modelSelect');

  const predictedDigitEl = document.getElementById('predictedDigit');
  const predictionConfidenceEl = document.getElementById('predictionConfidence');
  const inferenceLatencyEl = document.getElementById('inferenceLatency');
  const probBarsContainer = document.getElementById('probBars');

  // Drawing state
  let isDrawing = false;
  let lastX = 0;
  let lastY = 0;
  let strokeWidth = parseInt(strokeWidthInput.value, 10);

  // Initialize Canvas
  function initCanvas() {
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = strokeWidth;

    ctx8x8.fillStyle = '#000000';
    ctx8x8.fillRect(0, 0, preview8x8.width, preview8x8.height);
    ctxPca.fillStyle = '#000000';
    ctxPca.fillRect(0, 0, previewPca.width, previewPca.height);

    buildProbabilityBars();
    resetPrediction();
  }

  // Build the 10 probability bars
  function buildProbabilityBars() {
    probBarsContainer.innerHTML = '';
    for (let i = 0; i < 10; i++) {
      const row = document.createElement('div');
      row.className = 'prob-row';
      row.id = `prob-row-${i}`;
      row.innerHTML = `
        <span class="prob-digit">${i}</span>
        <div class="prob-bar-track">
          <div class="prob-bar-fill" id="prob-fill-${i}"></div>
        </div>
        <span class="prob-percent" id="prob-val-${i}">0%</span>
      `;
      probBarsContainer.appendChild(row);
    }
  }

  function resetPrediction() {
    predictedDigitEl.textContent = '-';
    predictionConfidenceEl.textContent = '-';
    inferenceLatencyEl.textContent = '< 1 ms';

    for (let i = 0; i < 10; i++) {
      const fill = document.getElementById(`prob-fill-${i}`);
      const val = document.getElementById(`prob-val-${i}`);
      const row = document.getElementById(`prob-row-${i}`);
      if (fill) fill.style.width = '0%';
      if (val) val.textContent = '0%';
      if (row) row.classList.remove('active');
    }
  }

  // Coordinates helper
  function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const clientX = e.touches ? e.touches[0].clientX : e.clientX;
    const clientY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clientX - rect.left) * (canvas.width / rect.width),
      y: (clientY - rect.top) * (canvas.height / rect.height)
    };
  }

  // Drawing event listeners
  function startDrawing(e) {
    isDrawing = true;
    const coords = getCanvasCoords(e);
    lastX = coords.x;
    lastY = coords.y;
    drawPoint(coords.x, coords.y);
    e.preventDefault();
  }

  function drawPoint(x, y) {
    ctx.beginPath();
    ctx.arc(x, y, strokeWidth / 2, 0, Math.PI * 2);
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    processAndPredict();
  }

  function drawMove(e) {
    if (!isDrawing) return;
    const coords = getCanvasCoords(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(coords.x, coords.y);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = strokeWidth;
    ctx.stroke();

    lastX = coords.x;
    lastY = coords.y;
    processAndPredict();
    e.preventDefault();
  }

  function stopDrawing() {
    if (isDrawing) {
      isDrawing = false;
      processAndPredict();
    }
  }

  // Mouse Listeners
  canvas.addEventListener('mousedown', startDrawing);
  canvas.addEventListener('mousemove', drawMove);
  window.addEventListener('mouseup', stopDrawing);

  // Touch Listeners
  canvas.addEventListener('touchstart', startDrawing, { passive: false });
  canvas.addEventListener('touchmove', drawMove, { passive: false });
  canvas.addEventListener('touchend', stopDrawing, { passive: false });

  // Control Listeners
  btnClear.addEventListener('click', initCanvas);

  strokeWidthInput.addEventListener('input', (e) => {
    strokeWidth = parseInt(e.target.value, 10);
  });

  modelSelect.addEventListener('change', () => {
    processAndPredict();
  });

  // Image Upload Listener
  imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = () => {
        // Draw image onto canvas maintaining aspect ratio
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        const hRatio = canvas.width / img.width;
        const vRatio = canvas.height / img.height;
        const ratio = Math.min(hRatio, vRatio) * 0.75;
        const centerShiftX = (canvas.width - img.width * ratio) / 2;
        const centerShiftY = (canvas.height - img.height * ratio) / 2;

        ctx.drawImage(img, 0, 0, img.width, img.height,
                      centerShiftX, centerShiftY, img.width * ratio, img.height * ratio);

        // Convert image colors if it has white background (invert if necessary)
        const imgData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const data = imgData.data;
        let whitePixels = 0;
        let blackPixels = 0;
        for (let i = 0; i < data.length; i += 4) {
          const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
          if (brightness > 128) whitePixels++;
          else blackPixels++;
        }
        // If image background is mostly light, invert it
        if (whitePixels > blackPixels) {
          for (let i = 0; i < data.length; i += 4) {
            const gray = 255 - (data[i] + data[i + 1] + data[i + 2]) / 3;
            data[i] = gray;
            data[i + 1] = gray;
            data[i + 2] = gray;
          }
          ctx.putImageData(imgData, 0, 0);
        }

        processAndPredict();
      };
      img.src = event.target.result;
    };
    reader.readAsDataURL(file);
  });

  // Random Digit Sample Generator
  btnSample.addEventListener('click', () => {
    if (!MODEL_DATA || !MODEL_DATA.knn_samples) return;
    const randomIndex = Math.floor(Math.random() * MODEL_DATA.knn_samples.length);
    const targetDigit = MODEL_DATA.knn_labels[randomIndex];
    
    // Draw synthetic representation on canvas
    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.save();
    ctx.font = 'bold 180px "Geist", -apple-system, sans-serif';
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(targetDigit.toString(), canvas.width / 2, canvas.height / 2 + 10);
    ctx.restore();

    processAndPredict();
  });

  /**
   * Preprocessing & Machine Learning Pipeline
   */
  function processAndPredict() {
    if (typeof MODEL_DATA === 'undefined') {
      console.warn("Model data not loaded yet.");
      return;
    }

    const t0 = performance.now();
    const srcData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const pixels = srcData.data;

    // 1. Find bounding box of drawn digit
    let minX = canvas.width, minY = canvas.height, maxX = 0, maxY = 0;
    let hasInk = false;

    for (let y = 0; y < canvas.height; y++) {
      for (let x = 0; x < canvas.width; x++) {
        const idx = (y * canvas.width + x) * 4;
        const brightness = pixels[idx]; // white is 255
        if (brightness > 20) {
          hasInk = true;
          if (x < minX) minX = x;
          if (x > maxX) maxX = x;
          if (y < minY) minY = y;
          if (y > maxY) maxY = y;
        }
      }
    }

    if (!hasInk) {
      resetPrediction();
      return;
    }

    // 2. Center and resize bounding box into an 8x8 grid
    const bboxW = Math.max(maxX - minX + 1, 1);
    const bboxH = Math.max(maxY - minY + 1, 1);

    // Create intermediate canvas for centered digit
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 8;
    tempCanvas.height = 8;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.fillStyle = '#000000';
    tempCtx.fillRect(0, 0, 8, 8);

    // Maintain aspect ratio with 1-pixel border padding (6x6 inside 8x8)
    const size = Math.max(bboxW, bboxH);
    const scale = 5.5 / size;
    const dx = 4 - (bboxW * scale) / 2;
    const dy = 4 - (bboxH * scale) / 2;

    tempCtx.drawImage(
      canvas,
      minX, minY, bboxW, bboxH,
      dx, dy, bboxW * scale, bboxH * scale
    );

    // 3. Extract 64 pixel values (0.0 to 16.0 range as in load_digits)
    const gridImgData = tempCtx.getImageData(0, 0, 8, 8);
    const gridData = gridImgData.data;
    const rawFeatures = new Float64Array(64);

    for (let i = 0; i < 64; i++) {
      const b = gridData[i * 4]; // R channel
      rawFeatures[i] = (b / 255.0) * 16.0;
    }

    // Render 8x8 visualizer preview
    renderToPreview(rawFeatures, ctx8x8, preview8x8);

    // 4. Feature Scaling: (X - mean) / scale
    const scaledFeatures = new Float64Array(64);
    for (let i = 0; i < 64; i++) {
      const mean = MODEL_DATA.scaler_mean[i];
      const scaleVal = MODEL_DATA.scaler_scale[i] || 1.0;
      scaledFeatures[i] = (rawFeatures[i] - mean) / scaleVal;
    }

    // 5. PCA Projection: X_pca = (X_scaled - pca_mean) * Components^T
    const nComp = MODEL_DATA.n_components; // 31
    const pcaFeatures = new Float64Array(nComp);

    // Center by PCA mean
    const centeredScaled = new Float64Array(64);
    for (let i = 0; i < 64; i++) {
      centeredScaled[i] = scaledFeatures[i] - MODEL_DATA.pca_mean[i];
    }

    for (let k = 0; k < nComp; k++) {
      let dot = 0.0;
      const compK = MODEL_DATA.pca_components[k];
      for (let i = 0; i < 64; i++) {
        dot += centeredScaled[i] * compK[i];
      }
      pcaFeatures[k] = dot;
    }

    // 6. PCA Reconstruction Preview: X_rec = X_pca * Components + pca_mean
    const reconstructedScaled = new Float64Array(64);
    for (let i = 0; i < 64; i++) {
      let rec = MODEL_DATA.pca_mean[i];
      for (let k = 0; k < nComp; k++) {
        rec += pcaFeatures[k] * MODEL_DATA.pca_components[k][i];
      }
      // Denormalize
      const unscaled = rec * (MODEL_DATA.scaler_scale[i] || 1.0) + MODEL_DATA.scaler_mean[i];
      reconstructedScaled[i] = Math.max(0, Math.min(16, unscaled));
    }
    renderToPreview(reconstructedScaled, ctxPca, previewPca);

    // 7. Model Inference (KNN vs Logistic Regression)
    const selectedModel = modelSelect.value;
    let probabilities = new Float64Array(10);

    if (selectedModel === 'knn') {
      probabilities = predictKNN(pcaFeatures, 5);
    } else {
      probabilities = predictLogisticRegression(pcaFeatures);
    }

    const latency = (performance.now() - t0).toFixed(1);

    // 8. Find predicted class & update UI
    let bestDigit = 0;
    let maxProb = -1;
    for (let d = 0; d < 10; d++) {
      if (probabilities[d] > maxProb) {
        maxProb = probabilities[d];
        bestDigit = d;
      }
    }

    updateUI(bestDigit, maxProb, probabilities, latency);
  }

  // Logistic Regression Softmax Inference
  function predictLogisticRegression(pcaFeatures) {
    const logits = new Float64Array(10);
    const nComp = pcaFeatures.length;

    let maxLogit = -Infinity;
    for (let c = 0; c < 10; c++) {
      let score = MODEL_DATA.lr_intercept[c];
      const wRow = MODEL_DATA.lr_weights[c];
      for (let k = 0; k < nComp; k++) {
        score += pcaFeatures[k] * wRow[k];
      }
      logits[c] = score;
      if (score > maxLogit) maxLogit = score;
    }

    // Softmax with numerical stability
    let sumExp = 0.0;
    const exps = new Float64Array(10);
    for (let c = 0; c < 10; c++) {
      exps[c] = Math.exp(logits[c] - maxLogit);
      sumExp += exps[c];
    }

    const probs = new Float64Array(10);
    for (let c = 0; c < 10; c++) {
      probs[c] = exps[c] / sumExp;
    }
    return probs;
  }

  // KNN Inference in 31-dim PCA Space
  function predictKNN(pcaFeatures, k = 5) {
    const samples = MODEL_DATA.knn_samples;
    const labels = MODEL_DATA.knn_labels;
    const n = samples.length;
    const nComp = pcaFeatures.length;

    // Calculate Euclidean distances
    const distances = [];
    for (let i = 0; i < n; i++) {
      const s = samples[i];
      let distSq = 0.0;
      for (let d = 0; d < nComp; d++) {
        const diff = pcaFeatures[d] - s[d];
        distSq += diff * diff;
      }
      distances.push({ dist: Math.sqrt(distSq), label: labels[i] });
    }

    // Sort to get top k neighbors
    distances.sort((a, b) => a.dist - b.dist);
    const topK = distances.slice(0, k);

    // Distance-weighted voting
    const votes = new Float64Array(10);
    let totalWeight = 0.0;

    for (let i = 0; i < k; i++) {
      const weight = 1.0 / (topK[i].dist + 0.0001);
      votes[topK[i].label] += weight;
      totalWeight += weight;
    }

    const probs = new Float64Array(10);
    for (let c = 0; c < 10; c++) {
      probs[c] = totalWeight > 0 ? (votes[c] / totalWeight) : 0.1;
    }
    return probs;
  }

  // Render 8x8 array to canvas
  function renderToPreview(data64, context, targetCanvas) {
    const imgData = context.createImageData(8, 8);
    for (let i = 0; i < 64; i++) {
      const val = Math.round((data64[i] / 16.0) * 255);
      const idx = i * 4;
      imgData.data[idx] = val;     // R
      imgData.data[idx + 1] = val; // G
      imgData.data[idx + 2] = val; // B
      imgData.data[idx + 3] = 255; // Alpha
    }

    // Scale onto target preview canvas (64x64)
    const offscreen = document.createElement('canvas');
    offscreen.width = 8;
    offscreen.height = 8;
    offscreen.getContext('2d').putImageData(imgData, 0, 0);

    context.imageSmoothingEnabled = false;
    context.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
    context.drawImage(offscreen, 0, 0, targetCanvas.width, targetCanvas.height);
  }

  // Update UI Elements
  function updateUI(digit, confidence, probs, latency) {
    predictedDigitEl.textContent = digit;
    predictionConfidenceEl.textContent = `${(confidence * 100).toFixed(1)}%`;
    inferenceLatencyEl.textContent = `${latency} ms`;

    for (let d = 0; d < 10; d++) {
      const pct = (probs[d] * 100).toFixed(1);
      const fill = document.getElementById(`prob-fill-${d}`);
      const val = document.getElementById(`prob-val-${d}`);
      const row = document.getElementById(`prob-row-${d}`);

      if (fill) fill.style.width = `${pct}%`;
      if (val) val.textContent = `${pct}%`;

      if (row) {
        if (d === digit) {
          row.classList.add('active');
        } else {
          row.classList.remove('active');
        }
      }
    }
  }

  // Start app
  initCanvas();
});
