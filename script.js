/**
 * VisiónClara — script.js
 * Procesamiento digital de imágenes para deuteranopia
 * Métodos: Daltonize (LMS), Simulación, Realce + Segmentación K-means
 * Autores: Johan Cárdenas & Jhon Blandón · UNMG 2026
 *
 * Referencias:
 *  - Fidaner & Çadallı (2009). "Deficiency Correction"
 *  - Viénot et al. (1999). "Digital Video Colourmaps for Checking the Legibility"
 *  - Machado et al. (2009). "A Physiologically-based Model for Simulation of Color Vision Deficiency"
 */

'use strict';

/* ══════════════════════════════════════════════════════
   1. MATRICES DE TRANSFORMACIÓN DE COLOR
   Fuente: Fidaner & Çadallı (2009) / Viénot et al. (1999)
   Operan directamente sobre valores RGB 0–255
   ══════════════════════════════════════════════════════ */

/**
 * M_rgb2lms: Transformación RGB → espacio LMS
 * Modela la sensibilidad de los conos L (Largo ~560nm),
 * M (Medio ~530nm) y S (Corto ~420nm) de la retina humana.
 *
 *   ⎡ L ⎤   ⎡ 17.8824    43.5161    4.11935 ⎤   ⎡ R ⎤
 *   ⎢ M ⎥ = ⎢  3.45565   27.1554    3.86714 ⎥ · ⎢ G ⎥
 *   ⎣ S ⎦   ⎣  0.029957   0.184309  1.46709 ⎦   ⎣ B ⎦
 */
const RGB_TO_LMS = [
  [17.8824,    43.5161,   4.11935 ],
  [ 3.45565,   27.1554,   3.86714 ],
  [ 0.0299566,  0.184309,  1.46709 ]
];

/**
 * M_lms2rgb: Inversa de RGB_TO_LMS
 * Reconstruye el vector RGB desde el espacio LMS.
 */
const LMS_TO_RGB = [
  [ 0.0809444479, -0.130504409,  0.116721066  ],
  [-0.0102485335,  0.0540193266,-0.113614708  ],
  [-0.000365297,  -0.00412161,   0.693511405  ]
];

/**
 * M_deut: Proyección de deuteranopia en espacio LMS.
 * Los conos M (verdes) son no funcionales; su señal se
 * aproxima mediante combinación lineal de L y S:
 *   M' = 0.4942·L + 0·M + 1.2483·S
 *
 *   ⎡ L' ⎤   ⎡ 1.0    0.0    0.0    ⎤   ⎡ L ⎤
 *   ⎢ M' ⎥ = ⎢ 0.4942 0.0    1.2483 ⎥ · ⎢ M ⎥
 *   ⎣ S' ⎦   ⎣ 0.0    0.0    1.0    ⎦   ⎣ S ⎦
 */
const DEUT_SIM = [
  [1.0,    0.0,    0.0   ],
  [0.4942, 0.0,    1.2483],
  [0.0,    0.0,    1.0   ]
];

/**
 * M_shift: Redistribución del error de percepción (Daltonize).
 * El error E = RGB_orig - RGB_sim se desplaza hacia los
 * canales G y B que el sistema deuteranópico sí distingue.
 *
 *   ΔR' = 0·E_r + 0·E_g + 0·E_b
 *   ΔG' = 0.7·E_r + 1·E_g + 0·E_b
 *   ΔB' = 0.7·E_r + 0·E_g + 1·E_b
 */
const ERR_SHIFT = [
  [0.0, 0.0, 0.0],
  [0.7, 1.0, 0.0],
  [0.7, 0.0, 1.0]
];

/* ══════════════════════════════════════════════════════
   2. ÁLGEBRA LINEAL — Helpers
   ══════════════════════════════════════════════════════ */

/** Producto Matriz × Vector: result[i] = Σ_j M[i][j]·v[j] */
function matVec(M, v) {
  return [
    M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
    M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
    M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2]
  ];
}

/** Restricción de valor al rango [lo, hi] */
function clamp(v, lo = 0, hi = 255) {
  return Math.max(lo, Math.min(hi, v));
}

/** Distancia euclidiana al cuadrado en espacio de color RGB */
function colorDistSq(a, b) {
  const dr = a[0]-b[0], dg = a[1]-b[1], db = a[2]-b[2];
  return dr*dr + dg*dg + db*db;
}

/* ══════════════════════════════════════════════════════
   3. ALGORITMOS DE PROCESAMIENTO DE COLOR
   ══════════════════════════════════════════════════════ */

/**
 * ALGORITMO 1 — DALTONIZE (Fidaner & Çadallı, 2009)
 *
 * Pipeline completo:
 *   1. RGB → LMS  (espacio de conos retinales)
 *   2. LMS → LMS' (simulación deuteranopia: M' = 0.4942L + 1.2483S)
 *   3. LMS' → RGB_sim (imagen como la ve el deuteranópico)
 *   4. E = RGB - RGB_sim  (error de percepción)
 *   5. ΔC = M_shift · E  (redistribuir error a canales distinguibles)
 *   6. RGB_out = RGB + α·ΔC  (corrección ponderada por intensidad α)
 *
 * @param {number} r - Canal rojo   [0, 255]
 * @param {number} g - Canal verde  [0, 255]
 * @param {number} b - Canal azul   [0, 255]
 * @param {number} intensity - Factor de mezcla α ∈ [0, 1]
 * @returns {number[]} [r', g', b'] corregidos
 */
function daltonize(r, g, b, intensity) {
  // Paso 1: RGB → LMS
  const lms    = matVec(RGB_TO_LMS, [r, g, b]);

  // Paso 2: Simular deuteranopia en espacio LMS
  const lmsSim = matVec(DEUT_SIM, lms);

  // Paso 3: LMS simulado → RGB (percepción deuteranópica)
  const rgbSim = matVec(LMS_TO_RGB, lmsSim);

  // Paso 4: Error de percepción E = Original − Simulado
  const err = [r - rgbSim[0], g - rgbSim[1], b - rgbSim[2]];

  // Paso 5: Redistribuir error hacia canales perceptibles (G, B)
  const shift = matVec(ERR_SHIFT, err);

  // Paso 6: Aplicar corrección con factor de intensidad α
  return [
    clamp(Math.round(r + shift[0] * intensity)),
    clamp(Math.round(g + shift[1] * intensity)),
    clamp(Math.round(b + shift[2] * intensity))
  ];
}

/**
 * ALGORITMO 2 — SIMULACIÓN DE DEUTERANOPIA
 *
 * Aplica la transformación completa RGB → LMS → LMS'(deut) → RGB
 * sin corrección, mostrando la imagen tal como la percibiría
 * una persona con deuteranopia.
 *
 * @param {number} r, g, b - Valores RGB [0, 255]
 * @returns {number[]} [r', g', b'] percibidos por deuteranópico
 */
function simulateDeuteranopia(r, g, b) {
  const lms    = matVec(RGB_TO_LMS, [r, g, b]);
  const lmsSim = matVec(DEUT_SIM, lms);
  const rgb    = matVec(LMS_TO_RGB, lmsSim);
  return [
    clamp(Math.round(rgb[0])),
    clamp(Math.round(rgb[1])),
    clamp(Math.round(rgb[2]))
  ];
}

/**
 * ALGORITMO 3 — REALCE DE CONTRASTE AZUL–AMARILLO
 *
 * Detecta tonos rojo-verde conflictivos (|R-G| > umbral)
 * y los redirige hacia el eje azul–amarillo, que es el eje
 * cromático preservado en la deuteranopia.
 *
 * Criterio: píxeles con conflicto RG significativo y sin
 * componente azul dominante (B < 128).
 *
 * @param {number} r, g, b - Valores RGB [0, 255]
 * @param {number} intensity - Factor de corrección α ∈ [0, 1]
 * @returns {number[]} [r', g', b'] realzados
 */
function enhance(r, g, b, intensity) {
  const rn = r / 255, gn = g / 255, bn = b / 255;
  const conflicto = Math.abs(rn - gn);

  // Sin conflicto perceptivo → sin cambio
  if (conflicto < 0.1 || bn > 0.5) return [r, g, b];

  if (rn > gn) {
    // Dominante ROJO → añadir componente azul (distinguible)
    return [
      clamp(Math.round(r * (1 - 0.3 * intensity))),
      g,
      clamp(Math.round(b + r * 0.5 * intensity))
    ];
  } else {
    // Dominante VERDE → empujar hacia AMARILLO (R+G visible)
    return [
      clamp(Math.round(r + g * 0.25 * intensity)),
      g,
      clamp(Math.round(b * (1 - 0.2 * intensity)))
    ];
  }
}

/**
 * PROCESADOR PRINCIPAL — itera sobre todos los píxeles del ImageData
 * y aplica el algoritmo seleccionado según canales activos.
 *
 * @param {ImageData} imageData - Datos de la imagen (se modifica in-place)
 * @param {string} mode - 'daltonize' | 'simulate' | 'enhance'
 * @param {number} intensity - Factor α ∈ [0, 1]
 * @param {Object} channels - {r, g, b}: qué canales aplicar la corrección
 * @returns {{affected: number, total: number}} Estadísticas de píxeles
 */
function processImage(imageData, mode, intensity, channels) {
  const data  = imageData.data;
  const total = data.length / 4;
  let affected = 0;

  for (let i = 0; i < data.length; i += 4) {
    const r = data[i], g = data[i+1], b = data[i+2];
    let nr = r, ng = g, nb = b;

    if      (mode === 'daltonize') [nr, ng, nb] = daltonize(r, g, b, intensity);
    else if (mode === 'simulate')  [nr, ng, nb] = simulateDeuteranopia(r, g, b);
    else if (mode === 'enhance')   [nr, ng, nb] = enhance(r, g, b, intensity);

    data[i]   = channels.r ? nr : r;
    data[i+1] = channels.g ? ng : g;
    data[i+2] = channels.b ? nb : b;

    if (nr !== r || ng !== g || nb !== b) affected++;
  }
  return { affected, total };
}

/* ══════════════════════════════════════════════════════
   4. K-MEANS SEGMENTACIÓN
   Función objetivo: J = Σ_k Σ_{x ∈ Cₖ} ‖x − μₖ‖²
   Donde μₖ es el centroide del cluster k y x son
   los vectores RGB de los píxeles asignados al cluster.
   ══════════════════════════════════════════════════════ */

class KMeansSegmenter {
  constructor() {
    this.k            = 5;
    this.maxIter      = 15;
    this.assignments  = null;  // Uint8Array: cluster por píxel
    this.centroids    = null;  // Array de centroides RGB
    this.width        = 0;
    this.height       = 0;
    this.selectedSet  = new Set(); // clusters seleccionados

    /** Paleta de colores para overlay de segmentos */
    this.palette = [
      [255,  80,  80],  // rojo
      [ 60, 140, 255],  // azul
      [ 60, 210, 110],  // verde
      [255, 200,  40],  // amarillo
      [200,  70, 220],  // violeta
      [ 40, 210, 210],  // cian
      [255, 140,  50],  // naranja
      [180, 180,  50],  // oliva
    ];
  }

  /**
   * Inicialización K-means++ — Selecciona centroides iniciales con
   * probabilidad proporcional a D²(x, C) para reducir iteraciones.
   *
   * Complejidad: O(k · n) donde n = número de píxeles muestreados.
   *
   * @param {Array} pixels - Muestra de píxeles [[r,g,b], ...]
   * @param {number} k     - Número de clusters
   * @returns {Array}      - Centroides iniciales
   */
  _kmeanspp(pixels, k) {
    const n = pixels.length;
    const centroids = [pixels[Math.floor(Math.random() * n)].slice()];

    for (let c = 1; c < k; c++) {
      // D²(x, C) = distancia mínima al centroide más cercano
      let totalD = 0;
      const dists = new Float32Array(n);
      for (let i = 0; i < n; i++) {
        let minD = Infinity;
        for (const ct of centroids) minD = Math.min(minD, colorDistSq(pixels[i], ct));
        dists[i] = minD;
        totalD  += minD;
      }

      // Muestrear siguiente centroide ∝ D²
      let r = Math.random() * totalD;
      let chosen = pixels[n-1].slice();
      for (let i = 0; i < n; i++) {
        r -= dists[i];
        if (r <= 0) { chosen = pixels[i].slice(); break; }
      }
      centroids.push(chosen);
    }
    return centroids;
  }

  /**
   * Segmentar imagen con K-means.
   * Usa submuestreo (1 de cada `step` píxeles) para la fase de
   * entrenamiento y luego asigna todos los píxeles.
   *
   * Complejidad entrenamiento: O(maxIter · n/step · k)
   * Complejidad asignación:    O(n · k)
   *
   * @param {ImageData} imageData  - Datos de imagen a segmentar
   * @param {number} k             - Número de segmentos
   * @param {Function} progressCb  - Callback de progreso [0,1]
   * @returns {{assignments, centroids}}
   */
  segment(imageData, k, progressCb) {
    this.k       = k;
    this.width   = imageData.width;
    this.height  = imageData.height;
    this.selectedSet.clear();
    const d = imageData.data;
    const n = this.width * this.height;

    // ── Fase de muestreo (1 de cada 4 píxeles) ──────────
    const step   = 4;
    const sample = [];
    for (let i = 0; i < n; i += step) {
      sample.push([d[i*4], d[i*4+1], d[i*4+2]]);
    }

    // ── Inicialización K-means++ ──────────────────────────
    let centroids = this._kmeanspp(sample, k);
    const sampleA = new Uint8Array(sample.length);

    // ── Iteraciones E-M ──────────────────────────────────
    for (let iter = 0; iter < this.maxIter; iter++) {
      let changed = false;

      // E: Asignación — cada muestra al centroide más cercano
      for (let i = 0; i < sample.length; i++) {
        let minD = Infinity, best = 0;
        for (let j = 0; j < k; j++) {
          const d2 = colorDistSq(sample[i], centroids[j]);
          if (d2 < minD) { minD = d2; best = j; }
        }
        if (sampleA[i] !== best) { sampleA[i] = best; changed = true; }
      }

      if (!changed) break; // Convergencia

      // M: Actualización — recalcular centroide μₖ = (1/|Cₖ|) Σ x
      const sums   = Array.from({length: k}, () => [0, 0, 0]);
      const counts = new Array(k).fill(0);
      for (let i = 0; i < sample.length; i++) {
        const c = sampleA[i];
        sums[c][0] += sample[i][0];
        sums[c][1] += sample[i][1];
        sums[c][2] += sample[i][2];
        counts[c]++;
      }
      for (let j = 0; j < k; j++) {
        if (counts[j] > 0) {
          centroids[j] = [sums[j][0]/counts[j], sums[j][1]/counts[j], sums[j][2]/counts[j]];
        }
      }

      if (progressCb) progressCb((iter + 1) / this.maxIter * 0.8);
    }

    // ── Asignación final de TODOS los píxeles ─────────────
    this.assignments = new Uint8Array(n);
    for (let i = 0; i < n; i++) {
      const r = d[i*4], g = d[i*4+1], b = d[i*4+2];
      let minD = Infinity, best = 0;
      for (let j = 0; j < k; j++) {
        const d2 = colorDistSq([r, g, b], centroids[j]);
        if (d2 < minD) { minD = d2; best = j; }
      }
      this.assignments[i] = best;
    }

    this.centroids = centroids;
    if (progressCb) progressCb(1.0);
    return { assignments: this.assignments, centroids };
  }

  /** Obtener índice de cluster en posición (x, y) */
  getClusterAt(x, y) {
    if (!this.assignments) return -1;
    const idx = Math.floor(y) * this.width + Math.floor(x);
    return (idx >= 0 && idx < this.assignments.length) ? this.assignments[idx] : -1;
  }

  /**
   * Dibujar overlay de segmentos sobre un canvas.
   * Los clusters seleccionados se muestran con alta opacidad;
   * los demás con baja opacidad.
   *
   * @param {CanvasRenderingContext2D} ctx
   */
  drawOverlay(ctx) {
    const imgData = ctx.createImageData(this.width, this.height);
    const d = imgData.data;
    const hasSelection = this.selectedSet.size > 0;

    for (let i = 0; i < this.assignments.length; i++) {
      const c     = this.assignments[i];
      const color = this.palette[c % this.palette.length];
      const isSel = this.selectedSet.has(c);
      const alpha = hasSelection ? (isSel ? 210 : 35) : 150;
      d[i*4]   = color[0];
      d[i*4+1] = color[1];
      d[i*4+2] = color[2];
      d[i*4+3] = alpha;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  /**
   * Aplicar corrección de color solo a los clusters seleccionados.
   * El resto de píxeles permanece sin modificar.
   *
   * @param {ImageData} srcData  - Imagen original
   * @param {string} mode        - 'daltonize' | 'simulate' | 'enhance'
   * @param {number} intensity   - Factor α ∈ [0, 1]
   * @returns {ImageData}        - Imagen con corrección selectiva
   */
  applyToSelected(srcData, mode, intensity) {
    const result = new ImageData(
      new Uint8ClampedArray(srcData.data),
      srcData.width, srcData.height
    );
    const d = result.data;

    for (let i = 0; i < this.assignments.length; i++) {
      if (!this.selectedSet.has(this.assignments[i])) continue;
      const r = d[i*4], g = d[i*4+1], b = d[i*4+2];
      let nr, ng, nb;

      if      (mode === 'daltonize') [nr, ng, nb] = daltonize(r, g, b, intensity);
      else if (mode === 'simulate')  [nr, ng, nb] = simulateDeuteranopia(r, g, b);
      else                           [nr, ng, nb] = enhance(r, g, b, intensity);

      d[i*4] = nr; d[i*4+1] = ng; d[i*4+2] = nb;
    }
    return result;
  }

  /** Calcular inercia total J = Σ_k Σ_{x∈Cₖ} ‖x − μₖ‖² */
  computeInertia(pixels) {
    if (!this.assignments || !this.centroids) return 0;
    let J = 0;
    for (let i = 0; i < this.assignments.length; i++) {
      const c = this.assignments[i];
      J += colorDistSq(pixels[i], this.centroids[c]);
    }
    return J;
  }
}

/* ══════════════════════════════════════════════════════
   5. ESTADO GLOBAL
   ══════════════════════════════════════════════════════ */
const state = {
  originalImage : null,
  mode          : 'daltonize',
  intensity     : 1.0,
  channels      : { r: true, g: true, b: false },
  currentTab    : 'original'
};

const segState = {
  image         : null,
  segmenter     : new KMeansSegmenter(),
  k             : 5,
  mode          : 'daltonize',
  intensity     : 1.0,
  segmented     : false,
  origImageData : null  // cache del ImageData original de segmentación
};

/* ══════════════════════════════════════════════════════
   6. REFERENCIAS DOM — Herramienta principal
   ══════════════════════════════════════════════════════ */
const uploadZone        = document.getElementById('uploadZone');
const fileInput         = document.getElementById('fileInput');
const controlsPanel     = document.getElementById('controlsPanel');
const processBtn        = document.getElementById('processBtn');
const resetBtn          = document.getElementById('resetBtn');
const intensitySlider   = document.getElementById('intensitySlider');
const intensityVal      = document.getElementById('intensityVal');
const downloadBtn       = document.getElementById('downloadBtn');
const statsBar          = document.getElementById('statsBar');
const statDim           = document.getElementById('statDim');
const statPx            = document.getElementById('statPx');
const statMode          = document.getElementById('statMode');
const processingOverlay = document.getElementById('processingOverlay');

const canvasOriginal    = document.getElementById('canvasOriginal');
const canvasProcessed   = document.getElementById('canvasProcessed');
const canvasCompareL    = document.getElementById('canvasCompareL');
const canvasCompareR    = document.getElementById('canvasCompareR');
const viewOriginal      = document.getElementById('viewOriginal');
const viewProcessed     = document.getElementById('viewProcessed');
const viewCompare       = document.getElementById('viewCompare');
const canvasPlaceholder = document.getElementById('canvasPlaceholder');

/* ══════════════════════════════════════════════════════
   7. REFERENCIAS DOM — Segmentación K-means
   ══════════════════════════════════════════════════════ */
const segUploadZone    = document.getElementById('segUploadZone');
const segFileInput     = document.getElementById('segFileInput');
const segKSlider       = document.getElementById('segKSlider');
const segKVal          = document.getElementById('segKVal');
const segBtn           = document.getElementById('segBtn');
const segModeSelect    = document.getElementById('segModeSelect');
const segIntensity     = document.getElementById('segIntensity');
const segIntensityVal  = document.getElementById('segIntensityVal');
const applySegBtn      = document.getElementById('applySegBtn');
const downloadSegBtn   = document.getElementById('downloadSegBtn');
const segProgress      = document.getElementById('segProgress');
const segProgressBar   = document.getElementById('segProgressBar');
const segCanvasOrig    = document.getElementById('segCanvasOrig');
const segCanvasOverlay = document.getElementById('segCanvasOverlay');
const segCanvasResult  = document.getElementById('segCanvasResult');
const segPlaceholder   = document.getElementById('segPlaceholder');
const segResultArea    = document.getElementById('segResultArea');
const segLegend        = document.getElementById('segLegend');
const segInfoText      = document.getElementById('segInfoText');
const segInertiaVal    = document.getElementById('segInertiaVal');
const segClearBtn      = document.getElementById('segClearBtn');

/* ══════════════════════════════════════════════════════
   8. FUNCIONES DE CANVAS — Herramienta principal
   ══════════════════════════════════════════════════════ */

/** Escala y dibuja una imagen en un canvas (máx. 800px) */
function drawToCanvas(canvas, img, maxSize = 800) {
  let w = img.width, h = img.height;
  if (w > maxSize || h > maxSize) {
    const r = Math.min(maxSize/w, maxSize/h);
    w = Math.round(w * r);
    h = Math.round(h * r);
  }
  canvas.width = w; canvas.height = h;
  canvas.getContext('2d').drawImage(img, 0, 0, w, h);
}

/** Cargar imagen desde File y preparar UI */
function loadImage(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      state.originalImage = img;
      drawToCanvas(canvasOriginal, img);
      drawToCanvas(canvasCompareL, img);

      controlsPanel.style.display = 'flex';
      statsBar.style.display      = 'flex';
      canvasPlaceholder.style.display = 'none';
      viewOriginal.style.display  = 'flex';
      viewProcessed.style.display = 'none';
      viewCompare.style.display   = 'none';

      statDim.textContent  = `${img.width} × ${img.height}`;
      statPx.textContent   = '—';
      statMode.textContent = '—';
      downloadBtn.style.display = 'none';
      setTab('original');
      if (typeof updateStepGuide === 'function') updateStepGuide(2);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

/** Ejecutar procesamiento con overlay de espera */
function runProcessing() {
  if (!state.originalImage) return;
  processingOverlay.style.display = 'flex';
  setTab('processed');

  setTimeout(() => {
    drawToCanvas(canvasProcessed, state.originalImage);
    const ctx  = canvasProcessed.getContext('2d');
    const imgD = ctx.getImageData(0, 0, canvasProcessed.width, canvasProcessed.height);

    const result = processImage(imgD, state.mode, state.intensity, state.channels);
    ctx.putImageData(imgD, 0, 0);

    // Copiar al canvas de comparación (lado derecho)
    canvasCompareR.width  = canvasProcessed.width;
    canvasCompareR.height = canvasProcessed.height;
    canvasCompareR.getContext('2d').putImageData(imgD, 0, 0);

    // Actualizar stats
    const pct = ((result.affected / result.total) * 100).toFixed(1);
    statPx.textContent  = `${pct}% · ${result.affected.toLocaleString()} px`;
    const labels = { daltonize: 'Daltonize LMS', simulate: 'Simulación', enhance: 'Realce' };
    statMode.textContent = labels[state.mode];
    downloadBtn.style.display = 'inline-block';

    processingOverlay.style.display = 'none';
    if (typeof updateStepGuide === 'function') updateStepGuide(3);
    // Auto-cambiar a vista de comparación al procesar
    setTab('compare');
  }, 80);
}

/* ══════════════════════════════════════════════════════
   9. GESTIÓN DE TABS
   ══════════════════════════════════════════════════════ */
function setTab(tab) {
  state.currentTab = tab;
  document.querySelectorAll('.tab').forEach(t =>
    t.classList.toggle('active', t.dataset.tab === tab)
  );
  viewOriginal.style.display  = tab === 'original'  ? 'flex' : 'none';
  viewProcessed.style.display = tab === 'processed' ? 'flex' : 'none';
  viewCompare.style.display   = tab === 'compare'   ? 'flex' : 'none';
}

/* ══════════════════════════════════════════════════════
   10. VISTA DE COMPARACIÓN — Drag interactivo
   ══════════════════════════════════════════════════════ */
(function initCompareView() {
  let dragging = false;
  const divider = document.getElementById('compareDivider');
  if (!divider) return;

  function updateSplit(clientX) {
    const wrapper = viewCompare;
    const rect    = wrapper.getBoundingClientRect();
    const pct     = clamp((clientX - rect.left) / rect.width * 100, 5, 95);
    canvasCompareL.style.width = `${pct}%`;
    canvasCompareR.style.width = `${100 - pct}%`;
    divider.style.left = `calc(${pct}% - 16px)`;
  }

  divider.addEventListener('mousedown',  () => { dragging = true; });
  document.addEventListener('mousemove', e => { if (dragging) updateSplit(e.clientX); });
  document.addEventListener('mouseup',   () => { dragging = false; });
  divider.addEventListener('touchstart', e => { dragging = true; e.preventDefault(); }, { passive: false });
  document.addEventListener('touchmove', e => { if (dragging) updateSplit(e.touches[0].clientX); }, { passive: true });
  document.addEventListener('touchend',  () => { dragging = false; });
})();

/* ══════════════════════════════════════════════════════
   11. SEGMENTACIÓN K-MEANS — Lógica
   ══════════════════════════════════════════════════════ */

/** Coordenadas de canvas corregidas por escala CSS */
function getCanvasCoords(canvas, event) {
  const rect   = canvas.getBoundingClientRect();
  const scaleX = canvas.width  / rect.width;
  const scaleY = canvas.height / rect.height;
  const src    = event.touches ? event.touches[0] : event;
  return {
    x: Math.floor((src.clientX - rect.left) * scaleX),
    y: Math.floor((src.clientY - rect.top)  * scaleY)
  };
}

/** Cargar imagen en la sección de segmentación */
function loadSegImage(file) {
  const reader = new FileReader();
  reader.onload = e => {
    const img = new Image();
    img.onload = () => {
      segState.image    = img;
      segState.segmented = false;
      segState.origImageData = null;
      segState.segmenter.assignments = null;
      segState.segmenter.selectedSet.clear();

      drawToCanvas(segCanvasOrig, img, 700);

      // Limpiar overlays
      const octx = segCanvasOverlay.getContext('2d');
      segCanvasOverlay.width  = segCanvasOrig.width;
      segCanvasOverlay.height = segCanvasOrig.height;
      octx.clearRect(0, 0, segCanvasOverlay.width, segCanvasOverlay.height);

      segPlaceholder.style.display  = 'none';
      segResultArea.style.display   = 'flex';
      segLegend.style.display       = 'none';
      applySegBtn.disabled          = true;
      downloadSegBtn.style.display  = 'none';
      segInfoText.textContent       = 'Imagen cargada. Haz clic en "Segmentar" para ejecutar K-means.';

      // Caché del ImageData original
      const ctx = segCanvasOrig.getContext('2d');
      segState.origImageData = ctx.getImageData(0, 0, segCanvasOrig.width, segCanvasOrig.height);
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

/** Ejecutar K-means y dibujar overlay */
function runSegmentation() {
  if (!segState.image || !segState.origImageData) return;

  segBtn.disabled = true;
  segProgress.style.display = 'block';
  segInfoText.textContent   = 'Ejecutando K-means…';

  // Usar setTimeout para liberar el hilo y mostrar UI
  setTimeout(() => {
    const k = segState.k;

    segState.segmenter.segment(segState.origImageData, k, pct => {
      segProgressBar.style.width = `${Math.round(pct * 100)}%`;
    });

    segState.segmented = true;
    segState.segmenter.selectedSet.clear();

    // Dibujar overlay
    const octx = segCanvasOverlay.getContext('2d');
    segCanvasOverlay.width  = segCanvasOrig.width;
    segCanvasOverlay.height = segCanvasOrig.height;
    segState.segmenter.drawOverlay(octx);

    // Leyenda de clusters
    renderSegLegend();

    segProgress.style.display = 'none';
    segBtn.disabled           = false;
    applySegBtn.disabled      = true;
    downloadSegBtn.style.display = 'none';
    segInfoText.textContent   = `✓ ${k} segmentos encontrados. Haz clic en una región para seleccionarla.`;

    // Limpiar canvas de resultado
    const rctx = segCanvasResult.getContext('2d');
    rctx.clearRect(0, 0, segCanvasResult.width, segCanvasResult.height);
  }, 60);
}

/** Renderizar leyenda de colores por cluster */
function renderSegLegend() {
  segLegend.style.display = 'flex';
  segLegend.innerHTML = '<span class="seg-legend-label">Segmentos:</span>';
  const seg = segState.segmenter;

  for (let i = 0; i < seg.k; i++) {
    const color = seg.palette[i % seg.palette.length];
    const hex   = `rgb(${color[0]},${color[1]},${color[2]})`;
    const cent  = seg.centroids[i];
    const label = `#${i+1} (${cent.map(v => Math.round(v)).join(',')})`;

    const item = document.createElement('button');
    item.className = 'seg-legend-item';
    item.dataset.cluster = i;
    item.innerHTML = `<span class="seg-swatch" style="background:${hex}"></span><span>${label}</span>`;
    item.addEventListener('click', () => toggleCluster(i));
    segLegend.appendChild(item);
  }
}

/** Seleccionar / deseleccionar un cluster */
function toggleCluster(c) {
  const seg = segState.segmenter;
  if (seg.selectedSet.has(c)) seg.selectedSet.delete(c);
  else                         seg.selectedSet.add(c);

  // Redibujar overlay con selección actualizada
  const octx = segCanvasOverlay.getContext('2d');
  seg.drawOverlay(octx);

  // Actualizar estilos de leyenda
  document.querySelectorAll('.seg-legend-item').forEach(btn => {
    const idx = parseInt(btn.dataset.cluster);
    btn.classList.toggle('selected', seg.selectedSet.has(idx));
  });

  applySegBtn.disabled = seg.selectedSet.size === 0;
  const n = seg.selectedSet.size;
  segInfoText.textContent = n > 0
    ? `${n} segmento${n > 1 ? 's' : ''} seleccionado${n > 1 ? 's' : ''}. Haz clic en "Aplicar" para corregir.`
    : 'Haz clic en una región del canvas o en la leyenda para seleccionar.';
}

/** Click sobre el canvas de segmentación */
function handleSegCanvasClick(event) {
  if (!segState.segmented) return;
  const coords = getCanvasCoords(segCanvasOrig, event);
  const c = segState.segmenter.getClusterAt(coords.x, coords.y);
  if (c >= 0) toggleCluster(c);
}

/** Aplicar corrección a segmentos seleccionados */
function applySegmentCorrection() {
  if (!segState.segmented || segState.segmenter.selectedSet.size === 0) return;

  const result = segState.segmenter.applyToSelected(
    segState.origImageData,
    segState.mode,
    segState.intensity
  );

  segCanvasResult.width  = segCanvasOrig.width;
  segCanvasResult.height = segCanvasOrig.height;
  segCanvasResult.getContext('2d').putImageData(result, 0, 0);
  downloadSegBtn.style.display = 'inline-block';

  const n = segState.segmenter.selectedSet.size;
  segInfoText.textContent = `✓ Corrección aplicada a ${n} segmento${n > 1 ? 's' : ''}. Puedes descargar el resultado.`;
}

/* ══════════════════════════════════════════════════════
   12. EVENT LISTENERS — Herramienta principal
   ══════════════════════════════════════════════════════ */

// Drag & drop zona de carga
uploadZone.addEventListener('dragover',  e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', ()  => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file?.type.startsWith('image/')) loadImage(file);
});
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => { if (e.target.files[0]) loadImage(e.target.files[0]); });

// Botones de modo (compatibilidad)
document.querySelectorAll('.mode-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    state.mode = btn.dataset.mode;
  });
});

// Tarjetas de modo (nuevo UI)
document.querySelectorAll('.mode-card').forEach(card => {
  card.addEventListener('click', () => {
    document.querySelectorAll('.mode-card').forEach(c => c.classList.remove('active'));
    card.classList.add('active');
    state.mode = card.dataset.mode;
    // Sincronizar con botones ocultos
    document.querySelectorAll('.mode-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.mode === card.dataset.mode)
    );
  });
});

// Guía de pasos — activar visualmente
function updateStepGuide(step) {
  document.querySelectorAll('.tool-step-item').forEach((el, i) => {
    el.classList.remove('active', 'done');
    if (i + 1 < step) el.classList.add('done');
    if (i + 1 === step) el.classList.add('active');
  });
}

// Slider de intensidad
intensitySlider.addEventListener('input', e => {
  state.intensity = e.target.value / 100;
  intensityVal.textContent = `${e.target.value}%`;
});

// Checkboxes de canales
['R','G','B'].forEach(ch => {
  const el = document.getElementById(`ch${ch}`);
  if (el) el.addEventListener('change', e => { state.channels[ch.toLowerCase()] = e.target.checked; });
});

// Procesar y resetear
processBtn.addEventListener('click', runProcessing);

resetBtn.addEventListener('click', () => {
  state.originalImage = null;
  fileInput.value = '';
  controlsPanel.style.display = 'none';
  statsBar.style.display      = 'none';
  canvasPlaceholder.style.display = 'block';
  viewOriginal.style.display  = 'none';
  viewProcessed.style.display = 'none';
  viewCompare.style.display   = 'none';
  downloadBtn.style.display   = 'none';
  statDim.textContent = statPx.textContent = statMode.textContent = '—';
  intensitySlider.value = 100;
  state.intensity = 1.0;
  intensityVal.textContent = '100%';
  document.querySelectorAll('.mode-btn').forEach((b, i) => b.classList.toggle('active', i === 0));
  document.querySelectorAll('.mode-card').forEach((c, i) => c.classList.toggle('active', i === 0));
  state.mode = 'daltonize';
  if (typeof updateStepGuide === 'function') updateStepGuide(1);
});

// Tabs de canvas
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => setTab(t.dataset.tab)));

// Descarga
downloadBtn.addEventListener('click', () => {
  const a = document.createElement('a');
  a.download = 'visionclara_corregida.png';
  a.href = canvasProcessed.toDataURL('image/png');
  a.click();
});

/* ══════════════════════════════════════════════════════
   13. EVENT LISTENERS — Segmentación K-means
   ══════════════════════════════════════════════════════ */

// Carga de imagen en segmentación
segUploadZone.addEventListener('dragover',  e => { e.preventDefault(); segUploadZone.classList.add('drag-over'); });
segUploadZone.addEventListener('dragleave', ()  => segUploadZone.classList.remove('drag-over'));
segUploadZone.addEventListener('drop', e => {
  e.preventDefault();
  segUploadZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file?.type.startsWith('image/')) loadSegImage(file);
});
segUploadZone.addEventListener('click', () => segFileInput.click());
segFileInput.addEventListener('change', e => { if (e.target.files[0]) loadSegImage(e.target.files[0]); });

// Slider K
segKSlider.addEventListener('input', e => {
  segState.k = parseInt(e.target.value);
  segKVal.textContent = e.target.value;
});

// Modo y intensidad de segmentación
segModeSelect.addEventListener('change',  e => { segState.mode = e.target.value; });
segIntensity.addEventListener('input', e => {
  segState.intensity = e.target.value / 100;
  segIntensityVal.textContent = `${e.target.value}%`;
});

// Botón segmentar
segBtn.addEventListener('click', runSegmentation);

// Click en canvas de segmentación
segCanvasOrig.addEventListener('click',      handleSegCanvasClick);
segCanvasOverlay.addEventListener('click',   handleSegCanvasClick);

// Limpiar selección
segClearBtn?.addEventListener('click', () => {
  segState.segmenter.selectedSet.clear();
  if (segState.segmented) {
    const octx = segCanvasOverlay.getContext('2d');
    segState.segmenter.drawOverlay(octx);
  }
  document.querySelectorAll('.seg-legend-item').forEach(b => b.classList.remove('selected'));
  applySegBtn.disabled = true;
  segInfoText.textContent = 'Selección limpiada. Haz clic en una región para seleccionar.';
});

// Aplicar corrección y descargar
applySegBtn.addEventListener('click', applySegmentCorrection);

downloadSegBtn.addEventListener('click', () => {
  // Componer: original + corrección
  const a = document.createElement('a');
  a.download = 'visionclara_segmentada.png';
  a.href = segCanvasResult.toDataURL('image/png');
  a.click();
});

/* ══════════════════════════════════════════════════════
   14. NAV ACTIVO EN SCROLL
   ══════════════════════════════════════════════════════ */
const navObs = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (entry.isIntersecting) {
      document.querySelectorAll('.nav-link').forEach(l =>
        l.classList.toggle('active', l.getAttribute('href') === '#' + entry.target.id)
      );
    }
  });
}, { threshold: 0.35 });

document.querySelectorAll('section[id]').forEach(s => navObs.observe(s));