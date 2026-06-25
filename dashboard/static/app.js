const form = document.querySelector("#configForm");
const previewBtn = document.querySelector("#previewBtn");
const clearBtn = document.querySelector("#clearBtn");
const stopBtn = document.querySelector("#stopBtn");
const runBtn = document.querySelector("#runBtn");
const refreshResultsBtn = document.querySelector("#refreshResultsBtn");
const refreshChartsBtn = document.querySelector("#refreshChartsBtn");
const trainMlBtn = document.querySelector("#trainMlBtn");
const saveMlBtn = document.querySelector("#saveMlBtn");
const resetMlBtn = document.querySelector("#resetMlBtn");
const predictMlBtn = document.querySelector("#predictMlBtn");
const selectValidationBtn = document.querySelector("#selectValidationBtn");
const calibrateProxyBtn = document.querySelector("#calibrateProxyBtn");
const calibrationStatus = document.querySelector("#calibrationStatus");
const mlStatus = document.querySelector("#mlStatus");
const predictionResult = document.querySelector("#predictionResult");
const previewList = document.querySelector("#previewList");
const previewGallery = document.querySelector("#previewGallery");
const previewSearch = document.querySelector("#previewSearch");
const previewRunFilter = document.querySelector("#previewRunFilter");
const previewPageInfo = document.querySelector("#previewPageInfo");
const previewPagination = document.querySelector("#previewPagination");
const pushoverResults = document.querySelector("#pushoverResults");
const chartCanvas = document.querySelector("#chartCanvas");
const chartSummary = document.querySelector("#chartSummary");
const chartType = document.querySelector("#chartType");
const chartXMetric = document.querySelector("#chartXMetric");
const chartYMetric = document.querySelector("#chartYMetric");
const chartColorMetric = document.querySelector("#chartColorMetric");
const chartHideMissing = document.querySelector("#chartHideMissing");
const trainSomBtn = document.querySelector("#trainSomBtn");
const optimizeSomBtn = document.querySelector("#optimizeSomBtn");
const selectSomRepresentativeBtn = document.querySelector("#selectSomRepresentativeBtn");
const refreshSomBtn = document.querySelector("#refreshSomBtn");
const somWidth = document.querySelector("#somWidth");
const somHeight = document.querySelector("#somHeight");
const somIterations = document.querySelector("#somIterations");
const somMinGrid = document.querySelector("#somMinGrid");
const somMaxGrid = document.querySelector("#somMaxGrid");
const somResultMetric = document.querySelector("#somResultMetric");
const somMapMode = document.querySelector("#somMapMode");
const somSummary = document.querySelector("#somSummary");
const somCanvas = document.querySelector("#somCanvas");
const somDetails = document.querySelector("#somDetails");
const somClassTarget = document.querySelector("#somClassTarget");
const somBinCount = document.querySelector("#somBinCount");
const somCustomBins = document.querySelector("#somCustomBins");
const somClassAnalysis = document.querySelector("#somClassAnalysis");
const somXColumns = document.querySelector("#somXColumns");
const mlXColumns = document.querySelector("#mlXColumns");
const somLeakageWarning = document.querySelector("#somLeakageWarning");
const previewModal = document.querySelector("#previewModal");
const previewModalImage = document.querySelector("#previewModalImage");
const previewModalTitle = document.querySelector("#previewModalTitle");
const logBox = document.querySelector("#logBox");
const stateText = document.querySelector("#stateText");
const progressText = document.querySelector("#progressText");
const progressBar = document.querySelector("#progressBar");
const successCount = document.querySelector("#successCount");
const failedCount = document.querySelector("#failedCount");
const outputPath = document.querySelector("#outputPath");

let defaults = {};
let statusTimer = null;
let latestModelResults = [];
let latestSom = null;
let latestSomClassTargets = [];
let somClassRenderTimer = null;
let somRepresentativeView = null;
const OUTPUT_DIR_STORAGE_KEY = "sap2000_dashboard_output_dir";
const SHARED_X_COLUMNS_STORAGE_KEY = "sap2000_dashboard_shared_x_columns";
const PREVIEW_PAGE_SIZE = 50;
let sharedXColumns = null;
let previewCurrentPage = 1;
let previewSearchTimer = null;

const chartMetrics = {
  x: [
    { key: "total_height", label: "Toplam yükseklik", type: "number" },
    { key: "story_count", label: "Kat sayısı", type: "number" },
    { key: "target_drift", label: "Hedef drift", type: "number" },
    { key: "x_bay_count", label: "X açıklık sayısı", type: "number" },
    { key: "y_bay_count", label: "Y açıklık sayısı", type: "number" },
    { key: "avg_span_x", label: "Ort. X açıklık", type: "number" },
    { key: "avg_span_y", label: "Ort. Y açıklık", type: "number" },
    { key: "column_area", label: "Kolon kesit alanı", type: "number" },
    { key: "beam_depth", label: "Kiriş yüksekliği", type: "number" },
    { key: "rho_col", label: "Kolon donatı oranı", type: "number" },
    { key: "rho_beam_top", label: "Kiriş üst donatı oranı", type: "number" },
    { key: "rho_beam_bottom", label: "Kiriş alt donatı oranı", type: "number" },
    { key: "rho_slab", label: "Döşeme donatı oranı", type: "number" },
    { key: "rho_raft", label: "Radye donatı oranı", type: "number" },
    { key: "slab_thickness", label: "Döşeme kalınlığı", type: "number" },
    { key: "concrete_class", label: "Beton sınıfı", type: "category" },
    { key: "soil_class", label: "Zemin sınıfı", type: "category" },
    { key: "direction", label: "Yön", type: "category" },
  ],
  y: [
    { key: "peak_base_shear", label: "Maks. taban kesmesi", type: "number" },
    { key: "final_displacement", label: "Son tepe deplasmanı", type: "number" },
    { key: "max_rotation", label: "Maks. plastik rotasyon", type: "number" },
    { key: "max_story_drift_ratio", label: "Maksimum göreli kat ötelenmesi", type: "number", format: "percent" },
    { key: "critical_drift_story", label: "Kritik drift katı", type: "number" },
    { key: "first_hinge_type", label: "İlk mafsal tipi", type: "category" },
    { key: "first_hinge_story", label: "İlk plastik mafsal katı", type: "number" },
    { key: "first_hinge_plan_zone", label: "İlk plastik mafsal kenarda/ortada", type: "category" },
    { key: "critical_element_type", label: "Kritik eleman tipi", type: "category" },
    { key: "critical_element_story", label: "Kritik eleman katı", type: "number" },
    { key: "critical_element_plan_zone", label: "Kritik eleman kenarda/ortada", type: "category" },
    { key: "critical_state_rank", label: "Kritik hasar seviyesi", type: "number" },
    { key: "lscp_count", label: "LS-CP olay sayısı", type: "number" },
    { key: "cpc_count", label: "CP-C olay sayısı", type: "number" },
    { key: "first_plastic_step", label: "İlk plastik mafsal adımı", type: "number" },
    { key: "first_ls_step", label: "İlk LS adımı", type: "number" },
    { key: "first_cp_step", label: "İlk CP adımı", type: "number" },
    { key: "fema_ductility_mu", label: "FEMA 440 süneklik μ", type: "number" },
    { key: "fema_beta_eff_percent", label: "FEMA 440 etkin sönüm βeff", type: "number", format: "percent_value" },
    { key: "fema_teff_t0_ratio", label: "FEMA 440 Teff/T0", type: "number" },
    { key: "fema_r_proxy", label: "FEMA 440 R proxy", type: "number" },
    { key: "fema_target_capacity_ratio", label: "FEMA 440 hedef/kapasite oranı", type: "number" },
  ],
  color: [
    { key: "direction", label: "Yön", type: "category" },
    { key: "concrete_class", label: "Beton sınıfı", type: "category" },
    { key: "soil_class", label: "Zemin sınıfı", type: "category" },
    { key: "first_hinge_type", label: "İlk mafsal tipi", type: "category" },
    { key: "critical_element_type", label: "Kritik eleman tipi", type: "category" },
    { key: "critical_state", label: "Kritik hasar seviyesi", type: "category" },
    { key: "fema_capacity_status", label: "FEMA 440 kapasite durumu", type: "category" },
  ],
};

const defaultSomMetrics = {
  max_story_drift_ratio: { label: "Maksimum göreli kat ötelenmesi", type: "number", format: "percent" },
  peak_base_shear: { label: "Maksimum taban kesmesi", type: "number" },
  max_rotation: { label: "Maksimum plastik rotasyon", type: "number" },
  lscp_count: { label: "LS-CP olay sayısı", type: "number" },
  cpc_count: { label: "CP-C olay sayısı", type: "number" },
  first_hinge_type: { label: "İlk mafsal tipi", type: "category" },
  first_hinge_story: { label: "İlk plastik mafsal katı", type: "number" },
  first_hinge_plan_zone: { label: "İlk plastik mafsal kenarda/ortada", type: "category" },
  critical_element_type: { label: "Kritik eleman tipi", type: "category" },
  critical_element_story: { label: "Kritik eleman katı", type: "number" },
  critical_element_plan_zone: { label: "Kritik eleman kenarda/ortada", type: "category" },
  critical_state: { label: "Kritik hasar seviyesi", type: "category" },
  fema_ductility_mu: { label: "FEMA 440 süneklik μ", type: "number" },
  fema_beta_eff_percent: { label: "FEMA 440 etkin sönüm βeff", type: "number", format: "percent_value" },
  fema_teff_t0_ratio: { label: "FEMA 440 Teff/T0", type: "number" },
  fema_r_proxy: { label: "FEMA 440 R proxy", type: "number" },
  fema_c1: { label: "FEMA 440 C1", type: "number" },
  fema_target_displacement_proxy: { label: "FEMA 440 hedef deplasman proxy", type: "number" },
  fema_target_capacity_ratio: { label: "FEMA 440 hedef/kapasite oranı", type: "number" },
  fema_capacity_status: { label: "FEMA 440 kapasite durumu", type: "category" },
};

const somFeatureLabels = {
  story_count: "Kat sayısı",
  bay_count: "Açıklık sayısı",
  x_bay_count: "X açıklık sayısı",
  y_bay_count: "Y açıklık sayısı",
  avg_span: "Açıklık mesafesi",
  avg_span_x: "Ort. X açıklık",
  avg_span_y: "Ort. Y açıklık",
  max_span_x: "Maks. X açıklık",
  max_span_y: "Maks. Y açıklık",
  story_height: "Kat yüksekliği",
  total_height: "Toplam yükseklik",
  concrete_fck: "Beton fck",
  steel_fy: "Çelik fy",
  column_area: "Kolon alanı",
  beam_area: "Kiriş alanı",
  beam_depth: "Kiriş yüksekliği",
  rho_col: "Kolon donatı oranı",
  rho_beam_top: "Kiriş üst donatı oranı",
  rho_beam_bottom: "Kiriş alt donatı oranı",
  slab_thickness: "Döşeme kalınlığı",
  slab_rebar_ratio: "Döşeme donatı oranı",
  raft_rebar_ratio: "Radye donatı oranı",
  wall_area: "Perde alanı",
  subgrade_modulus: "Zemin yatak katsayısı",
  raft_thickness: "Radye kalınlığı",
  target_drift: "Hedef drift",
};

const somPercentFeatures = new Set(["rho_col", "rho_beam_top", "rho_beam_bottom", "slab_rebar_ratio", "raft_rebar_ratio", "target_drift"]);
const somResultKeys = new Set(Object.keys(defaultSomMetrics));

const numberFields = new Set([
  "n_iter",
  "random_seed",
  "max_attempts_per_model",
  "story_count_min",
  "story_count_max",
  "x_bay_count_min",
  "x_bay_count_max",
  "y_bay_count_min",
  "y_bay_count_max",
  "bay_length_min_m",
  "bay_length_max_m",
  "bay_length_step_m",
  "story_height_min_m",
  "story_height_max_m",
  "story_height_step_m",
  "slab_thickness_min_m",
  "slab_thickness_max_m",
  "slab_thickness_step_m",
  "min_slab_thickness_m",
  "max_slab_continuous_span_to_thickness_ratio",
  "slab_rho_min",
  "slab_rho_max",
  "raft_rho_min",
  "raft_rho_max",
  "rho_col_min",
  "rho_col_max",
  "rho_col_random_min",
  "rho_col_random_max",
  "rho_beam_min",
  "rho_beam_max",
  "rho_beam_random_min",
  "rho_beam_random_max",
  "min_column_dimension_m",
  "min_beam_width_m",
  "min_beam_depth_m",
  "max_span_to_beam_depth_ratio",
  "strong_column_factor",
  "dead_load_kn_m",
  "live_load_kn_m",
  "raft_thickness_min_m",
  "raft_thickness_max_m",
  "raft_thickness_step_m",
  "min_raft_thickness_m",
  "foundation_edge_offset_m",
  "wall_thickness_min_m",
  "wall_thickness_max_m",
  "wall_thickness_step_m",
  "wall_length_min_m",
  "wall_length_max_m",
  "wall_length_step_m",
  "wall_rho_min",
  "wall_rho_max",
  "min_wall_thickness_m",
  "min_wall_aspect_ratio",
  "pushover_target_drift_ratio_fixed",
  "pushover_target_drift_ratio_min",
  "pushover_target_drift_ratio_max",
  "pushover_target_drift_ratio_step",
  "pushover_base_shear_proxy_kn",
  "exact_hinge_export_max_attempts_per_model",
  "exact_hinge_export_min_rows_per_direction",
  "sap_screenshot_delay_s",
  "ml_random_seed"
]);

const intFields = new Set([
  "n_iter",
  "random_seed",
  "max_attempts_per_model",
  "exact_hinge_export_max_attempts_per_model",
  "exact_hinge_export_min_rows_per_direction",
  "story_count_min",
  "story_count_max",
  "x_bay_count_min",
  "x_bay_count_max",
  "y_bay_count_min",
  "y_bay_count_max",
  "ml_random_seed"
]);

const checkboxFields = new Set([
  "keep_sap_open",
  "enable_floor_slabs",
  "enable_raft_foundation",
  "assign_soil_vertical_springs",
  "enable_shear_walls",
  "enable_pushover_cases",
  "run_analysis_after_save",
  "fix_pushover_target_drift_ratio",
  "enable_plastic_hinges",
  "require_plastic_hinge_assignment",
  "assign_column_pmm_hinges",
  "assign_beam_m3_hinges",
  "assign_shear_hinges",
  "enable_exact_hinge_ui_export",
  "enable_sap_screenshots",
  "ml_auto_train_after_generation",
  "ml_preserve_existing_weights",
  "ml_use_som_features"
]);

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
    button.classList.add("active");
    document.querySelector(`[data-panel="${button.dataset.tab}"]`).classList.add("active");
  });
});

document.querySelector("#addColumnSection").addEventListener("click", () => addSectionRow("columnSections", { width: 0.5, depth: 0.5 }));
document.querySelector("#addBeamSection").addEventListener("click", () => addSectionRow("beamSections", { width: 0.3, depth: 0.6 }));
previewBtn.addEventListener("click", previewCandidates);
clearBtn.addEventListener("click", clearArtifacts);
stopBtn.addEventListener("click", stopRun);
runBtn.addEventListener("click", startRun);
previewRunFilter?.addEventListener("change", () => {
  previewCurrentPage = 1;
  refreshArtifacts();
});
previewSearch?.addEventListener("input", () => {
  window.clearTimeout(previewSearchTimer);
  previewSearchTimer = window.setTimeout(() => {
    previewCurrentPage = 1;
    refreshArtifacts();
  }, 250);
});
refreshResultsBtn?.addEventListener("click", refreshResults);
refreshChartsBtn?.addEventListener("click", refreshCharts);
trainSomBtn?.addEventListener("click", trainSom);
optimizeSomBtn?.addEventListener("click", optimizeSom);
selectSomRepresentativeBtn?.addEventListener("click", selectSomRepresentative);
refreshSomBtn?.addEventListener("click", refreshSom);
somResultMetric?.addEventListener("change", () => {
  somRepresentativeView = null;
  renderSom();
});
somMapMode?.addEventListener("change", () => {
  somRepresentativeView = null;
  renderSom();
});
somClassTarget?.addEventListener("change", scheduleSomClassAnalysisRender);
somBinCount?.addEventListener("change", renderSom);
somCustomBins?.addEventListener("change", renderSom);
[somXColumns, mlXColumns].forEach((container) => container?.addEventListener("change", syncSharedXColumnSelection));
trainMlBtn?.addEventListener("click", trainMlModel);
saveMlBtn?.addEventListener("click", saveMlSnapshot);
resetMlBtn?.addEventListener("click", resetMlModel);
predictMlBtn?.addEventListener("click", predictMlModel);
selectValidationBtn?.addEventListener("click", selectHingeValidationSubset);
calibrateProxyBtn?.addEventListener("click", calibrateHingeProxy);
[chartType, chartXMetric, chartYMetric, chartColorMetric, chartHideMissing].forEach((input) => input?.addEventListener("change", renderCharts));
form.elements.fix_pushover_target_drift_ratio?.addEventListener("change", updatePushoverDriftMode);
document.querySelectorAll("[data-close-preview]").forEach((button) => {
  button.addEventListener("click", closePreviewModal);
});

async function init() {
  try {
    populateChartControls();
    populateSomControls(defaultSomMetrics);
    defaults = await fetchJson("/api/defaults");
    hydrateForm(defaults);
    renderStatus(await fetchJson("/api/status"));
    await refreshMlStatus();
    await refreshSom();
    await refreshArtifacts();
    statusTimer = window.setInterval(pollStatus, 1500);
  } catch (error) {
    logBox.textContent = error.message;
  }
}

function hydrateForm(config) {
  Object.entries(config).forEach(([key, value]) => {
    const input = form.elements[key];
    if (!input || Array.isArray(value) || typeof value === "object") return;
    if (checkboxFields.has(key)) {
      input.checked = Boolean(value);
    } else {
      input.value = value ?? "";
    }
  });
  renderClassChips("concreteClasses", ["C25", "C30", "C35", "C40"], config.concrete_classes);
  renderClassChips("steelClasses", ["B420C", "B500C"], config.steel_classes);
  renderClassChips("pushoverDirections", ["X", "Y"], config.pushover_directions);
  renderSectionTable("columnSections", config.column_sections);
  renderSectionTable("beamSections", config.beam_sections);
  renderClassChips("soilClasses", ["ZA", "ZB", "ZC", "ZD", "ZE"], config.soil_classes || []);
  renderSoilModulusTable(config.soil_subgrade_modulus_kn_m3 || {});
  renderRatioInputs("columnRebarRatios", config.column_rebar_ratios || []);
  renderRatioInputs("beamTopRatioSupports", config.beam_top_ratio_supports || []);
  renderRatioInputs("beamBottomRatioSpans", config.beam_bottom_ratio_spans || []);
  renderRatioInputs("wallRebarRatios", config.wall_rebar_ratios || []);
  renderRatioInputs("slabRebarRatios", config.slab_rebar_ratios || []);
  renderRatioInputs("raftRebarRatios", config.raft_rebar_ratios || []);
  applyStoredOutputDir();
  updatePushoverDriftMode();
}

function applyStoredOutputDir() {
  const stored = readStoredOutputDir();
  if (stored && form.elements.output_dir) {
    form.elements.output_dir.value = stored;
  }
}

function readStoredOutputDir() {
  try {
    const value = window.localStorage.getItem(OUTPUT_DIR_STORAGE_KEY) || "";
    return String(value).trim();
  } catch {
    return "";
  }
}

function storeOutputDir(value) {
  const text = String(value || "").trim();
  if (!text) return;
  try {
    window.localStorage.setItem(OUTPUT_DIR_STORAGE_KEY, text);
  } catch {
    // no-op
  }
}

function updatePushoverDriftMode() {
  const fixed = Boolean(form.elements.fix_pushover_target_drift_ratio?.checked);
  ["pushover_target_drift_ratio_min", "pushover_target_drift_ratio_max", "pushover_target_drift_ratio_step"].forEach((name) => {
    const input = form.elements[name];
    if (input) input.disabled = fixed;
  });
  const fixedInput = form.elements.pushover_target_drift_ratio_fixed;
  if (fixedInput) fixedInput.disabled = !fixed;
}

function renderClassChips(containerId, allValues, activeValues) {
  const container = document.querySelector(`#${containerId}`);
  container.innerHTML = "";
  allValues.forEach((value) => {
    const label = document.createElement("label");
    label.className = "chip";
    label.innerHTML = `<input type="checkbox" value="${value}"><span>${value}</span>`;
    label.querySelector("input").checked = activeValues.includes(value);
    container.appendChild(label);
  });
}

function renderSectionTable(tableId, sections) {
  const table = document.querySelector(`#${tableId}`);
  table.innerHTML = "";
  const head = document.createElement("thead");
  head.innerHTML = "<tr><th>Genişlik m</th><th>Yükseklik m</th><th></th></tr>";
  table.appendChild(head);
  table.appendChild(document.createElement("tbody"));
  sections.forEach((section) => addSectionRow(tableId, section));
}

function renderSoilModulusTable(values) {
  const container = document.querySelector("#soilModulusTable");
  container.innerHTML = "";
  const notes = {
    ZA: "&Ccedil;ok sert kaya/zemin kabul&uuml;ne yak&#305;n; yayg&#305;n parametrik ks: 100000-150000 kN/m3.",
    ZB: "Sert zemin/kaya ayr&#305;&#351;mas&#305; kabul&uuml;ne yak&#305;n; yayg&#305;n parametrik ks: 70000-100000 kN/m3.",
    ZC: "Orta s&#305;k&#305;/orta sert zemin senaryosu; yayg&#305;n parametrik ks: 40000-70000 kN/m3.",
    ZD: "Daha yumu&#351;ak zemin senaryosu; yayg&#305;n parametrik ks: 20000-40000 kN/m3.",
    ZE: "&Ccedil;ok yumu&#351;ak/problemli zemin senaryosu; geoteknik rapor kritik, yayg&#305;n parametrik ks: 10000-20000 kN/m3."
  };
  ["ZA", "ZB", "ZC", "ZD", "ZE"].forEach((soilClass) => {
    const label = document.createElement("label");
    label.innerHTML = `${soilClass}<input data-soil-modulus="${soilClass}" type="number" min="1" step="1000" value="${values[soilClass] ?? 0}"><span class="field-note">${notes[soilClass]}</span>`;
    container.appendChild(label);
  });
}

function renderRatioInputs(containerId, values) {
  const container = document.querySelector(`#${containerId}`);
  if (!container) return;
  container.innerHTML = "";
  values.forEach((value) => addRatioInput(container, value, true));
  const button = document.createElement("button");
  button.type = "button";
  button.className = "add-ratio-btn";
  button.textContent = "Ekle";
  button.addEventListener("click", () => addRatioInput(container, 0.01, true));
  container.appendChild(button);
}

function addRatioInput(container, value, checked = true) {
  const wrap = document.createElement("label");
  wrap.className = "ratio-input";
  wrap.innerHTML = `
    <input class="ratio-enabled" type="checkbox" ${checked ? "checked" : ""}>
    <input class="ratio-value" type="number" min="0" step="0.001" value="${value}">
    <button type="button" title="Sil">Sil</button>
  `;
  wrap.querySelector("button").addEventListener("click", () => wrap.remove());
  const addButton = container.querySelector(".add-ratio-btn");
  if (addButton) {
    container.insertBefore(wrap, addButton);
  } else {
    container.appendChild(wrap);
  }
}

function addSectionRow(tableId, section) {
  const table = document.querySelector(`#${tableId}`);
  let tbody = table.querySelector("tbody");
  if (!tbody) {
    tbody = document.createElement("tbody");
    table.appendChild(tbody);
  }
  const row = document.createElement("tr");
  row.innerHTML = `
    <td><input class="section-width" type="number" min="0.01" step="0.01" value="${section.width}"></td>
    <td><input class="section-depth" type="number" min="0.01" step="0.01" value="${section.depth}"></td>
    <td><button type="button" class="remove-row" title="Satırı sil">Sil</button></td>
  `;
  row.querySelector(".remove-row").addEventListener("click", () => row.remove());
  tbody.appendChild(row);
}

function collectConfig() {
  const config = structuredClone(defaults);
  Object.keys(defaults).forEach((key) => {
    const input = form.elements[key];
    if (!input || Array.isArray(defaults[key]) || typeof defaults[key] === "object") return;
    if (checkboxFields.has(key)) {
      config[key] = input.checked;
    } else if (numberFields.has(key)) {
      config[key] = input.value === "" ? null : Number(input.value);
      if (intFields.has(key) && config[key] !== null) config[key] = Math.trunc(config[key]);
    } else {
      config[key] = input.value;
    }
  });
  config.concrete_classes = checkedValues("concreteClasses");
  config.steel_classes = checkedValues("steelClasses");
  config.pushover_directions = checkedValues("pushoverDirections");
  config.soil_classes = checkedValues("soilClasses");
  config.soil_subgrade_modulus_kn_m3 = collectSoilModulus();
  config.column_rebar_ratios = collectRatioInputs("columnRebarRatios");
  config.beam_top_ratio_supports = collectRatioInputs("beamTopRatioSupports");
  config.beam_bottom_ratio_spans = collectRatioInputs("beamBottomRatioSpans");
  config.wall_rebar_ratios = collectRatioInputs("wallRebarRatios");
  config.slab_rebar_ratios = collectRatioInputs("slabRebarRatios");
  config.raft_rebar_ratios = collectRatioInputs("raftRebarRatios");
  config.column_sections = collectSections("columnSections");
  config.beam_sections = collectSections("beamSections");
  storeOutputDir(config.output_dir);
  return config;
}

function checkedValues(containerId) {
  return Array.from(document.querySelectorAll(`#${containerId} input:checked`)).map((input) => input.value);
}

function collectSections(tableId) {
  return Array.from(document.querySelectorAll(`#${tableId} tbody tr`)).map((row) => ({
    width: Number(row.querySelector(".section-width").value),
    depth: Number(row.querySelector(".section-depth").value)
  }));
}

function collectSoilModulus() {
  const values = {};
  document.querySelectorAll("[data-soil-modulus]").forEach((input) => {
    values[input.dataset.soilModulus] = Number(input.value);
  });
  return values;
}

function collectRatioInputs(containerId) {
  return Array.from(document.querySelectorAll(`#${containerId} .ratio-input`))
    .filter((wrap) => wrap.querySelector(".ratio-enabled").checked)
    .map((wrap) => Number(wrap.querySelector(".ratio-value").value))
    .filter((value) => Number.isFinite(value) && value > 0);
}

async function previewCandidates() {
  previewBtn.disabled = true;
  previewList.innerHTML = "";
  try {
    const response = await postJson("/api/preview", { config: collectConfig() });
    renderPreview(response.candidates);
  } catch (error) {
    previewList.innerHTML = `<div class="preview-item invalid"><div class="preview-title">${escapeHtml(error.message)}</div></div>`;
  } finally {
    previewBtn.disabled = false;
  }
}

function renderPreview(candidates) {
  previewList.innerHTML = "";
  candidates.forEach((candidate) => {
    const item = document.createElement("div");
    item.className = `preview-item${candidate.valid ? "" : " invalid"}`;
    const reason = candidate.reasons.length ? `<br>${escapeHtml(candidate.reasons.join("; "))}` : "";
    item.innerHTML = `
      <div class="preview-title">${escapeHtml(candidate.name)}</div>
      <div class="preview-meta">
        ${candidate.story_count} kat, X${candidate.x_bays} Y${candidate.y_bays}, ${candidate.concrete},
        kolon ${candidate.column}, kiriş ${candidate.beam},
        rhoC ${candidate.rho_col}, üst ${candidate.beam_top_ratio_support}, alt ${candidate.beam_bottom_ratio_span}, push ${candidate.push_drift},
        döşeme ${candidate.slab_thickness_m} m, rhoS ${candidate.slab_rebar_ratio}, zemin ${candidate.soil_class}, radye ${candidate.raft_thickness_m} m, rhoR ${candidate.raft_rebar_ratio}, ks ${candidate.subgrade_modulus_kn_m3},
        perde ${candidate.has_shear_walls ? `${candidate.wall_thickness_m} m x ${candidate.wall_length_m} m, rhoW ${candidate.wall_rebar_ratio}` : "yok"}
        ${reason}
      </div>
    `;
    previewList.appendChild(item);
  });
}

async function startRun() {
  runBtn.disabled = true;
  try {
    const response = await postJson("/api/run", { config: collectConfig() });
    if (response.status) {
      renderStatus(response.status);
    }
    await pollStatus();
  } catch (error) {
    logBox.textContent = error.message;
  } finally {
    window.setTimeout(async () => {
      try {
        const status = await fetchJson("/api/status");
        renderStatus(status);
      } catch {
        runBtn.disabled = false;
      }
    }, 1200);
  }
}

async function stopRun() {
  stopBtn.disabled = true;
  try {
    await postJson("/api/stop", { config: collectConfig() });
    await pollStatus();
  } catch (error) {
    logBox.textContent = error.message;
  }
}

async function clearArtifacts() {
  const config = collectConfig();
  const ok = window.confirm(`${config.output_dir} içindeki tüm run klasörleri ve üretilmiş model/metadata dosyaları silinecek. ML dosyaları korunur. Devam edilsin mi?`);
  if (!ok) return;
  clearBtn.disabled = true;
  try {
    await postJson("/api/clear-artifacts", { config });
    previewList.innerHTML = "";
    await refreshArtifacts();
    await pollStatus();
  } catch (error) {
    logBox.textContent = error.message;
  } finally {
    clearBtn.disabled = false;
  }
}

async function pollStatus() {
  const status = await fetchJson("/api/status");
  renderStatus(status);
  if (status.effective_running || status.running || status.current > 0) {
    await refreshArtifacts();
  }
}

function renderStatus(status) {
  const total = status.total || 0;
  const current = status.current || 0;
  const busy = Boolean(status.effective_running ?? status.running);
  const percent = total ? Math.min(100, Math.round((current / total) * 100)) : 0;
  stateText.textContent = busy
    ? status.cancel_requested ? "Durduruluyor" : "Çalışıyor"
    : status.error ? "Hata" : "Hazır";
  progressText.textContent = `${current} / ${total}`;
  progressBar.style.width = `${percent}%`;
  successCount.textContent = status.success_count;
  failedCount.textContent = status.failed_count;
  outputPath.textContent = status.metadata_csv || "";
  logBox.textContent = status.logs.join("\n");
  logBox.scrollTop = logBox.scrollHeight;
  runBtn.disabled = busy;
  stopBtn.disabled = !busy || status.cancel_requested;
  clearBtn.disabled = busy;
}

async function refreshArtifacts() {
  const offset = Math.max(0, (previewCurrentPage - 1) * PREVIEW_PAGE_SIZE);
  const params = new URLSearchParams({
    offset: String(offset),
    limit: String(PREVIEW_PAGE_SIZE),
    search: previewSearch?.value || "",
    run: previewRunFilter?.value || "",
  });
  const inventory = await fetchJson(`/api/artifacts?${params.toString()}`);
  const total = Number(inventory.preview_total_count ?? inventory.preview_count ?? 0);
  const totalPages = Math.max(1, Math.ceil(total / PREVIEW_PAGE_SIZE));
  if (total && previewCurrentPage > totalPages) {
    previewCurrentPage = totalPages;
    return refreshArtifacts();
  }
  const results = await refreshResults();
  const modelsByName = new Map((results.models || []).map((model) => [model.name, model]));
  renderPreviewRunFilter(inventory.preview_runs || []);
  renderPreviewPageInfo(inventory);
  renderPreviewPagination(inventory);
  if (!inventory.files.length || !inventory.previews.length) {
    const runningMessage = String(inventory.message || "").toLowerCase().includes("devam");
    if (!runningMessage || !previewGallery.children.length) {
      previewGallery.innerHTML = `<div class="preview-item"><div class="preview-meta">${escapeHtml(inventory.message || "Henuz model gorseli yok.")}</div></div>`;
    }
    return;
  }
  previewGallery.innerHTML = "";
  inventory.previews.forEach((file) => {
    const card = document.createElement("div");
    card.className = "preview-card";
    card.tabIndex = 0;
    card.setAttribute("role", "button");
    const src = `/api/artifact?name=${encodeURIComponent(file.name)}`;
    const modelName = file.model_name || String(file.basename || file.name).replace("_preview.svg", "");
    const model = modelsByName.get(modelName);
    const previewMarkup = model ? renderHighlightedPreview(model) : `<img src="${src}" alt="${escapeHtml(file.name)}">`;
    card.innerHTML = `
      ${previewMarkup}
      <div>
        <span>${escapeHtml(modelName)}</span>
        <small>${escapeHtml(file.run_id || "")}</small>
        <button type="button" class="open-model-btn" data-model-name="${escapeHtml(file.name)}">SAP2000'de Aç</button>
      </div>
    `;
    const modalSrc = model ? svgMarkupToDataUrl(previewMarkup) : src;
    card.addEventListener("click", () => openPreviewModal(file.name, modalSrc));
    card.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openPreviewModal(file.name, modalSrc);
      }
    });
    const openButton = card.querySelector(".open-model-btn");
    openButton.addEventListener("click", (event) => {
      event.stopPropagation();
      openSapModel(openButton.dataset.modelName);
    });
    previewGallery.appendChild(card);
  });
}

function renderPreviewRunFilter(runs) {
  if (!previewRunFilter) return;
  const current = previewRunFilter.value;
  const options = ['<option value="">Tüm run klasörleri</option>']
    .concat((runs || []).map((run) => `<option value="${escapeHtml(run)}">${escapeHtml(run)}</option>`));
  previewRunFilter.innerHTML = options.join("");
  previewRunFilter.value = runs.includes(current) ? current : "";
}

function renderPreviewPageInfo(inventory) {
  if (!previewPageInfo) return;
  const shown = Number(inventory.previews?.length || 0);
  const total = Number(inventory.preview_total_count ?? inventory.preview_count ?? 0);
  const offset = Number(inventory.preview_offset || 0);
  const start = total && shown ? offset + 1 : 0;
  const end = total && shown ? offset + shown : 0;
  previewPageInfo.textContent = total ? `${start}-${end} / ${total} gösteriliyor` : "";
}

function renderPreviewPagination(inventory) {
  if (!previewPagination) return;
  const total = Number(inventory.preview_total_count ?? inventory.preview_count ?? 0);
  const totalPages = Math.max(1, Math.ceil(total / PREVIEW_PAGE_SIZE));
  if (!total || totalPages <= 1) {
    previewPagination.innerHTML = "";
    return;
  }
  const pageButtons = [];
  const addButton = (page, label = String(page), disabled = false, active = false) => {
    pageButtons.push(`
      <button type="button" data-preview-page="${page}" ${disabled ? "disabled" : ""} class="${active ? "active" : ""}">
        ${escapeHtml(label)}
      </button>
    `);
  };
  const pageSet = new Set([1, totalPages]);
  for (let page = previewCurrentPage - 2; page <= previewCurrentPage + 2; page += 1) {
    if (page >= 1 && page <= totalPages) pageSet.add(page);
  }
  const pages = Array.from(pageSet).sort((a, b) => a - b);
  addButton(previewCurrentPage - 1, "Önceki", previewCurrentPage <= 1);
  pages.forEach((page, index) => {
    if (index > 0 && page - pages[index - 1] > 1) {
      addButton(pages[index - 1], "...", true);
    }
    addButton(page, String(page), false, page === previewCurrentPage);
  });
  addButton(previewCurrentPage + 1, "Sonraki", previewCurrentPage >= totalPages);
  previewPagination.innerHTML = pageButtons.join("");
  previewPagination.querySelectorAll("button[data-preview-page]").forEach((button) => {
    button.addEventListener("click", () => {
      const nextPage = Number(button.dataset.previewPage || 1);
      if (!Number.isFinite(nextPage) || nextPage < 1 || nextPage > totalPages || nextPage === previewCurrentPage) return;
      previewCurrentPage = nextPage;
      refreshArtifacts();
    });
  });
}

async function refreshResults() {
  const results = await fetchJson("/api/model-results");
  const models = results.models || [];
  const runningMessage = String(results.message || "").toLowerCase().includes("devam");
  if (!models.length && runningMessage && latestModelResults.length) {
    renderPushoverResults(latestModelResults, results.message || "");
    renderCharts(latestModelResults);
    return { ...results, models: latestModelResults };
  }
  latestModelResults = models;
  renderPushoverResults(models, results.message || "");
  renderCharts();
  return results;
}

async function refreshCharts() {
  const results = await refreshResults();
  renderCharts(results.models || []);
}

function populateChartControls() {
  fillSelect(chartXMetric, chartMetrics.x, "total_height");
  fillSelect(chartYMetric, chartMetrics.y, "peak_base_shear");
  fillSelect(chartColorMetric, chartMetrics.color, "concrete_class");
}

function populateSomControls(metrics) {
  if (!somResultMetric) return;
  const mergedMetrics = { ...(metrics || {}), ...defaultSomMetrics };
  const entries = Object.entries(mergedMetrics).map(([key, meta]) => ({ key, label: meta.label || key }));
  fillSelect(somResultMetric, entries, "max_story_drift_ratio");
}

function renderSomXColumns(status) {
  const columns = status?.x_columns || {};
  const available = new Set(Object.keys(columns));
  if (sharedXColumns === null) {
    const stored = normalizeSharedXColumns(readSharedXColumns(), available);
    const fallback = normalizeSharedXColumns(status?.selected_x_columns || status?.default_x_columns || Object.keys(columns), available);
    sharedXColumns = new Set(stored.length ? stored : fallback);
  } else {
    const normalized = normalizeSharedXColumns([...sharedXColumns], available);
    sharedXColumns = new Set(normalized.length ? normalized : (status?.default_x_columns || Object.keys(columns)));
  }
  const entries = Object.entries(columns).map(([key, meta]) => ({ key, label: meta.label || key, kind: meta.kind || "" }));
  const markup = entries.map((item) => `
    <label class="som-x-chip">
      <input type="checkbox" value="${escapeHtml(item.key)}" ${sharedXColumns.has(item.key) ? "checked" : ""}>
      <span>${escapeHtml(item.label)}</span>
      <em>${escapeHtml(item.kind)}</em>
    </label>
  `).join("");
  if (somXColumns) somXColumns.innerHTML = markup;
  if (mlXColumns) mlXColumns.innerHTML = markup;
}

function selectedSomXColumns() {
  if (sharedXColumns !== null) return [...sharedXColumns];
  if (!somXColumns) return [];
  return [...somXColumns.querySelectorAll("input[type='checkbox']:checked")].map((input) => input.value);
}

function syncSharedXColumnSelection(event) {
  const input = event.target;
  if (!(input instanceof HTMLInputElement) || input.type !== "checkbox") return;
  if (sharedXColumns === null) sharedXColumns = new Set(selectedSomXColumns());
  if (input.checked) sharedXColumns.add(input.value);
  else sharedXColumns.delete(input.value);
  [somXColumns, mlXColumns].forEach((container) => {
    const peer = [...(container?.querySelectorAll("input[type='checkbox']") || [])].find((item) => item.value === input.value);
    if (peer) peer.checked = input.checked;
  });
  window.localStorage.setItem(SHARED_X_COLUMNS_STORAGE_KEY, JSON.stringify([...sharedXColumns]));
}

function readSharedXColumns() {
  try {
    const value = JSON.parse(window.localStorage.getItem(SHARED_X_COLUMNS_STORAGE_KEY) || "[]");
    return Array.isArray(value) ? value.map(String) : [];
  } catch (_error) {
    return [];
  }
}

function normalizeSharedXColumns(columns, available) {
  const aliases = {
    x_bay_count: "bay_count",
    y_bay_count: "bay_count",
    avg_span_x: "avg_span",
    avg_span_y: "avg_span",
    max_span: "avg_span",
    max_span_x: "avg_span",
    max_span_y: "avg_span",
  };
  const removed = new Set(["initial_stiffness_proxy", "design_base_shear_ratio_proxy", "total_mass_proxy"]);
  const normalized = [];
  (columns || []).forEach((key) => {
    const mapped = aliases[key] || key;
    if (!removed.has(key) && available.has(mapped) && !normalized.includes(mapped)) normalized.push(mapped);
  });
  return normalized;
}

function validateSomSelection() {
  const selected = selectedSomXColumns();
  const leaked = selected.filter((key) => somResultKeys.has(key));
  if (!selected.length) return "SOM eğitimi için en az bir X parametresi seçmelisin.";
  if (leaked.length) return `Bu değişkenler Y sonucudur ve X eğitim matrisine alınamaz: ${leaked.join(", ")}`;
  return "";
}

function fillSelect(select, metrics, selectedKey) {
  if (!select) return;
  select.innerHTML = metrics.map((metric) => `<option value="${escapeHtml(metric.key)}">${escapeHtml(metric.label)}</option>`).join("");
  select.value = selectedKey;
}

async function refreshSom() {
  if (!somCanvas) return null;
  const status = await fetchJson("/api/som/status");
  latestSom = status;
  populateSomControls(status.result_metrics || defaultSomMetrics);
  if (status.result_metric && somResultMetric) somResultMetric.value = status.result_metric;
  renderSomXColumns(status);
  if (somLeakageWarning) somLeakageWarning.textContent = status.target_leakage_warning || "Y sonuç değişkenleri SOM eğitim matrisine alınmaz; hedef sızıntısı engellenir.";
  renderSom();
  return status;
}

async function trainSom() {
  if (!somCanvas) return;
  const selectionError = validateSomSelection();
  if (selectionError) {
    somCanvas.innerHTML = `<div class="error-box">${escapeHtml(selectionError)}</div>`;
    return;
  }
  somCanvas.innerHTML = `<div class="muted-box">SOM eğitiliyor...</div>`;
  somSummary.innerHTML = "";
  try {
    const response = await postJson("/api/som/train", {
      config: collectConfig(),
      width: Number(somWidth?.value || 8),
      height: Number(somHeight?.value || 8),
      iterations: Number(somIterations?.value || 800),
      result_metric: somResultMetric?.value || "max_story_drift_ratio",
      random_seed: Number(document.querySelector("#mlRandomSeed")?.value || 42),
      selected_x_columns: selectedSomXColumns(),
    }, 180000);
    latestSom = response.result;
    populateSomControls(latestSom.result_metrics || defaultSomMetrics);
    if (latestSom.result_metric && somResultMetric) somResultMetric.value = latestSom.result_metric;
    renderSomXColumns(latestSom);
    if (somLeakageWarning) somLeakageWarning.textContent = latestSom.target_leakage_warning || latestSom.source_note || "";
    renderSom();
  } catch (error) {
    somCanvas.innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  }
}

async function optimizeSom() {
  if (!somCanvas) return;
  const selectionError = validateSomSelection();
  if (selectionError) {
    somCanvas.innerHTML = `<div class="error-box">${escapeHtml(selectionError)}</div>`;
    return;
  }
  somCanvas.innerHTML = `<div class="muted-box">SOM boyutu optimize ediliyor... Bu işlem özellikle büyük grid ve yüksek iterasyonda birkaç dakika sürebilir. Sayfayı kapatma.</div>`;
  if (somSummary) somSummary.innerHTML = "";
  if (somDetails) somDetails.innerHTML = "";
  const minSize = Number(somMinGrid?.value || 3);
  const maxSize = Number(somMaxGrid?.value || 8);
  const iterations = Number(somIterations?.value || 800);
  if (minSize > maxSize) {
    somCanvas.innerHTML = `<div class="error-box">Min grid, maks grid değerinden büyük olamaz.</div>`;
    return;
  }
  try {
    const response = await postJson("/api/som/optimize", {
      config: collectConfig(),
      start_size: minSize,
      max_size: maxSize,
      iterations,
      result_metric: somResultMetric?.value || "max_story_drift_ratio",
      random_seed: Number(document.querySelector("#mlRandomSeed")?.value || 42),
      selected_x_columns: selectedSomXColumns(),
    }, 0);
    latestSom = response.result;
    if (latestSom?.optimization?.selected_width && somWidth) {
      somWidth.value = String(latestSom.optimization.selected_width);
    }
    if (latestSom?.optimization?.selected_height && somHeight) {
      somHeight.value = String(latestSom.optimization.selected_height);
    }
    populateSomControls(latestSom.result_metrics || defaultSomMetrics);
    if (latestSom.result_metric && somResultMetric) somResultMetric.value = latestSom.result_metric;
    renderSomXColumns(latestSom);
    if (somLeakageWarning) somLeakageWarning.textContent = latestSom.target_leakage_warning || latestSom.source_note || "";
    renderSom();
  } catch (error) {
    somCanvas.innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  }
}

function selectSomRepresentative() {
  if (!latestSom?.available) {
    if (somCanvas) somCanvas.innerHTML = `<div class="muted-box">Temsilci seçimi için önce SOM eğitilmelidir.</div>`;
    return;
  }
  const metricKey = somResultMetric?.value || latestSom.result_metric || "max_story_drift_ratio";
  const metrics = latestSom.result_metrics || defaultSomMetrics;
  const metric = metrics[metricKey] || defaultSomMetrics.max_story_drift_ratio;
  const representative = computeSomRepresentative(latestSom, metricKey, metric);
  if (!representative) {
    if (somDetails) somDetails.innerHTML = `<div class="muted-box">Temsilci seçilecek yeterli dolu hücre bulunamadı.</div>`;
    return;
  }
  somRepresentativeView = representative;
  renderSom();
}

async function refreshMlStatus() {
  if (!mlStatus) return null;
  const status = await fetchJson("/api/ml/status");
  renderMlStatus(status);
  return status;
}

async function trainMlModel() {
  if (!mlStatus) return;
  mlStatus.innerHTML = `<div class="muted-box">Model eğitiliyor...</div>`;
  try {
    const response = await postJson("/api/ml/train", {
      config: collectConfig(),
      algorithm: document.querySelector("#mlAlgorithm")?.value || "knn",
      random_seed: Number(document.querySelector("#mlRandomSeed")?.value || 42),
      preserve_existing: Boolean(form.elements.ml_preserve_existing_weights?.checked),
      use_som_features: Boolean(form.elements.ml_use_som_features?.checked),
      selected_x_columns: selectedSomXColumns()
    });
    renderMlStatus(response.result);
  } catch (error) {
    mlStatus.innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  }
}

async function saveMlSnapshot() {
  if (!mlStatus) return;
  try {
    const response = await postJson("/api/ml/save-snapshot", { config: collectConfig(), algorithm: document.querySelector("#mlAlgorithm")?.value || "knn" });
    const saved = response.result?.saved || [];
    mlStatus.insertAdjacentHTML("afterbegin", `<div class="muted-box">Model saklandı: ${saved.map(escapeHtml).join("<br>")}</div>`);
  } catch (error) {
    mlStatus.insertAdjacentHTML("afterbegin", `<div class="error-box">${escapeHtml(error.message)}</div>`);
  }
}

async function resetMlModel() {
  const algorithm = document.querySelector("#mlAlgorithm")?.value || "knn";
  if (!window.confirm(`${algorithm} ML modeli silinecek. SAP2000 model çıktıları korunacak. Devam edilsin mi?`)) return;
  if (!mlStatus) return;
  try {
    await postJson("/api/ml/reset", { config: collectConfig(), algorithm });
    await refreshMlStatus();
  } catch (error) {
    mlStatus.innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  }
}

async function predictMlModel() {
  if (!predictionResult) return;
  predictionResult.innerHTML = `<div class="muted-box">Tahmin hesaplanıyor...</div>`;
  if (predictMlBtn) predictMlBtn.disabled = true;
  try {
    const baseParams = collectPredictionParams();
    const config = collectConfig();
    const algorithm = document.querySelector("#predictAlgorithm")?.value || "knn";
    const [xResponse, yResponse] = await Promise.all([
      postJson("/api/ml/predict", { config, algorithm, params: { ...baseParams, direction: "X" } }, 12000),
      postJson("/api/ml/predict", { config, algorithm, params: { ...baseParams, direction: "Y" } }, 12000)
    ]);
    renderPrediction({ input: baseParams, directions: { X: xResponse.result, Y: yResponse.result } });
  } catch (error) {
    predictionResult.innerHTML = `<div class="error-box">${escapeHtml(error.message)}</div>`;
  } finally {
    if (predictMlBtn) predictMlBtn.disabled = false;
  }
}

async function selectHingeValidationSubset() {
  if (!calibrationStatus) return;
  calibrationStatus.textContent = "Doğrulama listesi hazırlanıyor...";
  try {
    const response = await postJson("/api/hinge-validation/select", { config: collectConfig(), count: 30 });
    const result = response.result || {};
    calibrationStatus.textContent = `${result.selected_count || 0} model seçildi. Liste: ${result.manifest_path || "-"}`;
  } catch (error) {
    calibrationStatus.textContent = error.message;
  }
}

async function calibrateHingeProxy() {
  if (!calibrationStatus) return;
  calibrationStatus.textContent = "SAP exportları okunuyor ve proxy kalibre ediliyor...";
  try {
    const response = await postJson("/api/hinge-validation/calibrate", { config: collectConfig() }, 120000);
    const report = response.result?.report || {};
    const calibration = response.result?.calibration || {};
    calibrationStatus.textContent = `Kalibrasyon tamamlandı. Eşleşen satır: ${report.matched_row_count || 0}/${report.exact_row_count || 0}. Durum doğruluğu: ${formatNumber(report.state_exact_match_rate)}. Kullanılabilir rotasyon: ${calibration.usable_rotation_count || 0}.`;
  } catch (error) {
    calibrationStatus.textContent = error.message;
  }
}

function collectPredictionParams() {
  const storyCount = numberValue("predStoryCount");
  const storyHeight = numberValue("predStoryHeight");
  const concrete = document.querySelector("#predConcrete")?.value || "C30";
  const steel = document.querySelector("#predSteel")?.value || "B420C";
  return {
    direction: document.querySelector("#predDirection")?.value || "X",
    story_count: storyCount,
    x_bay_count: numberValue("predXBayCount"),
    y_bay_count: numberValue("predYBayCount"),
    avg_span_x: numberValue("predAvgSpanX"),
    avg_span_y: numberValue("predAvgSpanY"),
    max_span_x: numberValue("predMaxSpanX"),
    max_span_y: numberValue("predMaxSpanY"),
    story_height: storyHeight,
    total_height: storyCount * storyHeight,
    concrete_class: concrete,
    concrete_fck: Number(concrete.replace("C", "")) || 30,
    steel_class: steel,
    steel_fy: steel.includes("500") ? 500 : 420,
    soil_class: document.querySelector("#predSoil")?.value || "ZC",
    column_width: numberValue("predColumnWidth"),
    column_depth: numberValue("predColumnDepth"),
    beam_width: numberValue("predBeamWidth"),
    beam_depth: numberValue("predBeamDepth"),
    rho_col: numberValue("predRhoCol"),
    rho_beam_top: numberValue("predRhoBeamTop"),
    rho_beam_bottom: numberValue("predRhoBeamBottom"),
    slab_thickness: numberValue("predSlabThickness"),
    slab_rebar_ratio: numberValue("predSlabRebarRatio"),
    raft_thickness: numberValue("predRaftThickness"),
    raft_rebar_ratio: numberValue("predRaftRebarRatio"),
    subgrade_modulus: numberValue("predSubgrade"),
    target_drift: numberValue("predTargetDrift"),
    has_shear_walls: Boolean(document.querySelector("#predHasShearWalls")?.checked),
    foundation_type: document.querySelector("#predFoundation")?.value || "radye",
    wall_count: numberValue("predWallCount"),
    wall_thickness: numberValue("predWallThickness"),
    wall_length: numberValue("predWallLength"),
    wall_rebar_ratio: numberValue("predWallRebarRatio")
  };
}

function numberValue(id) {
  return Number(document.querySelector(`#${id}`)?.value || 0);
}

function renderMlStatus(status) {
  if (!mlStatus) return;
  if (!status.available && !status.metrics) {
    mlStatus.innerHTML = `<div class="muted-box">Eğitilmiş model yok. Veri seti: ${escapeHtml(status.dataset_count || 0)} yön-sonucu.</div>`;
    return;
  }
  const metrics = status.metrics || {};
  const regressionMetrics = status.regression_metrics || {};
  const importance = status.feature_importance_proxy || [];
  const algorithmNames = new Set(["knn", "random_forest", "xgboost", "lightgbm"]);
  const modelStates = status.models
    ? Object.entries(status.models)
        .filter(([name, info]) => algorithmNames.has(name) && Object.prototype.hasOwnProperty.call(info, "available"))
        .map(([name, info]) => `${name}: ${info.available ? "var" : "yok"}`)
        .join(" / ")
    : "";
  mlStatus.innerHTML = `
    <div class="metric-grid">
      <div><span>Veri</span><b>${escapeHtml(status.sample_count || status.dataset_count || 0)}</b></div>
      <div><span>Model</span><b>${escapeHtml(status.algorithm || "KNN")}</b></div>
      <div><span>SOM özellikleri</span><b>${status.use_som_features ? "Açık" : "Kapalı"}</b></div>
      <div><span>Seçili X parametresi</span><b>${escapeHtml((status.selected_x_columns || []).length || 0)}</b></div>
      <div><span>Dosya</span><b>${escapeHtml(status.model_path || "")}</b></div>
    </div>
    ${status.feature_engineering_note ? `<div class="result-note">${escapeHtml(status.feature_engineering_note)}</div>` : ""}
    ${modelStates ? `<div class="muted-box">${escapeHtml(modelStates)}</div>` : ""}
    <div class="ml-metrics">${Object.entries(metrics).map(([target, metric]) => metricCard(target, metric)).join("")}</div>
    ${Object.keys(regressionMetrics).length ? `<h3>Sayısal sonuç regresyonları</h3><div class="ml-metrics">${Object.entries(regressionMetrics).map(([target, metric]) => regressionMetricCard(target, metric)).join("")}</div>` : ""}
    <div class="info-block"><b>Etki sıralaması proxy</b>${importance.map((item) => `<span>${escapeHtml(item.feature)}: ${formatNumber(item.score)}</span>`).join("")}</div>
  `;
}

function metricCard(target, metric) {
  return `
    <div class="ml-card">
      <b>${escapeHtml(targetLabel(target))}</b>
      <span>Accuracy: ${formatNumber(metric.accuracy)}</span>
      <span>Test: ${escapeHtml(metric.test_count || 0)} kayıt</span>
      <span>${metric.best_k ? `k: ${escapeHtml(metric.best_k)}` : ""}${metric.tree_count ? `Ağaç: ${escapeHtml(metric.tree_count)}` : ""}${metric.n_estimators ? `Estimator: ${escapeHtml(metric.n_estimators)}` : ""}</span>
    </div>
  `;
}

function targetLabel(target) {
  return {
    first_hinge_type: "İlk mafsal tipi",
    critical_element_type: "Kritik eleman tipi",
    critical_state: "Kritik seviye",
    has_lscp: "LS-CP riski",
    first_column_available: "İlk kolon mafsalı",
    first_ls_available: "İlk LS seviyesi",
    first_cp_available: "İlk CP seviyesi"
  }[target] || target;
}

function renderPrediction(result) {
  if (result.directions) {
    let visual = "";
    try {
      visual = renderPredictionVisual(result);
    } catch (error) {
      visual = `<div class="error-box">Temsili görsel üretilemedi: ${escapeHtml(error.message)}</div>`;
    }
    predictionResult.innerHTML = `
      ${visual}
      <div class="ml-metrics">
        ${Object.entries(result.directions).map(([direction, directionResult]) => renderDirectionPrediction(direction, directionResult)).join("")}
      </div>
      <p class="section-note">Not: Eleman konumu, en yakın öğrenilmiş örneğin normalize plan/kat konumunun girilen geometriye projeksiyonudur; SAP2000 analizi yerine geçmez.</p>
    `;
    return;
  }
  const predictions = result.predictions || {};
  predictionResult.innerHTML = `
    <div class="ml-metrics">
      ${Object.entries(predictions).map(([target, info]) => `
        <div class="ml-card">
          <b>${escapeHtml(targetLabel(target))}</b>
          <strong>${escapeHtml(info.prediction)}</strong>
          <span>${probabilityText(info.probabilities || {})}</span>
        </div>
      `).join("")}
    </div>
    <p class="section-note">Not: Bu tahmin, mevcut sentetik veri setinden öğrenilen yaklaşık sınıflandırmadır; SAP2000 analizi yerine geçmez.</p>
  `;
}

function renderDirectionPrediction(direction, result) {
  const predictions = result.predictions || {};
  const regressionPredictions = result.regression_predictions || {};
  const metrics = result.model_summary?.metrics || {};
  const first = result.predicted_events?.first || {};
  const critical = result.predicted_events?.critical || {};
  const firstColumn = result.predicted_events?.first_column || {};
  const firstLs = result.predicted_events?.first_ls || {};
  const firstCp = result.predicted_events?.first_cp || {};
  return `
    <div class="ml-card">
      <b>${escapeHtml(direction)} yönü</b>
      <span>İlk mafsal: ${escapeHtml(predictions.first_hinge_type?.prediction || "-")}</span>
      <strong>${escapeHtml(first.element_name || "-")}</strong>
      <span>${predictionMeta(predictions.first_hinge_type, metrics.first_hinge_type)}</span>
      <span>Kritik eleman: ${escapeHtml(predictions.critical_element_type?.prediction || "-")}</span>
      <strong>${escapeHtml(critical.element_name || "-")}</strong>
      <span>${predictionMeta(predictions.critical_element_type, metrics.critical_element_type)}</span>
      <span>Kritik seviye: ${escapeHtml(predictions.critical_state?.prediction || "-")} / ${predictionMeta(predictions.critical_state, metrics.critical_state)}</span>
      <span>İlk kolon mafsalı: ${escapeHtml(predictions.first_column_available?.prediction || "-")}</span>
      <strong>${escapeHtml(firstColumn.available === false ? "Ulaşmadı" : (firstColumn.element_name || "-"))}</strong>
      <span>${predictionMeta(predictions.first_column_available, metrics.first_column_available)}</span>
      <span>İlk LS seviyesi: ${escapeHtml(predictions.first_ls_available?.prediction || "-")}</span>
      <strong>${escapeHtml(firstLs.available === false ? "Ulaşmadı" : (firstLs.element_name || "-"))}</strong>
      <span>${predictionMeta(predictions.first_ls_available, metrics.first_ls_available)}</span>
      <span>İlk CP seviyesi: ${escapeHtml(predictions.first_cp_available?.prediction || "-")}</span>
      <strong>${escapeHtml(firstCp.available === false ? "Ulaşmadı" : (firstCp.element_name || "-"))}</strong>
      <span>${predictionMeta(predictions.first_cp_available, metrics.first_cp_available)}</span>
      <span>LS-CP: ${escapeHtml(predictions.has_lscp?.prediction || "-")} / ${predictionMeta(predictions.has_lscp, metrics.has_lscp)}</span>
      ${Object.entries(regressionPredictions).map(([target, value]) => `<span>${escapeHtml(regressionTargetLabel(target))}: <strong>${formatNumber(value)}</strong></span>`).join("")}
    </div>
  `;
}

function predictionMeta(prediction, metric) {
  const probability = probabilityText(prediction?.probabilities || {});
  const accuracy = metric && metric.accuracy !== undefined ? `accuracy ${formatNumber(metric.accuracy)}` : "accuracy -";
  return `${probability || "olasılık -"} / ${accuracy}`;
}
function renderPredictionVisual(result) {
  const input = result.input || {};
  const model = {
    name: "ML tahmin",
    story_count: input.story_count,
    x_bay_count: input.x_bay_count,
    y_bay_count: input.y_bay_count,
    spans_x: Array.from({ length: Math.max(1, Math.trunc(input.x_bay_count || 1)) }, () => input.avg_span_x || 5),
    spans_y: Array.from({ length: Math.max(1, Math.trunc(input.y_bay_count || 1)) }, () => input.avg_span_y || 5),
    story_height: input.story_height || 3.2,
    hinge_summary: {}
  };
  Object.entries(result.directions || {}).forEach(([direction, directionResult]) => {
    const first = directionResult.predicted_events?.first || {};
    const critical = directionResult.predicted_events?.critical || {};
    model.hinge_summary[direction] = {
      first_plastic_hinge: predictionEventToHinge(first, directionResult.predictions?.first_hinge_type?.prediction),
      critical_events: [predictionEventToHinge(critical, directionResult.predictions?.critical_element_type?.prediction, directionResult.predictions?.critical_state?.prediction)]
    };
  });
  return `<div class="prediction-visual">${renderHighlightedPreview(model)}</div>`;
}

function predictionEventToHinge(event, elementType, state = "") {
  return {
    element_name: event.element_name || "",
    element_type: event.element_type || elementType || "",
    hinge_state_level: event.source_state || state || "",
    step_number: event.source_step || "",
    hinge_location: "tahmin"
  };
}

function probabilityText(probabilities) {
  return Object.entries(probabilities).map(([key, value]) => `${key}: ${formatNumber(value)}`).join(" / ");
}

function renderHighlightedPreview(model) {
  const storyCount = Math.max(1, Number(model.story_count) || 1);
  const spansX = previewSpanList(model.spans_x, Number(model.x_bay_count) || 1);
  const spansY = previewSpanList(model.spans_y, Number(model.y_bay_count) || 1);
  const storyHeight = Number(model.story_height) || 3.2;
  const xCoords = previewCumulative(spansX);
  const yCoords = previewCumulative(spansY);
  const highlights = previewHighlightsByDirection(model, xCoords, yCoords, storyCount, storyHeight);
  const width = 440;
  const height = 245;
  const plan = { x: 12, y: 38, w: 184, h: 154 };
  const elev = { x: 250, y: 38, w: 158, h: 154 };
  const totalX = Math.max(xCoords[xCoords.length - 1], 1);
  const totalY = Math.max(yCoords[yCoords.length - 1], 1);
  const totalZ = Math.max(storyCount * storyHeight, 1);
  const px = (x) => plan.x + (x / totalX) * plan.w;
  const py = (y) => plan.y + (y / totalY) * plan.h;
  const ex = (x) => elev.x + (x / totalX) * elev.w;
  const ey = (z) => elev.y + elev.h - (z / totalZ) * elev.h;
  const grid = [
    ...yCoords.map((y) => `<line class="preview-grid-line" x1="${px(0)}" y1="${py(y)}" x2="${px(totalX)}" y2="${py(y)}"></line>`),
    ...xCoords.map((x) => `<line class="preview-grid-line" x1="${px(x)}" y1="${py(0)}" x2="${px(x)}" y2="${py(totalY)}"></line>`),
    ...xCoords.flatMap((x) => yCoords.map((y) => `<circle class="preview-node" cx="${px(x)}" cy="${py(y)}" r="2.3"></circle>`)),
    ...Array.from({ length: storyCount + 1 }, (_, story) => `<line class="preview-grid-line" x1="${ex(0)}" y1="${ey(story * storyHeight)}" x2="${ex(totalX)}" y2="${ey(story * storyHeight)}"></line>`),
    ...xCoords.map((x) => `<line class="preview-grid-line" x1="${ex(x)}" y1="${ey(0)}" x2="${ex(x)}" y2="${ey(totalZ)}"></line>`)
  ].join("");
  const highlightSvg = highlights.map((item) => previewHighlightSvg(item, px, py, ex, ey, totalX, totalY)).join("");
  const legend = highlights.length
    ? `<g class="preview-legend"><line class="preview-highlight-line highlight-x-critical" x1="16" y1="221" x2="28" y2="221"></line><text x="34" y="224">X kritik</text><line class="preview-highlight-line highlight-x-first" x1="92" y1="221" x2="104" y2="221"></line><text x="110" y="224">X ilk</text><line class="preview-highlight-line highlight-y-critical" x1="158" y1="221" x2="170" y2="221"></line><text x="176" y="224">Y kritik</text><line class="preview-highlight-line highlight-y-first" x1="234" y1="221" x2="246" y2="221"></line><text x="252" y="224">Y ilk</text></g>`
    : `<text class="preview-meta-svg" x="12" y="224">Plastik mafsal olusmadi</text>`;
  return `
    <svg class="highlight-preview" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Model temsil goruntusu">
      ${previewSvgEmbeddedStyle()}
      <rect x="0.5" y="0.5" width="${width - 1}" height="${height - 1}" rx="6"></rect>
      <text class="preview-title-svg" x="10" y="18">${escapeHtml(storyCount)}Story X${escapeHtml(spansX.length)} Y${escapeHtml(spansY.length)}</text>
      <text class="preview-label" x="${plan.x}" y="31">Plan</text>
      <text class="preview-label" x="${elev.x}" y="31">Elevasyon</text>
      ${grid}
      ${highlightSvg}
      <line class="preview-height-arrow" x1="${elev.x + elev.w + 17}" y1="${elev.y + elev.h}" x2="${elev.x + elev.w + 17}" y2="${elev.y}"></line>
      <path class="preview-height-arrow" d="M${elev.x + elev.w + 17} ${elev.y} l-7 12 M${elev.x + elev.w + 17} ${elev.y} l7 12"></path>
      ${legend}
    </svg>
  `;
}

function previewSvgEmbeddedStyle() {
  return `
    <style>
      .highlight-preview{background:#fbfcfd}.highlight-preview rect{fill:#fbfcfd;stroke:#d3dce7}
      .preview-title-svg{fill:#111827;font-size:13px;font-weight:800;font-family:Arial,sans-serif}
      .preview-label,.preview-meta-svg,.preview-legend text{fill:#111827;font-size:9px;font-weight:700;font-family:Arial,sans-serif}
      .preview-grid-line{stroke:#0f7890;stroke-width:1.25;vector-effect:non-scaling-stroke}
      .preview-node{fill:#2f855a;stroke:#2f855a}.preview-highlight-line{stroke-width:4;stroke-linecap:round;vector-effect:non-scaling-stroke}
      .preview-highlight-node{stroke:#fff;stroke-width:2;vector-effect:non-scaling-stroke}
      .preview-highlight-node.highlight-x-first,.preview-highlight-node.highlight-y-first{stroke:#111827;stroke-width:1.5}
      .highlight-x-critical{fill:#dc2626;stroke:#dc2626}.highlight-x-first{fill:#f59e0b;stroke:#f59e0b;stroke-dasharray:7 4}
      .highlight-y-critical{fill:#2563eb;stroke:#2563eb}.highlight-y-first{fill:#38bdf8;stroke:#38bdf8;stroke-dasharray:7 4}
      .preview-height-arrow{fill:none;stroke:#b35c2e;stroke-width:2;stroke-linecap:round}
    </style>
  `;
}

function previewHighlightsByDirection(model, xCoords, yCoords, storyCount, storyHeight) {
  const summaries = Object.entries(model.hinge_summary || {});
  const items = [];
  summaries.forEach(([direction, summary]) => {
    const dir = String(direction || "").toUpperCase() === "Y" ? "Y" : "X";
    const critical = Array.isArray(summary.critical_events)
    ? summary.critical_events.filter((event) => event && hingeStateRank(event.hinge_state_level) >= hingeStateRank("B-IO")).sort(compareCriticalEvents)[0]
      : null;
    const criticalSegment = previewElementSegment(critical?.element_name, xCoords, yCoords, storyCount, storyHeight);
    if (criticalSegment) items.push({ ...criticalSegment, role: "critical", direction: dir });
    const firstSegment = previewElementSegment(summary.first_plastic_hinge?.element_name, xCoords, yCoords, storyCount, storyHeight);
    if (firstSegment) {
      items.push({ ...firstSegment, role: "first", direction: dir });
    }
  });
  return items;
}

function compareCriticalEvents(a, b) {
  const stateDiff = hingeStateRank(b?.hinge_state_level) - hingeStateRank(a?.hinge_state_level);
  if (stateDiff) return stateDiff;
  const rotDiff = Number(b?.plastic_rotation_rad || 0) - Number(a?.plastic_rotation_rad || 0);
  if (rotDiff) return rotDiff;
  return Number(b?.demand_capacity_ratio || 0) - Number(a?.demand_capacity_ratio || 0);
}

function compareFirstEvents(a, b) {
  return Number(a?.step_number ?? 1e9) - Number(b?.step_number ?? 1e9);
}

function hingeStateRank(level) {
  const ranks = {
    "A-B": 0,
    "B-IO": 1,
    "IO-LS": 2,
    "LS-CP": 3,
    "CP-C": 4,
    "C-D": 5,
    "D-E": 6,
    "beyond E": 7
  };
  return ranks[String(level || "")] ?? -1;
}

function previewHighlightSvg(item, px, py, ex, ey, totalX, totalY) {
  const cls = `highlight-${String(item.direction || "X").toLowerCase()}-${item.role === "critical" ? "critical" : "first"}`;
  const elevOffset = item.role === "first" ? (item.direction === "Y" ? 3 : -3) : 0;
  const columnPlanOffsets = {
    X_critical: [8, 0],
    X_first: [-8, 0],
    Y_critical: [0, 8],
    Y_first: [0, -8]
  };
  const columnElevOffsets = {
    X_critical: 4,
    X_first: -4,
    Y_critical: 9,
    Y_first: -9
  };
  const offsetKey = `${item.direction || "X"}_${item.role === "critical" ? "critical" : "first"}`;
  const [planDx, planDy] = columnPlanOffsets[offsetKey] || [0, 0];
  const elevColumnOffset = columnElevOffsets[offsetKey] || 0;
  const planPart = item.type === "column"
    ? `<circle class="${cls} preview-highlight-node" cx="${px(item.a[0]) + planDx}" cy="${py(item.a[1]) + planDy}" r="${item.role === "critical" ? 6 : 4.5}"></circle>`
    : `<line class="${cls} preview-highlight-line" x1="${px(item.a[0])}" y1="${py(item.a[1])}" x2="${px(item.b[0])}" y2="${py(item.b[1])}"></line>`;
  if (item.type === "column") {
    return `${planPart}<line class="${cls} preview-highlight-line" x1="${ex(item.a[0]) + elevColumnOffset}" y1="${ey(item.a[2])}" x2="${ex(item.b[0]) + elevColumnOffset}" y2="${ey(item.b[2])}"></line>`;
  }
  const usesY = Math.abs(item.a[0] - item.b[0]) < 0.001;
  const a = usesY ? (item.a[1] / Math.max(totalY, 1)) * totalX : item.a[0];
  const b = usesY ? (item.b[1] / Math.max(totalY, 1)) * totalX : item.b[0];
  return `${planPart}<line class="${cls} preview-highlight-line" x1="${ex(a)}" y1="${ey(item.a[2]) + elevOffset}" x2="${ex(b)}" y2="${ey(item.b[2]) + elevOffset}"></line>`;
}

function previewElementSegment(name, xCoords, yCoords, storyCount, storyHeight) {
  const text = String(name || "").trim();
  let match = /^C_(\d+)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$/i.exec(text);
  if (match) {
    const story = Number(match[1]);
    const x = previewClosestCoord(xCoords, Number(match[2]));
    const y = previewClosestCoord(yCoords, Number(match[3]));
    if (story >= 1 && story <= storyCount && x !== null && y !== null) {
      return { name: text, type: "column", a: [x, y, (story - 1) * storyHeight], b: [x, y, story * storyHeight] };
    }
  }
  match = /^BX_(\d+)_(\d+)_(-?\d+(?:\.\d+)?)$/i.exec(text);
  if (match) {
    const story = Number(match[1]);
    const index = Number(match[2]);
    const y = previewClosestCoord(yCoords, Number(match[3]));
    if (story >= 1 && story <= storyCount && index >= 0 && index < xCoords.length - 1 && y !== null) {
      const z = story * storyHeight;
      return { name: text, type: "beam", a: [xCoords[index], y, z], b: [xCoords[index + 1], y, z] };
    }
  }
  match = /^BY_(\d+)_(\d+)_(-?\d+(?:\.\d+)?)$/i.exec(text);
  if (match) {
    const story = Number(match[1]);
    const index = Number(match[2]);
    const x = previewClosestCoord(xCoords, Number(match[3]));
    if (story >= 1 && story <= storyCount && index >= 0 && index < yCoords.length - 1 && x !== null) {
      const z = story * storyHeight;
      return { name: text, type: "beam", a: [x, yCoords[index], z], b: [x, yCoords[index + 1], z] };
    }
  }
  return null;
}

function previewSpanList(values, fallbackCount) {
  const clean = Array.isArray(values) ? values.map(Number).filter((value) => Number.isFinite(value) && value > 0) : [];
  if (clean.length) return clean;
  return Array.from({ length: Math.max(1, fallbackCount) }, () => 5);
}

function previewCumulative(spans) {
  const coords = [0];
  spans.forEach((span) => coords.push(Math.round((coords[coords.length - 1] + span) * 100) / 100));
  return coords;
}

function previewClosestCoord(coords, value) {
  if (!Number.isFinite(value)) return null;
  const best = coords.reduce((current, coord) => (Math.abs(coord - value) < Math.abs(current - value) ? coord : current), coords[0]);
  return Math.abs(best - value) <= 0.15 ? best : null;
}

function elementStory(name) {
  const text = String(name || "").trim();
  let match = /^C_(\d+)_/i.exec(text);
  if (match) return Number(match[1]);
  match = /^BX_(\d+)_/i.exec(text);
  if (match) return Number(match[1]);
  match = /^BY_(\d+)_/i.exec(text);
  if (match) return Number(match[1]);
  return null;
}

function elementPlanZone(name, spansX, spansY) {
  const xCoords = previewCumulative(previewSpanList(spansX, 1));
  const yCoords = previewCumulative(previewSpanList(spansY, 1));
  const text = String(name || "").trim();
  let match = /^C_(\d+)_(-?\d+(?:\.\d+)?)_(-?\d+(?:\.\d+)?)$/i.exec(text);
  if (match) {
    return classifyPlanZone(Number(match[2]), Number(match[3]), xCoords, yCoords);
  }
  match = /^BX_(\d+)_(\d+)_(-?\d+(?:\.\d+)?)$/i.exec(text);
  if (match) {
    const idx = Number(match[2]);
    const midX = idx >= 0 && idx < xCoords.length - 1 ? (xCoords[idx] + xCoords[idx + 1]) / 2 : null;
    return classifyPlanZone(midX, Number(match[3]), xCoords, yCoords);
  }
  match = /^BY_(\d+)_(\d+)_(-?\d+(?:\.\d+)?)$/i.exec(text);
  if (match) {
    const idx = Number(match[2]);
    const midY = idx >= 0 && idx < yCoords.length - 1 ? (yCoords[idx] + yCoords[idx + 1]) / 2 : null;
    return classifyPlanZone(Number(match[3]), midY, xCoords, yCoords);
  }
  return "-";
}

function classifyPlanZone(x, y, xCoords, yCoords) {
  const xBoundary = isCornerCoord(x, xCoords);
  const yBoundary = isCornerCoord(y, yCoords);
  if (xBoundary || yBoundary) return "edge";
  return "middle";
}

function isCornerCoord(value, coords) {
  if (!Number.isFinite(Number(value)) || !Array.isArray(coords) || coords.length < 2) return false;
  const numeric = Number(value);
  return Math.abs(numeric - coords[0]) <= 0.15 || Math.abs(numeric - coords[coords.length - 1]) <= 0.15;
}

function renderPushoverResults(models, emptyMessage = "") {
  if (!pushoverResults) return;
  pushoverResults.innerHTML = "";
  if (!models.length) {
    pushoverResults.innerHTML = `<div class="preview-item"><div class="preview-meta">${escapeHtml(emptyMessage || "Henüz pushover sonucu yok.")}</div></div>`;
    return;
  }
  models.slice(0, 12).forEach((model) => {
    const card = document.createElement("div");
    card.className = "result-card";
    const curves = model.curves || {};
    const curveBlocks = Object.entries(curves).map(([direction, curve]) => renderCurveBlock(direction, curve, model.story_drifts, model.fema440?.[direction])).join("");
    const statusText = model.status === "success" ? "Başarılı" : `Durum: ${model.status || "-"}`;
    const hingeBlocks = renderHingeInfo(model);
    const hingeText = `${model.hinge_assigned_count ?? 0}/${model.hinge_expected_count ?? 0} mafsal`;
    const hasProxyHinges = !model.exact_export_available && Object.keys(model.hinge_summary || {}).length > 0;
    const hingeSourceText = model.exact_export_available
      ? "Mafsal özeti: gerçek SAP Frame Hinge States exportu"
      : hasProxyHinges
        ? "Mafsal özeti: proxy hinge hesabı (moment/kapasite ve kalibre edilebilir rotasyon backbone'u)"
        : "Mafsal özeti henüz okunmadı";
    card.innerHTML = `
      <div class="result-head">
        <div>
          <div class="preview-title">${escapeHtml(model.name)}</div>
          <div class="preview-meta">${escapeHtml(statusText)} / ${escapeHtml(hingeText)} / hedef ${formatNumber(model.target_displacement_m)} m</div>
          <div class="preview-meta">${escapeHtml(hingeSourceText)}</div>
        </div>
        <button type="button" class="open-model-btn" data-model-name="${escapeHtml(model.model_file || model.name)}">SAP2000'de Aç</button>
      </div>
      ${curveBlocks || `<div class="preview-meta">${escapeHtml(model.message || "Kapasite eğrisi henüz okunmadı.")}</div>`}
      ${hingeBlocks}
    `;
    const button = card.querySelector(".open-model-btn");
    button.addEventListener("click", () => openSapModel(button.dataset.modelName));
    pushoverResults.appendChild(card);
  });
}

function renderCurveBlock(direction, curve, storyDrifts = {}, fema440 = null) {
  const points = normalizeCapacityPoints(Array.isArray(curve.points) ? curve.points : []);
  const message = curve.message ? `<div class="preview-meta">${escapeHtml(friendlyResultMessage(curve.message))}</div>` : "";
  const driftBlock = renderStoryDriftBlock(direction, storyDrifts);
  const femaBlock = renderFema440Block(fema440);
  if (!curve.available || !points.length) {
    return `
      <div class="curve-block">
        <div class="curve-title">${escapeHtml(direction)} yönü</div>
        <div class="preview-meta">Sonuç noktası yok. ${message}</div>
        ${driftBlock}
        ${femaBlock}
      </div>
    `;
  }
  return `
    <div class="curve-block">
      <div class="curve-title">${escapeHtml(direction)} yönü / tepe ${formatNumber(curve.peak_base_shear_kn)} kN / son ${formatNumber(curve.final_control_displacement_m)} m</div>
      ${renderStepInfoTable(points)}
      ${driftBlock}
      ${femaBlock}
    </div>
  `;
}

function renderFema440Block(fema) {
  if (!fema) return "";
  if (!fema.available) {
    return `
      <div class="info-block">
        <div class="info-title">FEMA 440 ön değerlendirme</div>
        <div class="preview-meta">${escapeHtml(fema.message || "Hesaplanamadı.")}</div>
      </div>
    `;
  }
  const ideal = fema.idealization || {};
  const el = fema.equivalent_linearization || {};
  const dm = fema.displacement_modification || {};
  return `
    <div class="info-block">
      <div class="info-title">FEMA 440 Equivalent Linearization</div>
      <table class="info-table">
        <tbody>
          <tr><th>${femaParamLabel("Yield deplasmanı", "Kapasite eğrisinde elastik davranıştan plastik davranışa geçiş için yaklaşık tepe deplasmanı.")}</th><td>${formatNumber(ideal.yield_displacement_m)} m</td><th>${femaParamLabel("Yield taban kesmesi", "Yield noktasında yapının taşıdığı yaklaşık toplam yatay taban kesmesi.")}</th><td>${formatNumber(ideal.yield_base_shear_kn)} kN</td></tr>
          <tr><th>${femaParamLabel("Süneklik μ", "Nihai/son deplasmanın yield deplasmanına oranı; plastik deformasyon kapasitesini özetler.")}</th><td>${formatNumber(el.ductility_mu)}</td><th>${femaParamLabel("T<sub>eff</sub>/T<sub>0</sub>", "Hasar ve rijitlik azalımı sonrası etkin periyodun başlangıç periyoduna göre büyümesini gösterir.")}</th><td>${formatNumber(el.effective_period_ratio_teff_t0)}</td></tr>
          <tr><th>${femaParamLabel("T<sub>0</sub> proxy", "İlk elastik periyot için kullanılan yaklaşık başlangıç değeri; burada kat sayısından türetilen ön kabul.")}</th><td>${formatNumber(el.initial_period_proxy_s)} s</td><th>${femaParamLabel("T<sub>eff</sub>", "Eşdeğer lineerleştirmede kullanılan hasarlı/etkin periyot; rijitlik azaldıkça genellikle artar.")}</th><td>${formatNumber(el.effective_period_teff_s)} s</td></tr>
          <tr><th>${femaParamLabel("β<sub>eff</sub>", "Histeretik enerji tüketimini temsil eden etkin sönüm oranı; daha büyük değer daha fazla enerji tüketimi demektir.")}</th><td>${formatNumber(el.effective_damping_beta_percent)}%</td><th>${femaParamLabel("B(β)", "Etkin sönüme bağlı spektral azaltma/düzeltme katsayısı; talep spektrumunu uyarlamak için kullanılır.")}</th><td>${formatNumber(el.damping_reduction_factor_b_beta)}</td></tr>
        </tbody>
      </table>
      <div class="result-note">${escapeHtml(el.message || fema.method_note || "")}</div>
    </div>
    <div class="info-block">
      <div class="info-title">FEMA 440 Displacement Modification</div>
      <table class="info-table">
        <tbody>
          <tr><th>${femaParamLabel("Zemin/site proxy", "TBDY zemin sınıfından yaklaşık FEMA katsayı sınıfına geçiştir; gerçek zemin spektrumu yerine ön kabul olarak kullanılır.")}</th><td>${escapeHtml(dm.coefficient_site_class || "-")} / a=${formatNumber(dm.site_coefficient_a)}</td><th>${femaParamLabel("R proxy", "Elastik talep/kapasite oranını temsil eden yaklaşık katsayı; burada kapasite eğrisinden türetilir.")}</th><td>${formatNumber(dm.r_capacity_proxy)}</td></tr>
          <tr><th>${femaParamLabel("C0", "SDOF deplasmanını çatı/tepe deplasmanına dönüştüren katsayı; bu aşamada 1.0 ön kabulü kullanılır.")}</th><td>${formatNumber(dm.c0)}</td><th>${femaParamLabel("C1", "İnelastik deplasmanın elastik deplasmana göre artışını düzeltir; kısa periyot ve yüksek R durumunda büyür.")}</th><td>${formatNumber(dm.c1)}</td></tr>
          <tr><th>${femaParamLabel("C2", "Çevrimsel dayanım/rijitlik bozulması ve histeretik şekil etkisini temsil eder; bu ön hesapta 1.0 alınır.")}</th><td>${formatNumber(dm.c2)}</td><th>${femaParamLabel("C3", "P-Delta etkileri nedeniyle deplasman büyümesini temsil eder; bu ön hesapta 1.0 alınır.")}</th><td>${formatNumber(dm.c3)}</td></tr>
          <tr><th>${femaParamLabel("Elastik depl. proxy", "Spektral talep yerine kapasite eğrisinden türetilmiş yaklaşık elastik deplasman göstergesi.")}</th><td>${formatNumber(dm.elastic_displacement_proxy_m)} m</td><th>${femaParamLabel("Hedef depl. proxy", "C0-C3 katsayıları uygulanmış yaklaşık hedef deplasman; gerçek FEMA hedef deplasmanı için talep spektrumu gerekir.")}</th><td>${formatNumber(dm.target_displacement_proxy_m)} m</td></tr>
        </tbody>
      </table>
      <div class="result-note">${escapeHtml(dm.message || "")}</div>
    </div>
  `;
}

function femaParamLabel(label, note) {
  return `<span>${label}</span><small class="fema-param-note">${escapeHtml(note)}</small>`;
}

function renderStoryDriftBlock(direction, storyDrifts = {}) {
  const data = storyDrifts?.by_direction?.[direction] || storyDrifts?.by_direction?.[String(direction).toUpperCase()] || null;
  const rows = Array.isArray(data?.max_by_story) ? data.max_by_story : [];
  if (!rows.length) return "";
  const max = data.max_drift || rows.reduce((best, row) => Number(row.drift_ratio || 0) > Number(best?.drift_ratio || 0) ? row : best, null);
  const body = rows.map((row) => {
    const ratio = Number(row.drift_ratio || 0);
    const cls = ratio >= 0.02 ? "drift-critical" : ratio >= 0.01 ? "drift-warning" : "";
    return `
      <tr class="${cls}">
        <td>${escapeHtml(row.story ?? "-")}</td>
        <td>${escapeHtml(row.step_number ?? "-")}</td>
        <td>${formatNumber(row.lower_displacement_m)}</td>
        <td>${formatNumber(row.upper_displacement_m)}</td>
        <td>${formatNumber(row.relative_displacement_m)}</td>
        <td>${formatPercent(ratio)}</td>
      </tr>
    `;
  }).join("");
  return `
    <div class="info-block">
      <div class="info-title">Göreli kat ötelemesi ${max ? `/ max kat ${escapeHtml(max.story ?? "-")} · ${formatPercent(max.drift_ratio)}` : ""}</div>
      <table class="info-table drift-table">
        <thead><tr><th>Kat</th><th>Step</th><th>Alt depl. m</th><th>Üst depl. m</th><th>Göreli depl. m</th><th>Drift</th></tr></thead>
        <tbody>${body}</tbody>
      </table>
    </div>
  `;
}

function renderStepInfoTable(points) {
  const visiblePoints = points.filter((point) => String(point.load_step || "").toLowerCase() !== "origin");
  const rows = visiblePoints.slice(0, 8).map((point) => `
    <tr>
      <td>${escapeHtml(point.step_number ?? point.step ?? "-")}</td>
      <td>${escapeHtml(point.load_step || "-")}</td>
      <td>${formatNumber(point.control_displacement_m)}</td>
      <td>${formatNumber(point.base_shear_kn)}</td>
    </tr>
  `).join("");
  return `
    <div class="info-block">
      <div class="info-title">Adim bilgileri</div>
      <table class="info-table">
        <thead><tr><th>Step number</th><th>Load step</th><th>Roof displacement m</th><th>Base shear kN</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
  `;
}

function renderHingeInfo(model) {
  const summaries = Object.entries(model.hinge_summary || {});
  if (!summaries.length) return "";
  const samples = model.hinge_event_samples || {};
  return summaries.map(([direction, summary]) => `
    <div class="info-block">
      <div class="info-title">${escapeHtml(direction)} plastik mafsal bilgileri / ${escapeHtml(model.hinge_event_counts?.[direction] ?? summary.event_count ?? 0)} kayıt</div>
      ${renderDamageSummary(summary)}
      ${renderMilestonesOrEnvelopeNote(summary, samples[direction] || [])}
      ${renderStateCounts(summary.state_counts || {})}
      ${renderHingeSampleTable(samples[direction] || [])}
    </div>
  `).join("");
}

function renderDamageSummary(summary) {
  const counts = summary.state_counts || {};
  const critical = Array.isArray(summary.critical_events)
    ? summary.critical_events.find((event) => event && hingeStateRank(event.hinge_state_level) >= hingeStateRank("B-IO"))
    : null;
  const lsCount = Number(counts["LS-CP"] || 0);
  const cpCount = Number(counts["CP-C"] || 0);
  const firstLs = summary.first_ls_level;
  const firstCp = summary.first_cp_level;
  return `
    <div class="damage-summary-title">Hasar Özeti</div>
    <div class="damage-summary">
      <div><span>LS-CP</span><b>${formatNumber(lsCount)} olay</b></div>
      <div><span>CP-C</span><b>${formatNumber(cpCount)} olay</b></div>
      <div><span>İlk LS</span><b>${firstLs ? `step ${formatNumber(firstLs.step_number)}` : "Ulaşmadı"}</b></div>
      <div><span>İlk CP</span><b>${firstCp ? `step ${formatNumber(firstCp.step_number)}` : "Ulaşmadı"}</b></div>
      <div><span>Kritik eleman</span><b>${critical ? `${escapeHtml(critical.element_type || "-")} / ${escapeHtml(critical.hinge_state_level || "-")}` : "-"}</b><small>${critical ? escapeHtml(critical.element_name || "-") : ""}</small></div>
    </div>
  `;
}

function renderMilestonesOrEnvelopeNote(summary, sampleEvents) {
  const envelopeOnly = summary.is_envelope_only
    || isEnvelopeEvent(summary.first_plastic_hinge)
    || (Array.isArray(sampleEvents) && sampleEvents.length > 0 && sampleEvents.every(isEnvelopeEvent));
  if (envelopeOnly) {
    return `
      <div class="milestone-grid">
        ${statusMilestone("İlk plastik mafsal", "Okunamadı", "SAP2000 sadece Max/Min envelope sonucu verdi.")}
        ${statusMilestone("İlk kolon mafsalı", "Okunamadı", "Gerçek adım geçmişi yok.")}
        ${statusMilestone("İlk LS seviyesi", "Okunamadı", "Gerçek adım geçmişi yok.")}
        ${statusMilestone("İlk CP seviyesi", "Okunamadı", "Gerçek adım geçmişi yok.")}
      </div>
      <div class="result-note">Alttaki tablo kritik eleman proxy bilgisidir; ilk oluşum adımı olarak yorumlanmamalıdır.</div>
    `;
  }
  return `
    <div class="milestone-grid">
      ${milestoneItem("İlk plastik mafsal", summary.first_plastic_hinge, "Ulaşmadı")}
      ${milestoneItem("İlk kolon mafsalı", summary.first_column_hinge, "Ulaşmadı")}
      ${milestoneItem("İlk LS seviyesi", summary.first_ls_level, "Ulaşmadı")}
      ${milestoneItem("İlk CP seviyesi", summary.first_cp_level, "Ulaşmadı")}
    </div>
  `;
}

function isEnvelopeEvent(event) {
  if (!event) return false;
  const loadStep = String(event.load_step || "").toLowerCase();
  return Number(event.step_number) === 0 && ["max", "min", "envelope"].includes(loadStep);
}

function milestoneItem(label, event, emptyStatus = "Ulaşmadı") {
  if (!event) {
    return statusMilestone(label, emptyStatus, "Bu eşik analiz adımları içinde görülmedi.");
  }
  const loadStep = event.load_step || "-";
  const stepText = Number(event.step_number) === 0 && ["Max", "Min", "Envelope"].includes(String(loadStep))
    ? `load ${escapeHtml(loadStep)}`
    : `step ${formatNumber(event.step_number)} / load ${escapeHtml(loadStep)}`;
  return `
    <div class="milestone">
      <span>${escapeHtml(label)}</span>
      <b>${escapeHtml(event.element_name || "-")} / ${escapeHtml(event.hinge_state_level || "-")}</b>
      <small>${stepText} / ${escapeHtml(event.element_type || "-")} / ${escapeHtml(event.hinge_location || "-")}</small>
    </div>
  `;
}

function statusMilestone(label, status, detail) {
  return `
    <div class="milestone status-${status.toLowerCase()}">
      <span>${escapeHtml(label)}</span>
      <b>${escapeHtml(status)}</b>
      <small>${escapeHtml(detail)}</small>
    </div>
  `;
}

function renderStateCounts(counts) {
  const entries = Object.entries(counts);
  if (!entries.length) return "";
  return `<div class="state-chips">${entries.map(([key, value]) => `<span>${escapeHtml(key)}: ${escapeHtml(value)}</span>`).join("")}</div>`;
}

function renderHingeSampleTable(events) {
  const plasticEvents = events.filter((event) => event && hingeStateRank(event.hinge_state_level) >= hingeStateRank("B-IO"));
  if (!plasticEvents.length) return '<div class="preview-meta">Plastik mafsal oluşmadı.</div>';
  const rows = plasticEvents.slice(0, 8).map((event) => `
    <tr>
      <td>${escapeHtml(event.step_number ?? "-")}</td>
      <td>${escapeHtml(event.load_step || "-")}</td>
      <td>${escapeHtml(event.element_name || "-")}</td>
      <td>${escapeHtml(event.element_type || "-")}</td>
      <td>${escapeHtml(event.hinge_location || "-")}</td>
      <td>${escapeHtml(event.hinge_state_level || "-")}</td>
      <td>${formatNumber(event.plastic_rotation_rad)}</td>
      <td>${formatNumber(event.moment_kn_m)}</td>
      <td>${formatNumber(event.axial_force_kn)}</td>
      <td>${formatNumber(event.shear_v2_kn)} / ${formatNumber(event.shear_v3_kn)}</td>
    </tr>
  `).join("");
  return `
    <div class="info-title">Kritik mafsal olayları</div>
    <table class="info-table hinge-table">
      <thead><tr><th>Step</th><th>Load</th><th>Eleman</th><th>Tip</th><th>Konum</th><th>Seviye</th><th>Rot.</th><th>M</th><th>P</th><th>V2/V3</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  `;
}

function renderSom() {
  if (!somCanvas || !somSummary) return;
  const som = latestSom;
  if (!som || !som.available) {
    somSummary.innerHTML = "";
    somCanvas.innerHTML = `<div class="muted-box">${escapeHtml(som?.message || "Henüz SOM eğitilmedi.")}</div>`;
    return;
  }
  const metricKey = somResultMetric?.value || som.result_metric || "max_story_drift_ratio";
  const metrics = som.result_metrics || defaultSomMetrics;
  const metric = metrics[metricKey] || metrics[som.result_metric] || defaultSomMetrics.max_story_drift_ratio;
  const mode = somMapMode?.value || "result";
  const cells = Array.isArray(som.cells) ? som.cells : [];
  const uValues = new Map((som.u_matrix || []).map((item) => [`${item.x},${item.y}`, Number(item.value || 0)]));
  const coloredValues = mode === "umatrix"
    ? [...uValues.values()]
    : mode === "hits"
      ? cells.map((cell) => Number(cell.hit_count || 0))
      : metric.type === "category"
        ? cells.map((cell) => Number(somMetricSummary(cell, metricKey).dominant_count || 0)).filter(Number.isFinite)
        : cells.map((cell) => Number(somMetricSummary(cell, metricKey).avg)).filter(Number.isFinite);
  const min = Math.min(...coloredValues, 0);
  const max = Math.max(...coloredValues, 1);

  somSummary.innerHTML = `
    <div><span>Kayıt</span><b>${escapeHtml(som.sample_count || 0)}</b></div>
    <div><span>Grid</span><b>${escapeHtml(som.width)} x ${escapeHtml(som.height)}</b></div>
    <div><span>Dolu hücre</span><b>${cells.filter((cell) => Number(cell.hit_count || 0) > 0).length} / ${cells.length}</b></div>
    <div><span>Quantization error</span><b>${formatNumber(som.quantization_error)}</b></div>
    <div><span>Topographic error</span><b>${formatNumber(som.topographic_error)}</b></div>
    <div><span>Purity</span><b>${som.purity === null || som.purity === undefined ? "-" : formatPercent(som.purity)}</b></div>
    <div><span>X parametre</span><b>${escapeHtml((som.selected_x_columns || []).length || 0)}</b></div>
    <div><span>Renk</span><b>${escapeHtml(metric.label || metricKey)}</b></div>
    ${som.optimization?.available ? `<div><span>Optimum grid</span><b>${escapeHtml(som.optimization.selected_width)} x ${escapeHtml(som.optimization.selected_height)}</b></div>` : ""}
    ${som.optimization?.available ? `<div><span>Denenen aralık</span><b>${escapeHtml(som.optimization.start_size)} - ${escapeHtml(som.optimization.max_size)}</b></div>` : ""}
    ${som.optimization?.available ? `<div><span>En iyi skor</span><b>${formatNumber(som.optimization.selected_score)}</b></div>` : ""}
  `;

  const gridStyle = `grid-template-columns: repeat(${Number(som.width || 1)}, minmax(54px, 1fr));`;
  const lowOccupancyNote = Number(som.sample_count || 0) < cells.length / 2
    ? `Bu haritada kayıt sayısı hücre sayısına göre düşük. Daha okunur kümeler için ${Math.max(2, Math.ceil(Math.sqrt(Number(som.sample_count || 1))))}x${Math.max(2, Math.ceil(Math.sqrt(Number(som.sample_count || 1))))} gibi daha küçük grid deneyebilirsin.`
    : "";
  const representativeNote = somRepresentativeView
    ? `Temsilci seçim aktif: (${somRepresentativeView.cell.x},${somRepresentativeView.cell.y}) hücresi ve komşuluk bölgesi gösteriliyor. Dolu hücre ortalamasının altında kalan hücreler aday dışı bırakıldı.`
    : "";
  somCanvas.innerHTML = `
    ${renderSomOptimizationBlock(som.optimization)}
    ${representativeNote ? `<div class="muted-box som-representative-note">${escapeHtml(representativeNote)}</div>` : ""}
    <div class="som-grid" style="${gridStyle}">
      ${cells.map((cell, index) => somCellMarkup(cell, index, mode, metricKey, metric, min, max, uValues, somRepresentativeView)).join("")}
    </div>
    ${som.interpretation ? `<p class="section-note">${escapeHtml(som.interpretation)}</p>` : ""}
    ${lowOccupancyNote ? `<p class="section-note">${escapeHtml(lowOccupancyNote)}</p>` : ""}
    <p class="section-note">${escapeHtml(som.training_note || som.source_note || "")}</p>
  `;
  somCanvas.querySelectorAll(".som-cell").forEach((cellNode) => {
    cellNode.addEventListener("click", () => {
      const index = Number(cellNode.dataset.index || 0);
      renderSomDetails(cells[index], metric);
    });
  });
  renderSomClassOptions(metricKey, metric);
  renderSomClassAnalysis(metricKey, metric);
  if (somRepresentativeView) renderSomRepresentativeDetails(somRepresentativeView, metricKey, metric);
}

function renderSomClassOptions(metricKey, metric) {
  if (!somClassTarget) return;
  const rows = Array.isArray(latestSom?.analysis_rows) ? latestSom.analysis_rows : [];
  toggleSomNumericBinInputs(metric);
  latestSomClassTargets = buildSomClassTargets(rows, metricKey, metric);
  if (!latestSomClassTargets.length) {
    somClassTarget.innerHTML = `<option value="">Veri yok</option>`;
    return;
  }
  const current = somClassTarget.value;
  somClassTarget.innerHTML = latestSomClassTargets.map((target) => `<option value="${escapeHtml(target.id)}">${escapeHtml(target.label)}</option>`).join("");
  somClassTarget.value = latestSomClassTargets.some((item) => item.id === current) ? current : latestSomClassTargets[0].id;
}

function regressionMetricCard(target, metric) {
  return `
    <div class="ml-card">
      <b>${escapeHtml(regressionTargetLabel(target))}</b>
      <span>MAE: ${formatNumber(metric.mae)}</span>
      <span>RMSE: ${formatNumber(metric.rmse)}</span>
      <span>R²: ${formatNumber(metric.r2)}</span>
      <span>Test: ${escapeHtml(metric.test_count || 0)} kayıt</span>
    </div>
  `;
}

function regressionTargetLabel(target) {
  return {
    peak_base_shear: "Maksimum taban kesmesi",
    max_rotation: "Maksimum plastik rotasyon",
    max_story_drift_ratio: "Maksimum göreli kat ötelenmesi",
    max_displacement: "Tepe deplasmanı",
    fema_ductility_mu: "FEMA 440 süneklik μ",
    fema_beta_eff_percent: "FEMA 440 etkin sönüm",
    fema_target_displacement_proxy: "FEMA 440 hedef deplasman proxy"
  }[target] || target;
}

function toggleSomNumericBinInputs(metric) {
  const isNumeric = metric?.type === "number";
  const binLabel = somBinCount?.closest("label");
  const customLabel = somCustomBins?.closest("label");
  if (binLabel) binLabel.style.display = isNumeric ? "" : "none";
  if (customLabel) customLabel.style.display = isNumeric ? "" : "none";
}

function renderSomClassAnalysis(metricKey = somResultMetric?.value || latestSom?.result_metric || "critical_state", metric = null) {
  if (!somClassAnalysis) return;
  const rows = Array.isArray(latestSom?.analysis_rows) ? latestSom.analysis_rows : [];
  const metrics = latestSom?.result_metrics || defaultSomMetrics;
  const resolvedMetric = metric || metrics[metricKey] || defaultSomMetrics.critical_state;
  if (!rows.length) {
    somClassAnalysis.innerHTML = `<div class="muted-box">SOM sınıf analizi için ham kayıt bulunamadı. SOM'u yeniden eğitmek gerekebilir.</div>`;
    return;
  }
  const target = latestSomClassTargets.find((item) => item.id === somClassTarget?.value) || latestSomClassTargets[0];
  if (!target) {
    somClassAnalysis.innerHTML = `<div class="muted-box">Seçili sonuç için sınıf/aralık üretilemedi.</div>`;
    return;
  }
  let activeTarget = target;
  let selectedRows = rows.filter((row) => somRowMatchesTarget(row, activeTarget, metricKey));
  if (!selectedRows.length) {
    activeTarget = latestSomClassTargets.find((item) => rows.some((row) => somRowMatchesTarget(row, item, metricKey))) || activeTarget;
    if (somClassTarget && activeTarget?.id) somClassTarget.value = activeTarget.id;
    selectedRows = rows.filter((row) => somRowMatchesTarget(row, activeTarget, metricKey));
  }
  const otherRows = rows.filter((row) => !somRowMatchesTarget(row, activeTarget, metricKey));
  if (!selectedRows.length) {
    somClassAnalysis.innerHTML = `<div class="muted-box">Seçili sınıf/aralık için kayıt bulunamadı.</div>`;
    return;
  }
  const features = latestSom?.selected_x_columns || [];
  const featureRows = summarizeSomFeatureComparison(selectedRows, otherRows, rows, features).slice(0, 10);
  const subgroups = summarizeSomSubgroups(selectedRows, rows, features);
  const mergedGroups = summarizeMergedSomBehaviorTypes(subgroups);
  const pureDense = summarizeSomPureDenseCells(selectedRows, rows, features);
  somClassAnalysis.innerHTML = `
    <div class="som-class-summary">
      <div><span>Seçili sınıf / aralık</span><b>${escapeHtml(activeTarget.label)}</b></div>
      <div><span>Kayıt</span><b>${escapeHtml(selectedRows.length)}</b></div>
      <div><span>Diğer kayıt</span><b>${escapeHtml(otherRows.length)}</b></div>
      <div><span>Hücre</span><b>${escapeHtml(countUniqueCells(selectedRows))}</b></div>
    </div>
    <div class="som-analysis-block">
      <b>A. Seçilen sınıf vs diğer tüm sınıflar</b>
      <p class="section-note">Aşağıda seçilen sınıfa düşen kayıtların X parametre profili, diğer tüm kayıtlarla karşılaştırılır. Skor büyüdükçe ayırt edicilik artar.</p>
      ${renderSomFeatureComparisonTable(featureRows)}
    </div>
    <div class="som-analysis-block">
      <b>B. Seçilen sınıfın kendi içindeki alt davranış grupları</b>
      <p class="section-note">Aynı sonuca farklı SOM bölgelerinde ulaşılan her hücre ayrı değerlendirilir. Böylece aynı performans düzeyine farklı yapısal mekanizmalarla gidilmiş olabilecek tüm hücrelerin temsilci özellikleri tek tek görülebilir.</p>
      ${renderSomMergedGroups(mergedGroups)}
      ${renderSomSubgroups(subgroups)}
    </div>
    <div class="som-analysis-block">
      <b>C. En saf ve en yoğun SOM hücreleri</b>
      <p class="section-note">Seçilen sınıfın hem temsil gücü yüksek hem de örnek sayısı yoğun hücreleri ayrıca işaretlenir. Ortak kalan sürücüler etkili parametreler olarak yorumlanabilir.</p>
      ${renderSomPureDense(pureDense)}
    </div>
  `;
}

function buildSomClassTargets(rows, metricKey, metric) {
  if (!rows.length || !metric) return [];
  if (metric.type === "category") {
    const counts = new Map();
    rows.forEach((row) => {
      const rawValue = String(row[metricKey] ?? "").trim();
      const normalized = somCategoryComparableValue(rawValue);
      if (!normalized || normalized === "-") return;
      const current = counts.get(normalized) || { rawValue, count: 0 };
      current.count += 1;
      counts.set(normalized, current);
    });
    return [...counts.entries()]
      .sort((a, b) => Number(b[1].count) - Number(a[1].count))
      .map(([normalized, item]) => ({
        id: `cat:${normalized}`,
        kind: "category",
        value: normalized,
        rawValue: item.rawValue,
        label: `${displayCategory(item.rawValue)} (${item.count})`,
      }));
  }
  const values = rows.map((row) => Number(row[metricKey])).filter(Number.isFinite).sort((a, b) => a - b);
  if (!values.length) return [];
  const customTargets = parseSomCustomBins(metric);
  if (customTargets.length) {
    return customTargets
      .map((target, index) => {
        const count = rows.filter((row) => somRowMatchesTarget(row, target, metricKey)).length;
        if (!count) return null;
        return {
          ...target,
          id: `custom:${index}`,
          label: `${formatMetricValue(target.low, metric)} - ${formatMetricValue(target.high, metric)} (${count})`,
        };
      })
      .filter(Boolean);
  }
  const binCount = Math.max(3, Math.min(10, Number(somBinCount?.value || 5)));
  const min = values[0];
  const max = values[values.length - 1];
  if (Math.abs(max - min) < 1e-12) {
    return [{ id: "bin:0", kind: "number", low: min, high: max, includeHigh: true, label: `${formatMetricValue(min, metric)} (${values.length})` }];
  }
  const step = (max - min) / binCount;
  const targets = [];
  for (let index = 0; index < binCount; index += 1) {
    const low = min + step * index;
    const high = index === binCount - 1 ? max : min + step * (index + 1);
    const includeHigh = index === binCount - 1;
    const target = { id: `bin:${index}`, kind: "number", low, high, includeHigh };
    const count = rows.filter((row) => somRowMatchesTarget(row, target, metricKey)).length;
    if (!count) continue;
    targets.push({
      ...target,
      label: `${formatMetricValue(low, metric)} - ${formatMetricValue(high, metric)} (${count})`,
    });
  }
  return targets;
}

function scheduleSomClassAnalysisRender() {
  window.clearTimeout(somClassRenderTimer);
  if (somClassAnalysis) {
    somClassAnalysis.innerHTML = `<div class="muted-box">Seçilen sınıf/aralık hesaplanıyor...</div>`;
  }
  somClassRenderTimer = window.setTimeout(() => renderSomClassAnalysis(), 0);
}

function parseSomCustomBins(metric) {
  const raw = String(somCustomBins?.value || "").trim();
  if (!raw) return [];
  const parts = raw.split(";").map((item) => item.trim()).filter(Boolean);
  const targets = [];
  for (let index = 0; index < parts.length; index += 1) {
    const part = parts[index].replace(",", ".");
    const match = /^\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*$/.exec(part);
    if (!match) continue;
    const low = Number(match[1]);
    const high = Number(match[2]);
    if (!Number.isFinite(low) || !Number.isFinite(high) || high < low) continue;
    targets.push({
      kind: "number",
      low,
      high,
      includeHigh: index === parts.length - 1,
    });
  }
  return targets;
}

function somRowMatchesTarget(row, target, metricKey) {
  if (!target) return false;
  if (target.kind === "category") return somCategoryComparableValue(row[metricKey]) === somCategoryComparableValue(target.value);
  const value = Number(row[metricKey]);
  return Number.isFinite(value) && value >= Number(target.low) && (target.includeHigh ? value <= Number(target.high) : value < Number(target.high));
}

function somCategoryComparableValue(value) {
  return String(value ?? "").trim().replace(/\s+/g, " ").toLowerCase();
}

function summarizeSomFeatureComparison(selectedRows, otherRows, allRows, features) {
  return features.map((feature) => {
    const kind = latestSom?.x_columns?.[feature]?.kind || "";
    if (kind === "nominal") {
      const selectedStats = nominalStats(selectedRows, feature);
      const otherStats = nominalStats(otherRows, feature);
      const globalStats = nominalStats(allRows, feature);
      const score = Math.abs(Number(selectedStats.dominant_ratio || 0) - Number(otherStats.dominant_ratio || 0));
      return {
        feature,
        kind,
        score,
        selected: selectedStats,
        other: otherStats,
        global: globalStats,
      };
    }
    const selectedValues = selectedRows.map((row) => somComparableValue(feature, row[feature])).filter(Number.isFinite);
    const otherValues = otherRows.map((row) => somComparableValue(feature, row[feature])).filter(Number.isFinite);
    const allValues = allRows.map((row) => somComparableValue(feature, row[feature])).filter(Number.isFinite);
    const selectedStats = numericStats(selectedValues);
    const otherStats = numericStats(otherValues);
    const globalStats = numericStats(allValues);
    const std = Number(globalStats.std || 0);
    const score = std > 1e-12 ? Math.abs(Number(selectedStats.avg || 0) - Number(otherStats.avg || 0)) / std : Math.abs(Number(selectedStats.avg || 0) - Number(otherStats.avg || 0));
    return {
      feature,
      kind,
      score,
      selected: selectedStats,
      other: otherStats,
      global: globalStats,
    };
  }).sort((a, b) => Number(b.score || 0) - Number(a.score || 0));
}

function summarizeSomSubgroups(selectedRows, allRows, features) {
  const allCellTotals = groupRowsByCell(allRows);
  const selectedCellTotals = groupRowsByCell(selectedRows);
  const selectedFeatureSummary = summarizeSomFeatureComparison(selectedRows, [], selectedRows, features);
  return [...selectedCellTotals.entries()].map(([key, rowsInCell], index) => {
    const allInCell = allCellTotals.get(key) || [];
    const purity = rowsInCell.length / Math.max(allInCell.length, 1);
    const profile = summarizeSomFeatureComparison(rowsInCell, selectedRows.filter((row) => cellKeyOf(row) !== key), selectedRows, features);
    const distinctive = profile.slice(0, 4);
    return {
      tip: `Tip ${String.fromCharCode(65 + index)}`,
      cell: key,
      count: rowsInCell.length,
      totalCellCount: allInCell.length,
      purity,
      drivers: distinctive,
      profile,
      overlapScore: commonDriverCount(distinctive, selectedFeatureSummary.slice(0, 6)),
    };
  }).sort((a, b) => Number(b.count) - Number(a.count) || Number(b.purity) - Number(a.purity));
}

function summarizeMergedSomBehaviorTypes(groups) {
  if (!groups.length) return [];
  if (groups.length === 1) return [];
  const targetClusterCount = Math.max(2, Math.min(5, Math.round(Math.sqrt(groups.length))));
  let clusters = groups.map((group) => ({ groups: [group] }));
  while (clusters.length > targetClusterCount) {
    let bestI = 0;
    let bestJ = 1;
    let bestSimilarity = -Infinity;
    let bestSharedCount = 0;
    for (let i = 0; i < clusters.length; i += 1) {
      for (let j = i + 1; j < clusters.length; j += 1) {
        const similarity = somClusterSimilarity(clusters[i], clusters[j]);
        const sharedCount = sharedDriverKeysBetweenClusters(clusters[i], clusters[j]).length;
        if (sharedCount > 0 && (similarity > bestSimilarity || (similarity === bestSimilarity && sharedCount > bestSharedCount))) {
          bestSimilarity = similarity;
          bestSharedCount = sharedCount;
          bestI = i;
          bestJ = j;
        }
      }
    }
    if (!(bestSimilarity > 0) || bestSharedCount <= 0) break;
    clusters = clusters.filter((_, index) => index !== bestI && index !== bestJ).concat({
      groups: [...clusters[bestI].groups, ...clusters[bestJ].groups],
    });
  }
  return clusters
    .map((cluster, index) => {
      const cells = cluster.groups.map((group) => group.cell);
      const totalCount = cluster.groups.reduce((sum, group) => sum + Number(group.count || 0), 0);
      const averagePurity = cluster.groups.reduce((sum, group) => sum + Number(group.purity || 0), 0) / Math.max(cluster.groups.length, 1);
      return {
        name: `Davranış Tipi ${index + 1}`,
        groups: cluster.groups.sort((a, b) => Number(b.count) - Number(a.count)),
        cells,
        totalCount,
        averagePurity,
        sharedDrivers: sharedDriversForGroups(cluster.groups),
        mergeStrength: clusterMergeStrength(cluster.groups),
      };
    })
    .filter((cluster) => cluster.groups.length > 1 && cluster.sharedDrivers.length > 0)
    .sort((a, b) => Number(b.totalCount) - Number(a.totalCount));
}

function somClusterSimilarity(leftCluster, rightCluster) {
  const leftKeys = clusterDriverKeys(leftCluster.groups);
  const rightKeys = clusterDriverKeys(rightCluster.groups);
  const intersection = [...leftKeys].filter((key) => rightKeys.has(key));
  const unionSize = new Set([...leftKeys, ...rightKeys]).size || 1;
  const jaccard = intersection.length / unionSize;
  const weighted = driverWeightOverlap(leftCluster.groups, rightCluster.groups, intersection);
  return jaccard * 0.7 + weighted * 0.3;
}

function sharedDriversForGroups(groups) {
  const counts = new Map();
  groups.forEach((group) => {
    (group.drivers || []).forEach((driver) => {
      const key = `${driver.feature}:${driverDirection(driver)}`;
      const current = counts.get(key) || { feature: driver.feature, direction: driverDirection(driver), count: 0, driver };
      current.count += 1;
      counts.set(key, current);
    });
  });
  return [...counts.values()]
    .filter((item) => item.count >= Math.max(2, Math.ceil(groups.length / 2)))
    .sort((a, b) => Number(b.count) - Number(a.count) || Number(b.driver?.score || 0) - Number(a.driver?.score || 0))
    .slice(0, 5)
    .map((item) => item.driver);
}

function clusterDriverKeys(groups) {
  return new Set(
    groups.flatMap((group) => (group.drivers || []).map((driver) => driverKey(driver)))
  );
}

function sharedDriverKeysBetweenClusters(leftCluster, rightCluster) {
  const left = clusterDriverKeys(leftCluster.groups);
  const right = clusterDriverKeys(rightCluster.groups);
  return [...left].filter((key) => right.has(key));
}

function driverWeightOverlap(leftGroups, rightGroups, sharedKeys) {
  if (!sharedKeys.length) return 0;
  const leftWeights = clusterDriverWeightMap(leftGroups);
  const rightWeights = clusterDriverWeightMap(rightGroups);
  const overlaps = sharedKeys.map((key) => {
    const leftWeight = Number(leftWeights.get(key) || 0);
    const rightWeight = Number(rightWeights.get(key) || 0);
    const maxWeight = Math.max(leftWeight, rightWeight, 1e-9);
    return Math.min(leftWeight, rightWeight) / maxWeight;
  });
  return overlaps.reduce((sum, value) => sum + value, 0) / overlaps.length;
}

function clusterDriverWeightMap(groups) {
  const weights = new Map();
  groups.forEach((group) => {
    (group.drivers || []).forEach((driver) => {
      const key = driverKey(driver);
      weights.set(key, (weights.get(key) || 0) + Number(driver.score || 0) * Number(group.count || 1));
    });
  });
  return weights;
}

function clusterMergeStrength(groups) {
  const shared = sharedDriversForGroups(groups);
  if (!shared.length) return 0;
  return shared.reduce((sum, driver) => sum + Number(driver.score || 0), 0) / shared.length;
}

function summarizeSomPureDenseCells(selectedRows, allRows, features) {
  const allCellTotals = groupRowsByCell(allRows);
  const selectedCellTotals = groupRowsByCell(selectedRows);
  const selectedProfile = summarizeSomFeatureComparison(selectedRows, allRows.filter((row) => !selectedRows.includes(row)), allRows, features).slice(0, 8);
  const cells = [...selectedCellTotals.entries()].map(([key, rowsInCell]) => {
    const allInCell = allCellTotals.get(key) || [];
    const purity = rowsInCell.length / Math.max(allInCell.length, 1);
    const drivers = summarizeSomFeatureComparison(rowsInCell, selectedRows.filter((row) => cellKeyOf(row) !== key), selectedRows, features).slice(0, 4);
    return { cell: key, count: rowsInCell.length, totalCellCount: allInCell.length, purity, drivers };
  });
  const topPure = [...cells].sort((a, b) => Number(b.purity) - Number(a.purity) || Number(b.count) - Number(a.count)).slice(0, 3);
  const topDense = [...cells].sort((a, b) => Number(b.count) - Number(a.count) || Number(b.purity) - Number(a.purity)).slice(0, 3);
  const uniqueTopCells = [...new Map([...topPure, ...topDense].map((cell) => [cell.cell, cell])).values()];
  const repeated = commonDriversAcrossCells(uniqueTopCells, selectedProfile);
  const commonCells = topPure
    .filter((pureCell) => topDense.some((denseCell) => denseCell.cell === pureCell.cell))
    .map((pureCell) => {
      const denseCell = topDense.find((item) => item.cell === pureCell.cell) || pureCell;
      return {
        ...pureCell,
        densityRank: topDense.findIndex((item) => item.cell === pureCell.cell) + 1,
        purityRank: topPure.findIndex((item) => item.cell === pureCell.cell) + 1,
        drivers: pureCell.drivers || denseCell.drivers || [],
      };
    });
  const commonRepeated = commonDriversAcrossCells(commonCells, selectedProfile);
  return { topPure, topDense, repeated, unionCells: uniqueTopCells, commonCells, commonRepeated };
}

function renderSomFeatureComparisonTable(rows) {
  if (!rows.length) return `<p class="section-note">Karşılaştırma için yeterli ayırıcı X parametresi bulunamadı.</p>`;
  return `
    <table class="info-table">
      <thead><tr><th>Parametre</th><th>Seçili sınıf</th><th>Diğerleri</th><th>Genel</th><th>Skor</th></tr></thead>
      <tbody>${rows.map((item) => `
        <tr>
          <td>${escapeHtml(somFeatureLabel(item.feature))}</td>
          <td>${escapeHtml(renderSomFeatureStats(item.feature, item.kind, item.selected))}</td>
          <td>${escapeHtml(renderSomFeatureStats(item.feature, item.kind, item.other))}</td>
          <td>${escapeHtml(renderSomFeatureStats(item.feature, item.kind, item.global))}</td>
          <td>${formatNumber(item.score)}</td>
        </tr>
      `).join("")}</tbody>
    </table>
  `;
}

function renderSomSubgroups(groups) {
  if (!groups.length) return `<p class="section-note">Seçili sınıf için alt davranış grubu bulunamadı.</p>`;
  return `
    <div class="section-note">Alt tablo hücre bazlı ham ayrımı gösterir. Üstteki birleştirilmiş davranış tipleri ise benzer hücreleri birlikte yorumlar.</div>
    <table class="info-table">
      <thead><tr><th>Tip</th><th>Hücre</th><th>Kayıt</th><th>Saflık</th><th>Hücre toplamı</th><th>Ortak sürücü skoru</th><th>Bu hücreye özgü ayırıcılar</th></tr></thead>
      <tbody>
        ${groups.map((group) => `
          <tr>
            <td>${escapeHtml(group.tip)}</td>
            <td>${escapeHtml(group.cell)}</td>
            <td>${escapeHtml(group.count)}</td>
            <td>${formatPercent(group.purity)}</td>
            <td>${escapeHtml(group.totalCellCount)}</td>
            <td>${escapeHtml(group.overlapScore)}</td>
            <td>${group.drivers.length ? group.drivers.map((item) => `${escapeHtml(somFeatureLabel(item.feature))}: ${escapeHtml(shortSomDriverText(item))}`).join("<br>") : "-"}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderSomMergedGroups(groups) {
  if (!groups.length) {
    return `
      <div class="som-distinctive">
        <b>Birleştirilmiş davranış tipleri</b>
        <p class="section-note">Bu seçili sınıf için ortak ayırıcı parametreleri paylaşan hücre kümeleri bulunamadı. Bu yüzden davranış tipleri zorla birleştirilmedi; aşağıdaki hücre bazlı tabloyu ayrı ayrı okumak daha doğru.</p>
      </div>
    `;
  }
  return `
    <div class="som-distinctive">
      <b>Birleştirilmiş davranış tipleri</b>
      <p class="section-note">Buradaki birleşim yalnızca ortak ayırıcı parametre-durum paylaşan hücreler arasında yapılır. Yani önce ortak sürücüler aranır, sonra ancak bunları paylaşan hücreler aynı davranış tipine alınır.</p>
      <table class="info-table">
        <thead><tr><th>Birleşik tip</th><th>Hücreler</th><th>Toplam kayıt</th><th>Ort. saflık</th><th>Birleşim gücü</th><th>Ortak ayırıcılar</th></tr></thead>
        <tbody>
          ${groups.map((group) => `
            <tr>
              <td>${escapeHtml(group.name)}</td>
              <td>${escapeHtml(group.cells.join(", "))}</td>
              <td>${escapeHtml(group.totalCount)}</td>
              <td>${formatPercent(group.averagePurity)}</td>
              <td>${formatNumber(group.mergeStrength)}</td>
              <td>${group.sharedDrivers.length ? group.sharedDrivers.map((item) => `${escapeHtml(somFeatureLabel(item.feature))}: ${escapeHtml(shortSomDriverText(item))}`).join("<br>") : "-"}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderSomPureDense(summary) {
  const commonCells = Array.isArray(summary.commonCells) ? summary.commonCells : [];
  const commonRepeated = Array.isArray(summary.commonRepeated) ? summary.commonRepeated : [];
  const unionCells = Array.isArray(summary.unionCells) ? summary.unionCells : [];
  return `
    <div class="som-pure-dense-grid">
      <div>
        <b>En saf hücreler</b>
        ${renderSomCellRanking(summary.topPure)}
      </div>
      <div>
        <b>En yoğun hücreler</b>
        ${renderSomCellRanking(summary.topDense)}
      </div>
    </div>
    <div class="som-distinctive">
      <b>Birleşik üst hücre kümesi</b>
      <p class="section-note">En saf ve en yoğun listelerinin birleşimi alınır. Bu küme, seçilen sınıfın güçlü temsilci hücre havuzunu verir.</p>
      ${unionCells.length ? `
        <div class="som-mini-list">
          ${unionCells.map((cell) => `<div><b>${escapeHtml(cell.cell)}</b> · ${cell.count} kayıt · saflık ${formatPercent(cell.purity)}</div>`).join("")}
        </div>
      ` : `<p class="section-note">Birleşik üst hücre kümesi oluşturulamadı.</p>`}
    </div>
    <div class="som-distinctive">
      <b>Ortak ana temsilci hücreler</b>
      <p class="section-note">Hem en saf hem de en yoğun listesinde tekrar eden hücreler, seçilen sınıfın en karakteristik SOM hücreleri olarak yorumlanabilir.</p>
      ${commonCells.length ? `
        <div class="som-mini-list">
          ${commonCells.map((cell) => `<div><b>${escapeHtml(cell.cell)}</b> · ${cell.count} kayıt · saflık ${formatPercent(cell.purity)} · saf sıra ${cell.purityRank} · yoğun sıra ${cell.densityRank}</div>`).join("")}
        </div>
      ` : `<p class="section-note">En saf ve en yoğun listeleri arasında ortak hücre bulunamadı.</p>`}
    </div>
    <div class="som-distinctive">
      <b>Ortak ana temsilci hücrelere ait ortak kalan etkili parametreler</b>
      ${commonRepeated.length ? `
        <table class="info-table">
          <thead><tr><th>Parametre</th><th>Yön</th><th>Tekrar</th><th>Yorum</th></tr></thead>
          <tbody>${commonRepeated.map((item) => `
            <tr>
              <td>${escapeHtml(somFeatureLabel(item.feature))}</td>
              <td>${escapeHtml(item.direction)}</td>
              <td>${escapeHtml(item.count)}</td>
              <td>${escapeHtml(item.note)}</td>
            </tr>
          `).join("")}</tbody>
        </table>
      ` : `<p class="section-note">Ortak ana temsilci hücrelerde tekrar eden ortak sürücü bulunamadı.</p>`}
    </div>
    <div class="som-distinctive">
      <b>Tüm üst hücrelerde ortak kalan etkili parametreler</b>
      ${summary.repeated.length ? `
        <table class="info-table">
          <thead><tr><th>Parametre</th><th>Yön</th><th>Tekrar</th><th>Yorum</th></tr></thead>
          <tbody>${summary.repeated.map((item) => `
            <tr>
              <td>${escapeHtml(somFeatureLabel(item.feature))}</td>
              <td>${escapeHtml(item.direction)}</td>
              <td>${escapeHtml(item.count)}</td>
              <td>${escapeHtml(item.note)}</td>
            </tr>
          `).join("")}</tbody>
        </table>
      ` : `<p class="section-note">Üst hücrelerde tekrar eden ortak sürücü bulunamadı.</p>`}
    </div>
  `;
}

function renderSomCellRanking(cells) {
  if (!cells.length) return `<p class="section-note">Hücre bulunamadı.</p>`;
  return `
    <div class="som-mini-list">
      ${cells.map((cell) => `<div><b>${escapeHtml(cell.cell)}</b> · ${cell.count} kayıt · saflık ${formatPercent(cell.purity)}</div>`).join("")}
    </div>
  `;
}

function renderSomFeatureStats(feature, kind, stats) {
  if (!stats) return "-";
  if (kind === "nominal") {
    return `${displayCategory(stats.dominant || "-")} (${formatPercent(stats.dominant_ratio || 0)})`;
  }
  return `ort ${formatSomFeatureValue(feature, stats.avg)}`;
}

function shortSomDriverText(item) {
  if (!item) return "-";
  if (item.kind === "nominal") return `${displayCategory(item.selected?.dominant || "-")} baskin`;
  const avg = item.selected?.avg;
  const other = item.other?.avg;
  const direction = Number(avg) >= Number(other) ? "yüksek" : "düşük";
  return `${direction} / skor ${formatNumber(item.score)}`;
}

function groupRowsByCell(rows) {
  const map = new Map();
  rows.forEach((row) => {
    const key = cellKeyOf(row);
    const list = map.get(key) || [];
    list.push(row);
    map.set(key, list);
  });
  return map;
}

function cellKeyOf(row) {
  return `${row.cell_x},${row.cell_y}`;
}

function countUniqueCells(rows) {
  return new Set(rows.map((row) => cellKeyOf(row))).size;
}

function numericStats(values) {
  const clean = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!clean.length) return { avg: null, median: null, min: null, max: null, std: null, count: 0 };
  const avg = clean.reduce((sum, value) => sum + value, 0) / clean.length;
  const variance = clean.reduce((sum, value) => sum + (value - avg) ** 2, 0) / clean.length;
  return {
    avg,
    median: percentileSorted(clean, 0.5),
    min: clean[0],
    max: clean[clean.length - 1],
    std: Math.sqrt(variance),
    count: clean.length,
  };
}

function nominalStats(rows, feature) {
  const counts = new Map();
  rows.forEach((row) => {
    const key = String(row[feature] ?? "-");
    counts.set(key, (counts.get(key) || 0) + 1);
  });
  const sorted = [...counts.entries()].sort((a, b) => Number(b[1]) - Number(a[1]));
  const [dominant, dominantCount] = sorted[0] || ["-", 0];
  const total = rows.length || 0;
  return { dominant, dominant_count: dominantCount, dominant_ratio: total ? dominantCount / total : 0, count: total };
}

function percentileSorted(values, ratio) {
  if (!values.length) return null;
  const index = (values.length - 1) * ratio;
  const low = Math.floor(index);
  const high = Math.ceil(index);
  if (low === high) return values[low];
  return values[low] + (values[high] - values[low]) * (index - low);
}

function somComparableValue(feature, value) {
  if (["concrete_class"].includes(feature)) {
    const text = String(value || "C30").toUpperCase();
    return Number((text.match(/\d+/) || [30])[0]);
  }
  if (["steel_class"].includes(feature)) {
    const text = String(value || "B420C").toUpperCase();
    return text.includes("500") ? 500 : 420;
  }
  if (["soil_class"].includes(feature)) {
    return ({ ZA: 1, ZB: 2, ZC: 3, ZD: 4, ZE: 5 })[String(value || "ZC").toUpperCase()] || 3;
  }
  return Number(value);
}

function commonDriverCount(left, right) {
  const rightKeys = new Set(right.map((item) => driverKey(item)));
  return left.filter((item) => rightKeys.has(driverKey(item))).length;
}

function commonDriversAcrossCells(cells, selectedProfile) {
  const map = new Map();
  cells.forEach((cell) => {
    (cell.drivers || []).forEach((driver) => {
      const direction = driverDirection(driver);
      const key = driverKey(driver);
      const current = map.get(key) || { feature: driver.feature, direction, count: 0 };
      current.count += 1;
      map.set(key, current);
    });
  });
  const selectedKeys = new Set(selectedProfile.map((item) => driverKey(item)));
  return [...map.values()]
    .filter((item) => item.count >= 2)
    .sort((a, b) => Number(b.count) - Number(a.count))
    .map((item) => ({
      ...item,
      note: selectedKeys.has(`${item.feature}:${item.direction}`) ? "Seçili sınıf genel profilinde de tekrar ediyor." : "Üst hücrelerde tekrar ediyor.",
    }));
}

function driverDirection(item) {
  if (!item) return "-";
  if (item.kind === "nominal") return String(item.selected?.dominant || "-");
  return Number(item.selected?.avg || 0) >= Number(item.other?.avg || 0) ? "yüksek" : "düşük";
}

function driverKey(item) {
  if (!item) return "-";
  return `${item.feature}:${driverDirection(item)}`;
}

function renderSomOptimizationBlock(optimization) {
  if (!optimization?.available || !Array.isArray(optimization.candidates) || !optimization.candidates.length) return "";
  const candidates = [...optimization.candidates].sort((left, right) => Number(left.width || 0) - Number(right.width || 0));
  const selectedKey = `${optimization.selected_width}x${optimization.selected_height}`;
  return `
    <div class="som-optimization-block">
      <div class="som-optimization-head">
        <div>
          <b>SOM boyut optimizasyonu</b>
          <p class="section-note">Makale gorseli icin uygun ozet: her kare grid adayi icin birlesik optimizasyon skoru, QE, topographic error ve purity karsilastirilir; secilen optimum boyut vurgulanir.</p>
        </div>
        <div class="som-optimization-badges">
          <span>Secilen: ${escapeHtml(selectedKey)}</span>
          <span>Skor: ${formatNumber(optimization.selected_score)}</span>
          <span>Aday: ${escapeHtml(candidates.length)}</span>
        </div>
      </div>
      <div class="som-optimization-chart-wrap">
        ${somOptimizationSvg(candidates, selectedKey)}
      </div>
      <table class="info-table som-optimization-table">
        <thead><tr><th>Grid</th><th>Opt. skor</th><th>QE</th><th>Topo. error</th><th>Purity</th><th>Durum</th></tr></thead>
        <tbody>
          ${candidates.map((item) => {
            const key = `${item.width}x${item.height}`;
            const selected = key === selectedKey;
            return `
              <tr${selected ? ' class="som-opt-selected-row"' : ""}>
                <td>${escapeHtml(key)}</td>
                <td>${formatNumber(item.optimization_score)}</td>
                <td>${formatNumber(item.quantization_error)}</td>
                <td>${formatNumber(item.topographic_error)}</td>
                <td>${item.purity === null || item.purity === undefined ? "-" : formatPercent(item.purity)}</td>
                <td>${selected ? "Secildi" : "-"}</td>
              </tr>
            `;
          }).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function somOptimizationSvg(candidates, selectedKey) {
  const width = 920;
  const height = 270;
  const margin = { top: 30, right: 26, bottom: 40, left: 54 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;
  const values = candidates.map((item) => Number(item.optimization_score || 0));
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const xStep = candidates.length > 1 ? innerWidth / (candidates.length - 1) : 0;
  const xFor = (index) => margin.left + index * xStep;
  const yFor = (value) => {
    if (Math.abs(maxValue - minValue) < 1e-9) return margin.top + innerHeight / 2;
    const ratio = (Number(value) - minValue) / (maxValue - minValue);
    return margin.top + innerHeight - ratio * innerHeight;
  };
  const linePath = candidates.map((item, index) => `${index === 0 ? "M" : "L"} ${xFor(index).toFixed(2)} ${yFor(item.optimization_score).toFixed(2)}`).join(" ");
  const gridTicks = 4;
  const yTicks = Array.from({ length: gridTicks + 1 }, (_, index) => minValue + ((maxValue - minValue) * index) / gridTicks);
  return `
    <svg class="analysis-chart som-optimization-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="SOM boyut optimizasyon skoru grafiği">
      <text x="${margin.left}" y="18" class="chart-title">SOM boyut optimizasyonu</text>
      ${yTicks.map((tick) => {
        const y = yFor(tick);
        return `
          <line x1="${margin.left}" y1="${y}" x2="${width - margin.right}" y2="${y}" class="grid-line"></line>
          <text x="${margin.left - 10}" y="${y + 4}" text-anchor="end" class="axis-label">${formatNumber(tick)}</text>
        `;
      }).join("")}
      <line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + innerHeight}" class="axis-line"></line>
      <line x1="${margin.left}" y1="${margin.top + innerHeight}" x2="${width - margin.right}" y2="${margin.top + innerHeight}" class="axis-line"></line>
      <path d="${linePath}" fill="none" stroke="#176b87" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"></path>
      ${candidates.map((item, index) => {
        const key = `${item.width}x${item.height}`;
        const selected = key === selectedKey;
        const x = xFor(index);
        const y = yFor(item.optimization_score);
        return `
          <circle cx="${x}" cy="${y}" r="${selected ? 6.5 : 4.5}" fill="${selected ? "#b42318" : "#176b87"}" stroke="#ffffff" stroke-width="2"></circle>
          <text x="${x}" y="${margin.top + innerHeight + 18}" text-anchor="middle" class="axis-label">${escapeHtml(key)}</text>
          ${selected ? `<text x="${x}" y="${y - 12}" text-anchor="middle" class="axis-label">optimum</text>` : ""}
        `;
      }).join("")}
      <text x="${margin.left - 38}" y="${margin.top + innerHeight / 2}" transform="rotate(-90 ${margin.left - 38} ${margin.top + innerHeight / 2})" class="axis-name">Birleşik skor</text>
      <text x="${margin.left + innerWidth / 2}" y="${height - 8}" text-anchor="middle" class="axis-name">Grid boyutu</text>
    </svg>
  `;
}

function computeSomRepresentative(som, metricKey, metric) {
  const cells = Array.isArray(som.cells) ? som.cells : [];
  const filled = cells.filter((cell) => Number(cell.hit_count || 0) > 0);
  if (!filled.length) return null;
  const avgHitCount = filled.reduce((sum, cell) => sum + Number(cell.hit_count || 0), 0) / filled.length;
  const candidates = filled.filter((cell) => Number(cell.hit_count || 0) >= avgHitCount);
  if (!candidates.length) return null;
  const cellMap = new Map(cells.map((cell) => [somCellKey(cell), cell]));
  const numericValues = filled.map((cell) => Number(somMetricSummary(cell, metricKey).avg)).filter(Number.isFinite);
  const valueRange = numericValues.length ? Math.max(...numericValues) - Math.min(...numericValues) : 0;
  const maxCellHit = Math.max(...filled.map((cell) => Number(cell.hit_count || 0)), 1);
  const scored = candidates.map((cell) => {
    const neighbors = somNeighborCells(cell, cellMap).filter((neighbor) => Number(neighbor.hit_count || 0) > 0);
    const neighborhoodHitCount = neighbors.reduce((sum, neighbor) => sum + Number(neighbor.hit_count || 0), 0);
    return { cell, neighbors, neighborhoodHitCount, cellHitScore: Number(cell.hit_count || 0) / maxCellHit };
  });
  const maxNeighborhoodHit = Math.max(...scored.map((item) => item.neighborhoodHitCount), 1);
  const finalScored = scored.map((item) => {
    const cellSummary = somMetricSummary(item.cell, metricKey);
    const densityScore = 0.5 * item.cellHitScore + 0.5 * (item.neighborhoodHitCount / maxNeighborhoodHit);
    const resultScore = metric?.type === "category"
      ? categoryNeighborhoodScore(item.neighbors, metricKey, cellSummary)
      : numericNeighborhoodScore(item.neighbors, metricKey, cellSummary, valueRange);
    return { ...item, score: 0.55 * resultScore.proximity + 0.45 * densityScore, densityScore, resultScore };
  }).sort((a, b) => (
    b.score - a.score ||
    b.densityScore - a.densityScore ||
    b.resultScore.proximity - a.resultScore.proximity ||
    Number(b.cell.hit_count || 0) - Number(a.cell.hit_count || 0)
  ));
  const best = finalScored[0];
  const representativeKey = somCellKey(best.cell);
  const visibleKeys = new Set(best.neighbors.map(somCellKey));
  visibleKeys.add(representativeKey);
  return {
    metricKey,
    metricLabel: metric?.label || metricKey,
    averageHitCount: avgHitCount,
    candidates: finalScored,
    cell: best.cell,
    representativeKey,
    visibleKeys,
    neighbors: best.neighbors,
    score: best.score,
    densityScore: best.densityScore,
    resultScore: best.resultScore,
    neighborhoodHitCount: best.neighborhoodHitCount,
  };
}

function somCellKey(cell) {
  return `${Number(cell?.x ?? -1)},${Number(cell?.y ?? -1)}`;
}

function somNeighborCells(cell, cellMap) {
  const x = Number(cell?.x ?? 0);
  const y = Number(cell?.y ?? 0);
  const neighbors = [];
  for (let dy = -1; dy <= 1; dy += 1) {
    for (let dx = -1; dx <= 1; dx += 1) {
      const neighbor = cellMap.get(`${x + dx},${y + dy}`);
      if (neighbor) neighbors.push(neighbor);
    }
  }
  return neighbors;
}

function numericNeighborhoodScore(neighbors, metricKey, cellSummary, valueRange) {
  const cellValue = Number(cellSummary.avg);
  const weighted = neighbors.reduce((acc, neighbor) => {
    const value = Number(somMetricSummary(neighbor, metricKey).avg);
    const weight = Number(neighbor.hit_count || 0);
    if (!Number.isFinite(value) || weight <= 0) return acc;
    acc.sum += value * weight;
    acc.weight += weight;
    return acc;
  }, { sum: 0, weight: 0 });
  const neighborhoodValue = weighted.weight ? weighted.sum / weighted.weight : null;
  if (!Number.isFinite(cellValue) || !Number.isFinite(neighborhoodValue)) {
    return { proximity: 0, cellValue, neighborhoodValue, detail: "Sayısal sonuç değeri eksik." };
  }
  const scale = Math.max(Math.abs(valueRange), Math.abs(neighborhoodValue), 1e-9);
  const distance = Math.abs(cellValue - neighborhoodValue);
  return {
    proximity: Math.max(0, 1 - Math.min(distance / scale, 1)),
    cellValue,
    neighborhoodValue,
    detail: `fark ${formatNumber(distance)}`,
  };
}

function categoryNeighborhoodScore(neighbors, metricKey, cellSummary) {
  const counts = {};
  neighbors.forEach((neighbor) => {
    const summary = somMetricSummary(neighbor, metricKey);
    Object.entries(summary.counts || {}).forEach(([category, count]) => {
      counts[category] = (counts[category] || 0) + Number(count || 0);
    });
  });
  const ranked = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const neighborhoodCategory = ranked[0]?.[0] || "-";
  const cellCategory = cellSummary.dominant || "-";
  const cellPurity = Number(cellSummary.dominant_count || 0) / Math.max(Number(cellSummary.count || 0), 1);
  const neighborhoodTotal = ranked.reduce((sum, item) => sum + Number(item[1] || 0), 0);
  const neighborhoodPurity = neighborhoodTotal ? Number(ranked[0]?.[1] || 0) / neighborhoodTotal : 0;
  const sameCategory = cellCategory === neighborhoodCategory && cellCategory !== "-";
  return {
    proximity: sameCategory ? 0.65 + 0.35 * Math.min(cellPurity, neighborhoodPurity) : 0.25 * cellPurity,
    cellCategory,
    neighborhoodCategory,
    cellPurity,
    neighborhoodPurity,
    detail: sameCategory ? "komşuluk baskın sınıfıyla uyumlu" : "komşuluk baskın sınıfından farklı",
  };
}

function renderSomRepresentativeDetails(selection, metricKey, metric) {
  if (!somDetails || !selection) return;
  const result = selection.resultScore || {};
  const isCategory = metric?.type === "category";
  const resultLine = isCategory
    ? `${displayCategory(result.cellCategory || "-")} / komşuluk ${displayCategory(result.neighborhoodCategory || "-")}`
    : `${formatMetricValue(result.cellValue, metric)} / komşuluk ${formatMetricValue(result.neighborhoodValue, metric)}`;
  const neighborText = selection.neighbors
    .filter((cell) => somCellKey(cell) !== selection.representativeKey)
    .map((cell) => `(${cell.x},${cell.y})`)
    .join(", ") || "-";
  somDetails.innerHTML = `
    <div class="som-detail-grid">
      <div><span>Temsilci hücre</span><b>${escapeHtml(selection.cell.x)}, ${escapeHtml(selection.cell.y)}</b></div>
      <div><span>Yansıtılan sonuç</span><b>${escapeHtml(selection.metricLabel)}</b></div>
      <div><span>Temsil skoru</span><b>${formatNumber(selection.score)}</b></div>
      <div><span>Dolu hücre ort. kayıt</span><b>${formatNumber(selection.averageHitCount)}</b></div>
      <div><span>Temsilci kayıt</span><b>${escapeHtml(selection.cell.hit_count || 0)}</b></div>
      <div><span>Komşuluk kayıt</span><b>${escapeHtml(selection.neighborhoodHitCount || 0)}</b></div>
      <div><span>Sonuç / komşuluk</span><b>${resultLine}</b></div>
      <div><span>Sonuç yakınlığı</span><b>${formatNumber(result.proximity)}</b><small>${escapeHtml(result.detail || "")}</small></div>
      <div><span>Yoğunluk skoru</span><b>${formatNumber(selection.densityScore)}</b></div>
    </div>
    <div class="muted-box">
      <b>Temsilci seçim stratejisi</b>
      <p>Dolu hücrelerin ortalama kayıt sayısı altında kalan hücreler elendi. Kalan hücrelerde, seçili sonuç değerinin komşuluk bölgesinin kayıt ağırlıklı sonucuna yakınlığı ve hücre+komşuluk kayıt yoğunluğu birlikte puanlandı.</p>
      <p><b>Gösterilen komşular:</b> ${escapeHtml(neighborText)}</p>
    </div>
  `;
}

function somCellMarkup(cell, index, mode, metricKey, metric, min, max, uValues, representativeView = null) {
  const hitCount = Number(cell.hit_count || 0);
  const cellKey = somCellKey(cell);
  const isRepresentative = representativeView?.representativeKey === cellKey;
  const isNeighbor = representativeView?.visibleKeys?.has(cellKey) && !isRepresentative;
  const isOutsideRepresentative = representativeView && !representativeView.visibleKeys.has(cellKey);
  const repClass = isRepresentative ? " som-cell-representative" : isNeighbor ? " som-cell-representative-neighbor" : isOutsideRepresentative ? " som-cell-representative-outside" : "";
  if (!hitCount && mode !== "umatrix") {
    return `
      <button class="som-cell som-cell-empty${repClass}" type="button" data-index="${index}" title="Boş hücre">
        <span>boş</span>
      </button>
    `;
  }
  let label = hitCount ? String(hitCount) : "";
  let sub = "";
  let color = "#f8fafc";
  if (mode === "umatrix") {
    const value = Number(uValues.get(`${cell.x},${cell.y}`) || 0);
    label = formatNumber(value);
    sub = "U";
    color = numericColor(value, min, max, "#e8f3ff", "#075985");
  } else if (mode === "hits") {
    label = String(hitCount);
    sub = "model/yön";
    color = numericColor(hitCount, min, max, "#edf7ed", "#166534");
  } else if (metric.type === "category") {
    const summary = somMetricSummary(cell, metricKey);
    const category = summary.dominant || "-";
    label = displayCategory(category);
    sub = hitCount ? `${summary.dominant_count || 0}/${hitCount} kayıt` : "";
    color = categoryColor(category);
  } else {
    const value = Number(somMetricSummary(cell, metricKey).avg);
    label = Number.isFinite(value) ? formatMetricValue(value, metric) : "-";
    sub = hitCount ? `${hitCount} kayıt` : "";
    color = Number.isFinite(value) ? numericColor(value, min, max, "#eef7f4", "#0f766e") : "#f8fafc";
  }
  const title = `(${cell.x},${cell.y}) / ${hitCount} kayıt`;
  return `
    <button class="som-cell${repClass}" type="button" data-index="${index}" style="background:${color}" title="${escapeHtml(title)}">
      <b>${escapeHtml(label)}</b>
      <span>${escapeHtml(sub)}</span>
    </button>
  `;
}

function renderSomDetails(cell, metric) {
  if (!somDetails || !cell) return;
  const models = Array.isArray(cell.models) ? cell.models : [];
  const means = cell.feature_means || {};
  const distinctive = Array.isArray(cell.distinctive_features) ? cell.distinctive_features : [];
  const metricKey = somResultMetric?.value || latestSom?.result_metric || "max_story_drift_ratio";
  const summary = somMetricSummary(cell, metricKey);
  const hasMetricSummary = Number(summary.count || 0) > 0;
  const selectedResultText = !hasMetricSummary
    ? "-"
    : metric.type === "category"
    ? `${escapeHtml(displayCategory(summary.dominant || "-"))} (${escapeHtml(summary.dominant_count || 0)}/${escapeHtml(cell.hit_count || 0)})`
    : `${formatMetricValue(summary.avg, metric)} / min ${formatMetricValue(summary.min, metric)} / maks ${formatMetricValue(summary.max, metric)} / std ${formatNumber(summary.std)}`;
  somDetails.innerHTML = `
    <div class="som-detail-grid">
      <div><span>Hücre</span><b>${escapeHtml(cell.x)}, ${escapeHtml(cell.y)}</b></div>
      <div><span>Kayıt</span><b>${escapeHtml(cell.hit_count || 0)}</b></div>
      <div><span>Seçili sonuç</span><b>${selectedResultText}</b></div>
      <div><span>Ort. kat / yükseklik</span><b>${formatNumber(means.story_count)} / ${formatNumber(means.total_height)} m</b></div>
      <div><span>Ort. kolon alanı</span><b>${formatNumber(means.column_area)} m²</b></div>
      <div><span>Ort. hedef drift</span><b>${formatPercent(means.target_drift)}</b></div>
    </div>
    ${renderSomMetricDistribution(summary, metric)}
    <div class="som-distinctive">
      <b>En ayırıcı X parametreleri</b>
      ${distinctive.length ? `
        <table class="info-table">
          <thead><tr><th>Parametre</th><th>Küme ort.</th><th>Genel ort.</th><th>Yorum</th></tr></thead>
          <tbody>${distinctive.map((item) => `
            <tr>
              <td>${escapeHtml(somFeatureLabel(item.feature))}</td>
              <td>${formatSomFeatureValue(item.feature, item.cell_mean)}</td>
              <td>${formatSomFeatureValue(item.feature, item.global_mean)}</td>
              <td>${escapeHtml(item.direction)} / skor ${formatNumber(item.score)}</td>
            </tr>
          `).join("")}</tbody>
        </table>
      ` : `<p class="section-note">Bu hücre için ayırıcı parametre hesaplanamadı.</p>`}
    </div>
    ${renderSomXSummaryTable(cell)}
    <table class="info-table">
      <thead><tr><th>Model</th><th>Yön</th><th>Uzaklık</th></tr></thead>
      <tbody>${models.map((model) => `<tr><td>${escapeHtml(model.name)}</td><td>${escapeHtml(model.direction)}</td><td>${formatNumber(model.distance)}</td></tr>`).join("")}</tbody>
    </table>
  `;
}

function renderSomXSummaryTable(cell) {
  const summaries = cell?.x_summaries || {};
  const selected = latestSom?.selected_x_columns || Object.keys(summaries);
  const rows = selected
    .filter((key) => summaries[key])
    .map((key) => {
      const summary = summaries[key];
      const kind = summary.kind || latestSom?.x_columns?.[key]?.kind || "";
      const value = kind === "nominal"
        ? `${displayCategory(summary.dominant)} (${summary.dominant_count || 0}/${summary.count || 0})`
        : formatSomFeatureValue(key, summary.avg);
      const spread = kind === "nominal"
        ? Object.entries(summary.counts || {}).map(([label, count]) => `${displayCategory(label)}: ${count}`).join(", ")
        : `min ${formatSomFeatureValue(key, summary.min)} / maks ${formatSomFeatureValue(key, summary.max)} / std ${formatNumber(summary.std)}`;
      return `
        <tr>
          <td>${escapeHtml(somFeatureLabel(key))}</td>
          <td>${escapeHtml(kind)}</td>
          <td>${escapeHtml(value)}</td>
          <td>${escapeHtml(spread)}</td>
        </tr>
      `;
    })
    .join("");
  return `
    <div class="som-distinctive">
      <b>Tüm seçili X parametreleri</b>
      ${rows ? `
        <table class="info-table">
          <thead><tr><th>Parametre</th><th>Tip</th><th>Küme değeri</th><th>Dağılım / aralık</th></tr></thead>
          <tbody>${rows}</tbody>
        </table>
      ` : `<p class="section-note">Bu hücre için X özeti yok.</p>`}
    </div>
  `;
}

function renderSomMetricDistribution(summary, metric) {
  if (!summary || Number(summary.count || 0) <= 0) {
    return `<p class="section-note">Bu hücrede seçili sonuç için veri yok. SOM'u bu yeni sonuç metriğiyle yeniden eğitmek gerekebilir.</p>`;
  }
  if (metric.type === "category") {
    const counts = summary.counts || {};
    const rows = Object.entries(counts)
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .map(([key, value]) => `<tr><td>${escapeHtml(displayCategory(key))}</td><td>${escapeHtml(value)}</td></tr>`)
      .join("");
    return `
      <div class="som-distinctive">
        <b>Seçili Y dağılımı</b>
        ${rows ? `<table class="info-table"><thead><tr><th>Sınıf</th><th>Kayıt</th></tr></thead><tbody>${rows}</tbody></table>` : `<p class="section-note">Bu hücrede dağılım yok.</p>`}
      </div>
    `;
  }
  return `
    <div class="som-distinctive">
      <b>Seçili Y istatistikleri</b>
      <table class="info-table">
        <thead><tr><th>Ortalama</th><th>Min</th><th>Maks</th><th>Std</th></tr></thead>
        <tbody><tr>
          <td>${formatMetricValue(summary.avg, metric)}</td>
          <td>${formatMetricValue(summary.min, metric)}</td>
          <td>${formatMetricValue(summary.max, metric)}</td>
          <td>${formatNumber(summary.std)}</td>
        </tr></tbody>
      </table>
    </div>
  `;
}

function somMetricSummary(cell, metricKey) {
  const summaries = cell?.result_summaries || {};
  if (summaries[metricKey]) return summaries[metricKey];
  return {
    avg: cell?.result_avg ?? null,
    min: cell?.result_min ?? null,
    max: cell?.result_max ?? null,
    std: cell?.result_std ?? null,
    dominant: cell?.dominant_category ?? null,
    dominant_count: cell?.dominant_count ?? 0,
    counts: cell?.category_counts ?? {},
  };
}

function formatSomFeatureValue(feature, value) {
  if (somPercentFeatures.has(feature)) return formatSomPercent(value);
  if (feature === "soil_class") return formatOrdinalClass(value, ["ZA", "ZB", "ZC", "ZD", "ZE"]);
  if (feature === "concrete_class") return formatConcreteClassValue(value);
  if (feature === "steel_class") return formatSteelClassValue(value);
  if (["column_area", "beam_area", "wall_area"].includes(feature)) return `${formatNumber(value)} m²`;
  if (["avg_span", "avg_span_x", "avg_span_y", "max_span_x", "max_span_y", "story_height", "total_height", "column_width", "column_depth", "beam_width", "beam_depth", "slab_thickness", "raft_thickness", "wall_thickness", "wall_length"].includes(feature)) return `${formatNumber(value)} m`;
  if (feature === "subgrade_modulus") return `${formatNumber(value)} kN/m³`;
  if (feature === "concrete_fck" || feature === "steel_fy") return `${formatNumber(value)} MPa`;
  return formatNumber(value);
}

function somFeatureLabel(feature) {
  return latestSom?.x_columns?.[feature]?.label || somFeatureLabels[feature] || feature;
}

function formatOrdinalClass(value, labels) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const min = 1;
  const max = labels.length;
  const clamped = Math.max(min, Math.min(max, number));
  const lowerIndex = Math.max(0, Math.min(labels.length - 1, Math.floor(clamped) - 1));
  const upperIndex = Math.max(0, Math.min(labels.length - 1, Math.ceil(clamped) - 1));
  if (lowerIndex === upperIndex || Math.abs(clamped - Math.round(clamped)) < 1e-9) {
    return labels[Math.round(clamped) - 1] || "-";
  }
  const lower = labels[lowerIndex];
  const upper = labels[upperIndex];
  const lowerValue = lowerIndex + 1;
  const upperValue = upperIndex + 1;
  const closer = Math.abs(clamped - lowerValue) <= Math.abs(upperValue - clamped) ? lower : upper;
  return `${lower}-${upper} arası (${closer}'ye yakın)`;
}

function formatConcreteClassValue(value) {
  const classes = [25, 30, 35, 40];
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const closest = classes.reduce((best, item) => Math.abs(item - number) < Math.abs(best - number) ? item : best, classes[0]);
  if (Math.abs(number - closest) < 1e-9) return `C${closest}`;
  const lower = [...classes].reverse().find((item) => item <= number) ?? classes[0];
  const upper = classes.find((item) => item >= number) ?? classes[classes.length - 1];
  if (lower === upper) return `C${closest} civarı`;
  const nearer = Math.abs(number - lower) <= Math.abs(upper - number) ? lower : upper;
  return `C${lower}-C${upper} arası (C${nearer}'e yakın)`;
}

function formatSteelClassValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  if (Math.abs(number - 420) < 1e-9) return "B420C";
  if (Math.abs(number - 500) < 1e-9) return "B500C";
  const nearer = Math.abs(number - 420) <= Math.abs(500 - number) ? "B420C" : "B500C";
  return `B420C-B500C arası (${nearer}'ye yakın)`;
}

function numericColor(value, min, max, light, dark) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "#f8fafc";
  const ratio = Math.max(0, Math.min(1, (number - min) / Math.max(max - min, 1e-9)));
  return mixHex(light, dark, ratio);
}

function categoryColor(value) {
  const colors = ["#e0f2fe", "#dcfce7", "#fee2e2", "#fef3c7", "#dbeafe", "#ede9fe", "#ccfbf1", "#fce7f3"];
  const text = String(value || "-");
  let hash = 0;
  for (let index = 0; index < text.length; index += 1) hash = (hash * 31 + text.charCodeAt(index)) >>> 0;
  return colors[hash % colors.length];
}

function mixHex(left, right, ratio) {
  const a = hexToRgb(left);
  const b = hexToRgb(right);
  const mix = a.map((value, index) => Math.round(value + (b[index] - value) * ratio));
  return `rgb(${mix[0]}, ${mix[1]}, ${mix[2]})`;
}

function hexToRgb(hex) {
  const clean = String(hex).replace("#", "");
  return [0, 2, 4].map((index) => parseInt(clean.slice(index, index + 2), 16));
}

function renderCharts(models = latestModelResults) {
  if (!chartCanvas || !chartSummary) return;
  const rows = chartRowsFromModels(models || []);
  if (!rows.length) {
    chartSummary.innerHTML = "";
    chartCanvas.innerHTML = `<div class="muted-box">Grafik için henüz model sonucu yok.</div>`;
    return;
  }
  const xMetric = metricByKey(chartMetrics.x, chartXMetric?.value) || chartMetrics.x[0];
  const yMetric = metricByKey(chartMetrics.y, chartYMetric?.value) || chartMetrics.y[0];
  const colorMetric = metricByKey(chartMetrics.color, chartColorMetric?.value) || chartMetrics.color[0];
  const hideMissing = chartHideMissing?.checked !== false;
  const cleanRows = rows.filter((row) => {
    if (!metricValueAvailable(row, yMetric) || !metricValueAvailable(row, xMetric)) return false;
    if (!hideMissing) return true;
    return !rowHasMissingChartResult(row, yMetric, colorMetric);
  });
  if (!cleanRows.length) {
    chartSummary.innerHTML = "";
    chartCanvas.innerHTML = `<div class="muted-box">Seçilen Y sonucu için kullanılabilir veri yok.</div>`;
    return;
  }
  chartSummary.innerHTML = renderChartSummary(cleanRows, yMetric);
  const type = chartType?.value || "scatter";
  if (type === "bar") {
    chartCanvas.innerHTML = groupedBarSvg(cleanRows, xMetric, yMetric, colorMetric);
  } else if (type === "heatmap") {
    chartCanvas.innerHTML = heatmapSvg(cleanRows, xMetric, yMetric, colorMetric);
  } else {
    chartCanvas.innerHTML = scatterSvg(cleanRows, xMetric, yMetric, colorMetric);
  }
}

function chartRowsFromModels(models) {
  const rows = [];
  models.forEach((model) => {
    const parsed = parseModelName(model.name || "");
    ["X", "Y"].forEach((direction) => {
      const curve = model.curves?.[direction] || {};
      const summary = model.hinge_summary?.[direction] || {};
      const fema = model.fema440?.[direction] || {};
      const femaEquivalent = fema.equivalent_linearization || {};
      const femaDisplacement = fema.displacement_modification || {};
      const driftSummary = storyDriftForDirection(model.story_drifts, direction);
      const critical = Array.isArray(summary.critical_events) ? summary.critical_events[0] || {} : {};
      const first = summary.first_plastic_hinge || {};
      const counts = summary.state_counts || {};
      const femaTargetCapacityRatio = safeRatio(
        Number(femaDisplacement.target_displacement_proxy_m),
        Number(curve.final_control_displacement_m)
      );
      rows.push({
        model_name: model.name,
        direction,
        story_count: Number(model.story_count ?? parsed.story_count),
        x_bay_count: Number(model.x_bay_count ?? parsed.x_bay_count),
        y_bay_count: Number(model.y_bay_count ?? parsed.y_bay_count),
        story_height: Number(model.story_height || 0),
        total_height: Number(model.story_count || parsed.story_count || 0) * Number(model.story_height || 0),
        target_drift: Number(model.target_drift_ratio || parsed.target_drift || 0),
        avg_span_x: average(model.spans_x),
        avg_span_y: average(model.spans_y),
        concrete_class: parsed.concrete_class || "-",
        soil_class: parsed.soil_class || "-",
        column_area: parsed.column_width * parsed.column_depth || null,
        beam_depth: parsed.beam_depth || null,
        rho_col: parsed.rho_col,
        rho_beam_top: parsed.rho_beam_top,
        rho_beam_bottom: parsed.rho_beam_bottom,
        rho_slab: parsed.rho_slab,
        rho_raft: Number(model.raft_rebar_ratio ?? parsed.rho_raft ?? NaN),
        slab_thickness: Number(model.slab_thickness_m || parsed.slab_thickness || 0),
        peak_base_shear: Number(curve.peak_base_shear_kn ?? 0),
        final_displacement: Number(curve.final_control_displacement_m ?? 0),
        max_rotation: maxCriticalRotation(summary),
        max_story_drift_ratio: Number(driftSummary?.max_drift?.drift_ratio ?? NaN),
        critical_drift_story: Number(driftSummary?.max_drift?.story ?? NaN),
        critical_state_rank: hingeStateRank(critical.hinge_state_level),
        critical_state: critical.hinge_state_level || "-",
        critical_element_type: critical.element_type || "-",
        first_hinge_type: first.element_type || "-",
        first_hinge_story: Number(elementStory(first.element_name) ?? NaN),
        first_hinge_plan_zone: elementPlanZone(first.element_name, model.spans_x, model.spans_y),
        critical_element_story: Number(elementStory(critical.element_name) ?? NaN),
        critical_element_plan_zone: elementPlanZone(critical.element_name, model.spans_x, model.spans_y),
        lscp_count: Number(counts["LS-CP"] || 0),
        cpc_count: Number(counts["CP-C"] || 0),
        first_plastic_step: Number(first.step_number ?? NaN),
        first_ls_step: Number(summary.first_ls_level?.step_number ?? NaN),
        first_cp_step: Number(summary.first_cp_level?.step_number ?? NaN),
        fema_ductility_mu: Number(femaEquivalent.ductility_mu ?? NaN),
        fema_beta_eff_percent: Number(femaEquivalent.effective_damping_beta_percent ?? NaN),
        fema_teff_t0_ratio: Number(femaEquivalent.effective_period_ratio_teff_t0 ?? NaN),
        fema_r_proxy: Number(femaDisplacement.r_capacity_proxy ?? NaN),
        fema_c1: Number(femaDisplacement.c1 ?? NaN),
        fema_target_displacement_proxy: Number(femaDisplacement.target_displacement_proxy_m ?? NaN),
        fema_target_capacity_ratio: femaTargetCapacityRatio,
        fema_capacity_status: femaCapacityStatus(femaTargetCapacityRatio),
      });
    });
  });
  return rows;
}

function storyDriftForDirection(storyDrifts, direction) {
  const key = String(direction || "").toUpperCase();
  if (!storyDrifts || typeof storyDrifts !== "object") return null;
  return storyDrifts.by_direction?.[key] || storyDrifts.by_direction?.[direction] || null;
}

function parseModelName(name) {
  const text = String(name || "");
  const col = /Col(\d+)x(\d+)/i.exec(text);
  const beam = /Beam(\d+)x(\d+)/i.exec(text);
  const story = /_(\d+)Story_/i.exec(text);
  const bays = /_X(\d+)Bay_Y(\d+)Bay_/i.exec(text);
  const concrete = /_(C\d{2})_/i.exec(text);
  const soil = /_Soil([A-Z]{2})/i.exec(text);
  const push = /_PushD(\d+)/i.exec(text);
  const slab = /_Slab(\d+)/i.exec(text);
  const rhoCol = /_RhoC(\d+)/i.exec(text);
  const rhoBeamTop = /_RhoBT(\d+)/i.exec(text);
  const rhoBeamBottom = /_RhoBB(\d+)/i.exec(text);
  const rhoRaft = /_RhoR(\d+)/i.exec(text);
  const rhoSlab = /_RhoS(\d+)/i.exec(text);
  return {
    story_count: story ? Number(story[1]) : null,
    x_bay_count: bays ? Number(bays[1]) : null,
    y_bay_count: bays ? Number(bays[2]) : null,
    concrete_class: concrete ? concrete[1].toUpperCase() : "",
    soil_class: soil ? soil[1].toUpperCase() : "",
    target_drift: push ? Number(push[1]) / 1000 : null,
    slab_thickness: slab ? Number(slab[1]) / 100 : null,
    column_width: col ? Number(col[1]) / 100 : 0,
    column_depth: col ? Number(col[2]) / 100 : 0,
    beam_depth: beam ? Number(beam[2]) / 100 : null,
    rho_col: rhoCol ? Number(rhoCol[1]) / 1000 : null,
    rho_beam_top: rhoBeamTop ? Number(rhoBeamTop[1]) / 1000 : null,
    rho_beam_bottom: rhoBeamBottom ? Number(rhoBeamBottom[1]) / 1000 : null,
    rho_raft: rhoRaft ? Number(rhoRaft[1]) / 10000 : null,
    rho_slab: rhoSlab ? Number(rhoSlab[1]) / 10000 : null,
  };
}

function metricByKey(metrics, key) {
  return metrics.find((metric) => metric.key === key);
}

function average(values) {
  const numbers = Array.isArray(values) ? values.map(Number).filter(Number.isFinite) : [];
  return numbers.length ? numbers.reduce((sum, value) => sum + value, 0) / numbers.length : null;
}

function maxCriticalRotation(summary) {
  const events = Array.isArray(summary.critical_events) ? summary.critical_events : [];
  return events.reduce((max, event) => Math.max(max, Number(event?.plastic_rotation_rad || 0)), 0);
}

function renderChartSummary(rows, yMetric) {
  if (yMetric.type === "category") {
    const counts = new Map();
    rows.forEach((row) => {
      const key = String(row[yMetric.key] ?? "-");
      counts.set(key, (counts.get(key) || 0) + 1);
    });
    const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0] || ["-", 0];
    return `
      <div><span>Satır</span><b>${rows.length}</b></div>
      <div><span>Kategori sayısı</span><b>${counts.size}</b></div>
      <div><span>En sık</span><b>${escapeHtml(top[0])} / ${top[1]}</b></div>
    `;
  }
  const values = rows.map((row) => Number(row[yMetric.key])).filter(Number.isFinite);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const avg = values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
  return `
    <div><span>Satır</span><b>${rows.length}</b></div>
    <div><span>${escapeHtml(yMetric.label)} ort.</span><b>${formatMetricValue(avg, yMetric)}</b></div>
    <div><span>Min / Maks</span><b>${formatMetricValue(min, yMetric)} / ${formatMetricValue(max, yMetric)}</b></div>
  `;
}

function scatterSvg(rows, xMetric, yMetric, colorMetric) {
  const dims = chartDimensions();
  const xScale = buildScale(rows, xMetric, dims.left, dims.width - dims.right);
  const yScale = buildScale(rows, yMetric, dims.height - dims.bottom, dims.top);
  const palette = chartPalette(rows, colorMetric);
  const points = rows.map((row) => {
    const x = xScale.value(row[xMetric.key]);
    const y = yScale.value(row[yMetric.key]);
    const color = palette.color(row[colorMetric.key]);
    return `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="5.5" fill="${color}"><title>${escapeHtml(row.model_name)} / ${row.direction}: ${escapeHtml(xMetric.label)}=${formatMetricValue(row[xMetric.key], xMetric)}, ${escapeHtml(yMetric.label)}=${formatMetricValue(row[yMetric.key], yMetric)}</title></circle>`;
  }).join("");
  return chartSvgShell(dims, xMetric, yMetric, xScale, yScale, points, palette);
}

function groupedBarSvg(rows, xMetric, yMetric, colorMetric) {
  const grouped = aggregateRows(rows, xMetric, yMetric);
  const dims = chartDimensions();
  const xScale = buildCategoryScale(grouped.map((row) => row.key), dims.left, dims.width - dims.right);
  const yScale = buildNumericScale(grouped.map((row) => row.value), dims.height - dims.bottom, dims.top, yMetric);
  const bars = grouped.map((row) => {
    const x = xScale.value(row.key) - xScale.bandwidth * 0.36;
    const y = yScale.value(row.value);
    const h = dims.height - dims.bottom - y;
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(xScale.bandwidth * 0.72).toFixed(1)}" height="${Math.max(1, h).toFixed(1)}" fill="#176b87"><title>${escapeHtml(row.key)}: ${formatMetricValue(row.value, yMetric)} (${row.count} kayıt)</title></rect>`;
  }).join("");
  const title = yMetric.type === "category" ? "Grup kayıt sayısı" : "Grup ortalaması";
  return chartSvgShell(dims, xMetric, { ...yMetric, label: yMetric.type === "category" ? "Kayıt sayısı" : yMetric.label }, xScale, yScale, bars, null, title);
}

function heatmapSvg(rows, xMetric, yMetric, colorMetric) {
  const xCats = categoriesForMetric(rows, xMetric);
  const yCats = categoriesForMetric(rows, colorMetric);
  const cellMap = new Map();
  rows.forEach((row) => {
    const x = categoryValue(row, xMetric);
    const y = categoryValue(row, colorMetric);
    const key = `${x}|||${y}`;
    const current = cellMap.get(key) || [];
    current.push(yMetric.type === "category" ? 1 : Number(row[yMetric.key]));
    cellMap.set(key, current);
  });
  const values = [...cellMap.values()].map((items) => items.reduce((sum, value) => sum + value, 0) / items.length);
  const min = Math.min(...values, 0);
  const max = Math.max(...values, 1);
  const width = 760;
  const height = Math.max(260, 92 + yCats.length * 42);
  const left = 130;
  const top = 42;
  const cellW = Math.max(42, (width - left - 30) / Math.max(xCats.length, 1));
  const cellH = 34;
  const cells = [];
  yCats.forEach((yCat, yi) => {
    xCats.forEach((xCat, xi) => {
      const items = cellMap.get(`${xCat}|||${yCat}`) || [];
      const avg = items.length ? items.reduce((sum, value) => sum + value, 0) / items.length : null;
      const fill = avg === null ? "#edf1f5" : heatColor((avg - min) / Math.max(max - min, 1e-9));
      cells.push(`<rect x="${left + xi * cellW}" y="${top + yi * cellH}" width="${cellW - 3}" height="${cellH - 3}" rx="4" fill="${fill}"><title>${escapeHtml(xCat)} / ${escapeHtml(yCat)}: ${avg === null ? "veri yok" : formatMetricValue(avg, yMetric)}</title></rect>`);
      if (avg !== null) cells.push(`<text x="${left + xi * cellW + cellW / 2}" y="${top + yi * cellH + 22}" text-anchor="middle">${formatMetricValue(avg, yMetric)}</text>`);
    });
    cells.push(`<text x="${left - 10}" y="${top + yi * cellH + 22}" text-anchor="end" class="axis-label">${escapeHtml(yCat)}</text>`);
  });
  xCats.forEach((xCat, xi) => {
    cells.push(`<text x="${left + xi * cellW + cellW / 2}" y="${top + yCats.length * cellH + 24}" text-anchor="middle" class="axis-label">${escapeHtml(xCat)}</text>`);
  });
  const title = yMetric.type === "category" ? `${yMetric.label} kayıt yoğunluğu` : `${yMetric.label} ortalaması`;
  return `
    <svg class="analysis-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Heatmap">
      <text x="16" y="24" class="chart-title">${escapeHtml(title)}</text>
      ${cells.join("")}
      <text x="${left}" y="${height - 10}" class="axis-label">${escapeHtml(xMetric.label)}</text>
      <text x="16" y="42" class="axis-label">${escapeHtml(colorMetric.label)}</text>
    </svg>
  `;
}

function chartDimensions() {
  return { width: 880, height: 420, left: 98, right: 178, top: 64, bottom: 72 };
}

function chartSvgShell(dims, xMetric, yMetric, xScale, yScale, body, palette, title = "") {
  const plotRight = dims.width - dims.right;
  const plotBottom = dims.height - dims.bottom;
  const plotMidY = (dims.top + plotBottom) / 2;
  const xTicks = xScale.ticks().map((tick) => `<text x="${tick.x}" y="${dims.height - 34}" text-anchor="middle" class="axis-label">${escapeHtml(tick.label)}</text>`).join("");
  const yTicks = yScale.ticks().map((tick) => `
    <line x1="${dims.left}" y1="${tick.y}" x2="${plotRight}" y2="${tick.y}" class="grid-line"></line>
    <text x="${dims.left - 14}" y="${tick.y + 4}" text-anchor="end" class="axis-label">${escapeHtml(tick.label)}</text>
  `).join("");
  const legend = palette ? palette.legend(plotRight + 26, dims.top + 10) : "";
  return `
    <svg class="analysis-chart" viewBox="0 0 ${dims.width} ${dims.height}" role="img" aria-label="Grafik">
      <text x="${dims.left}" y="28" class="chart-title">${escapeHtml(title || `${xMetric.label} vs ${yMetric.label}`)}</text>
      ${yTicks}
      <line x1="${dims.left}" y1="${plotBottom}" x2="${plotRight}" y2="${plotBottom}" class="axis-line"></line>
      <line x1="${dims.left}" y1="${dims.top}" x2="${dims.left}" y2="${plotBottom}" class="axis-line"></line>
      ${body}
      ${xTicks}
      <text x="${(dims.left + plotRight) / 2}" y="${dims.height - 12}" text-anchor="middle" class="axis-label axis-name">${escapeHtml(xMetric.label)}</text>
      <text x="22" y="${plotMidY}" text-anchor="middle" transform="rotate(-90 22 ${plotMidY})" class="axis-label axis-name">${escapeHtml(yMetric.label)}</text>
      ${legend}
    </svg>
  `;
}

function buildScale(rows, metric, start, end) {
  return metric.type === "category"
    ? buildCategoryScale(rows.map((row) => categoryValue(row, metric)), start, end)
    : buildNumericScale(rows.map((row) => Number(row[metric.key])), start, end, metric);
}

function buildNumericScale(values, start, end, metric = null) {
  const clean = values.filter(Number.isFinite);
  let min = Math.min(...clean);
  let max = Math.max(...clean);
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    min = 0; max = 1;
  }
  if (Math.abs(max - min) < 1e-9) {
    min -= 1; max += 1;
  } else {
    const padding = (max - min) * 0.08;
    min -= padding;
    max += padding;
  }
  return {
    value(value) {
      const number = Number(value);
      return start + ((number - min) / (max - min)) * (end - start);
    },
    ticks() {
      return [0, 0.25, 0.5, 0.75, 1].map((ratio) => {
        const value = min + ratio * (max - min);
        return { x: start + ratio * (end - start), y: start + ratio * (end - start), label: metric ? formatMetricValue(value, metric) : formatNumber(value) };
      });
    },
  };
}

function buildCategoryScale(values, start, end) {
  const cats = [...new Set(values.map((value) => displayCategory(value)))].slice(0, 12);
  const step = (end - start) / Math.max(cats.length, 1);
  return {
    bandwidth: step,
    value(value) {
      const index = Math.max(0, cats.indexOf(displayCategory(value)));
      return start + step * index + step / 2;
    },
    ticks() {
      return cats.map((label, index) => {
        const position = start + step * index + step / 2;
        return { x: position, y: position, label };
      });
    },
  };
}

function chartPalette(rows, metric) {
  const colors = ["#176b87", "#2d7d46", "#dc2626", "#f59e0b", "#2563eb", "#7c3aed", "#0f766e", "#be185d"];
  const cats = [...new Set(rows.map((row) => displayCategory(row[metric.key])))].slice(0, colors.length);
  return {
    color(value) {
      const index = Math.max(0, cats.indexOf(displayCategory(value)));
      return colors[index % colors.length];
    },
    legend(x, y) {
      return cats.map((cat, index) => `
        <circle cx="${x}" cy="${y + index * 20}" r="5" fill="${colors[index % colors.length]}"></circle>
        <text x="${x + 12}" y="${y + index * 20 + 4}" class="axis-label">${escapeHtml(cat)}</text>
      `).join("");
    },
  };
}

function aggregateRows(rows, xMetric, yMetric) {
  const groups = new Map();
  rows.forEach((row) => {
    const key = categoryValue(row, xMetric);
    const value = yMetric.type === "category" ? 1 : Number(row[yMetric.key]);
    if (!Number.isFinite(value)) return;
    const list = groups.get(key) || [];
    list.push(value);
    groups.set(key, list);
  });
  return [...groups.entries()].map(([key, values]) => ({
    key,
    value: values.reduce((sum, value) => sum + value, 0) / values.length,
    count: values.length,
  })).slice(0, 14);
}

function metricValueAvailable(row, metric) {
  const value = row[metric.key];
  if (metric.type === "category") return value !== null && value !== undefined && String(value) !== "";
  return Number.isFinite(Number(value));
}

function rowHasMissingChartResult(row, yMetric, colorMetric) {
  const missingMarkers = new Set(["", "-", "yok", "veri yok", "ulaşmadı", "ulasmadi", "null", "undefined"]);
  const categoricalKeys = ["critical_element_type", "first_hinge_type", "critical_state"];
  if (Number(row.critical_state_rank) < 0) return true;
  if (["first_plastic_step", "first_ls_step", "first_cp_step"].includes(yMetric.key) && !Number.isFinite(Number(row[yMetric.key]))) return true;
  for (const key of categoricalKeys) {
    const value = String(row[key] ?? "").trim().toLowerCase();
    if (missingMarkers.has(value)) return true;
  }
  for (const metric of [yMetric, colorMetric]) {
    if (metric?.type === "category") {
      const value = String(row[metric.key] ?? "").trim().toLowerCase();
      if (missingMarkers.has(value)) return true;
    }
  }
  return false;
}

function formatMetricValue(value, metric) {
  if (metric.type === "category") return escapeHtml(displayCategory(value));
  if (metric.format === "percent") return formatPercent(value);
  if (metric.format === "percent_value") return formatPercentValue(value);
  return formatNumber(value);
}

function categoriesForMetric(rows, metric) {
  if (metric.type === "category") {
    return [...new Set(rows.map((row) => displayCategory(categoryValue(row, metric))))].slice(0, 12);
  }
  const values = rows.map((row) => Number(row[metric.key])).filter(Number.isFinite);
  if (!values.length) return ["-"];
  const min = Math.min(...values);
  const max = Math.max(...values);
  const bins = 5;
  return Array.from({ length: bins }, (_, index) => {
    const a = min + ((max - min) * index) / bins;
    const b = min + ((max - min) * (index + 1)) / bins;
    return `${formatNumber(a)}-${formatNumber(b)}`;
  });
}

function categoryValue(row, metric) {
  if (metric.type === "category") return displayCategory(row[metric.key]);
  const values = latestModelResults.length ? chartRowsFromModels(latestModelResults).map((item) => Number(item[metric.key])).filter(Number.isFinite) : [];
  const value = Number(row[metric.key]);
  if (!Number.isFinite(value) || !values.length) return "-";
  const min = Math.min(...values);
  const max = Math.max(...values);
  if (Math.abs(max - min) < 1e-9) return formatNumber(value);
  const bins = 5;
  const index = Math.min(bins - 1, Math.max(0, Math.floor(((value - min) / (max - min)) * bins)));
  const a = min + ((max - min) * index) / bins;
  const b = min + ((max - min) * (index + 1)) / bins;
  return `${formatNumber(a)}-${formatNumber(b)}`;
}

function displayCategory(value) {
  const text = String(value ?? "-").trim();
  const labels = {
    capacity_ok: "Kapasite içinde",
    near_capacity: "Sınıra yakın",
    capacity_exceeded: "Kapasite üstü",
    not_available: "Hesaplanamadı",
    edge: "Kenar",
    middle: "Orta",
  };
  return !text || text === "-" ? "Veri yok" : labels[text] || text;
}

function heatColor(ratio) {
  const t = Math.max(0, Math.min(1, ratio));
  const r = Math.round(232 - 190 * t);
  const g = Math.round(245 - 78 * t);
  const b = Math.round(248 - 91 * t);
  return `rgb(${r},${g},${b})`;
}

function friendlyResultMessage(message) {
  const text = String(message || "");
  if (text.includes("must be real number, not list")) {
    return "Eski sonuç okuma hatası. Kod düzeltildi; modeli yeniden analiz edince kapasite eğrisi okunacak.";
  }
  return text;
}

function capacitySvg(points) {
  const width = 300;
  const height = 120;
  const padLeft = 28;
  const padRight = 14;
  const padTop = 14;
  const padBottom = 24;
  const cleanPoints = points
    .map((point) => ({
      displacement: Math.abs(Number(point.control_displacement_m) || 0),
      shear: Math.abs(Number(point.base_shear_kn) || 0)
    }))
    .sort((a, b) => a.displacement - b.displacement);
  const maxX = Math.max(...cleanPoints.map((point) => point.displacement), 0.001);
  const maxY = Math.max(...cleanPoints.map((point) => point.shear), 1);
  const pairs = cleanPoints.map((point) => {
    const x = padLeft + (point.displacement / maxX) * (width - padLeft - padRight);
    const y = height - padBottom - (point.shear / maxY) * (height - padTop - padBottom);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  return `
    <svg class="capacity-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Pushover kapasite eğrisi">
      <line x1="${padLeft}" y1="${height - padBottom}" x2="${width - padRight}" y2="${height - padBottom}"></line>
      <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${height - padBottom}"></line>
      <polyline points="${pairs}"></polyline>
      <text x="${padLeft}" y="11">V</text>
      <text x="${width - 58}" y="${height - 6}">deplasman</text>
    </svg>
  `;
}

function normalizeCapacityPoints(points) {
  const clean = points
    .map((point) => ({
      ...point,
      control_displacement_m: Math.abs(Number(point.control_displacement_m) || 0),
      base_shear_kn: Math.abs(Number(point.base_shear_kn) || 0)
    }))
    .sort((a, b) => a.control_displacement_m - b.control_displacement_m);
  if (clean.length <= 2) {
    const maxDisp = Math.max(...clean.map((point) => point.control_displacement_m), 0);
    const maxShear = Math.max(...clean.map((point) => point.base_shear_kn), 0);
    if (maxDisp > 0 && maxShear > 0) {
      return [
        { step_number: 0, load_step: "Origin", control_displacement_m: 0, base_shear_kn: 0 },
        { step_number: clean[clean.length - 1]?.step_number ?? 1, load_step: "Envelope", control_displacement_m: maxDisp, base_shear_kn: maxShear }
      ];
    }
  }
  return clean;
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return Math.abs(number) >= 100 ? number.toFixed(0) : number.toFixed(3).replace(/0+$/, "").replace(/\.$/, "");
}

function formatPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${(number * 100).toFixed(2).replace(/0+$/, "").replace(/\.$/, "")}%`;
}

function formatSomPercent(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  const percent = number * 100;
  const decimals = Math.abs(percent) < 0.01 ? 4 : Math.abs(percent) < 0.1 ? 3 : 2;
  return `${percent.toFixed(decimals).replace(/0+$/, "").replace(/\.$/, "")}%`;
}

function formatPercentValue(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "-";
  return `${number.toFixed(2).replace(/0+$/, "").replace(/\.$/, "")}%`;
}

function safeRatio(numerator, denominator) {
  const num = Number(numerator);
  const den = Number(denominator);
  return Number.isFinite(num) && Number.isFinite(den) && Math.abs(den) > 1e-12 ? num / den : NaN;
}

function femaCapacityStatus(ratio) {
  const value = Number(ratio);
  if (!Number.isFinite(value)) return "not_available";
  if (value <= 0.9) return "capacity_ok";
  if (value <= 1.1) return "near_capacity";
  return "capacity_exceeded";
}

async function openSapModel(name) {
  try {
    await postJson("/api/open-model", { config: collectConfig(), name });
    logBox.textContent += `\nSAP2000 model açma isteği gönderildi: ${name}`;
    logBox.scrollTop = logBox.scrollHeight;
  } catch (error) {
    logBox.textContent += `\n${error.message}`;
    logBox.scrollTop = logBox.scrollHeight;
  }
}

function openPreviewModal(name, src) {
  previewModalTitle.textContent = name;
  previewModalImage.src = src;
  previewModal.hidden = false;
}

function svgMarkupToDataUrl(markup) {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(markup.trim())}`;
}

function closePreviewModal() {
  previewModal.hidden = true;
  previewModalImage.removeAttribute("src");
}

async function fetchJson(url) {
  const response = await fetch(url);
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!response.ok) throw new Error(data?.error || text || "İstek başarısız oldu.");
  if (data === null) throw new Error("Sunucu JSON yerine beklenmeyen bir cevap döndürdü.");
  return data;
}

async function postJson(url, payload, timeoutMs = 30000) {
  const controller = new AbortController();
  const useTimeout = Number.isFinite(timeoutMs) && timeoutMs > 0;
  const timer = useTimeout
    ? window.setTimeout(() => controller.abort(new Error("İstek zaman aşımına uğradı.")), timeoutMs)
    : null;
  let response;
  try {
    response = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
  } catch (error) {
    if (error.name === "AbortError" || String(error.message || "").toLowerCase().includes("aborted")) {
      throw new Error(`İstek zaman aşımına uğradı (${Math.round(timeoutMs / 1000)} sn). SOM için grid/iterasyon değerini düşürebilir veya tekrar deneyebilirsin.`);
    }
    throw error;
  } finally {
    if (timer) window.clearTimeout(timer);
  }
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = null;
  }
  if (!response.ok) throw new Error(data?.error || text || "İstek başarısız oldu.");
  if (data === null) {
    throw new Error("Sunucu JSON yerine HTML/boş cevap döndürdü. Dashboard server'ı yeniden başlatmak gerekebilir.");
  }
  return data;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;"
  })[char]);
}

init();


