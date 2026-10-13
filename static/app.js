/* ============================================================
   CoverGen frontend logic
   Talks to the Python gradio.Server backend over plain REST:
   - POST /api/cover  -> the response body IS the audio file
     (audio/mpeg | audio/flac | audio/wav) - never JSON. It is
     turned into a blob URL and fed straight to the player.
   - GET  /api/models -> JSON list (metadata only)
   - POST /api/models/download | /api/models/upload
   Stems are fetched from /song_output/<file> (real audio files).
   ============================================================ */

/* ---------------- element handles ---------------- */
const $ = (id) => document.getElementById(id);

const el = {
    themeToggle: $("themeToggle"), themeIcon: $("themeIcon"),
    customizeTheme: $("customizeTheme"),
    connDot: $("connDot"), connText: $("connText"),
    tabUrl: $("tabUrl"), tabUpload: $("tabUpload"),
    bodyUrl: $("bodyUrl"), bodyUpload: $("bodyUpload"),
    songInput: $("songInput"),
    dropzone: $("dropzone"), audioUpload: $("audioUpload"), dzMain: $("dzMain"), dzSub: $("dzSub"),
    voiceModel: $("voiceModel"), refreshModels: $("refreshModels"), openModelMgr: $("openModelMgr"),
    modelHint: $("modelHint"),
    pitch: $("pitch"), pitchVal: $("pitchVal"),
    outputFormat: $("outputFormat"), keepFiles: $("keepFiles"),
    f0Method: $("f0Method"), crepeHop: $("crepeHop"), hopVal: $("hopVal"),
    indexRate: $("indexRate"), indexVal: $("indexVal"),
    filterRadius: $("filterRadius"), filterVal: $("filterVal"),
    rmsMixRate: $("rmsMixRate"), rmsVal: $("rmsVal"),
    protect: $("protect"), protectVal: $("protectVal"),
    f0Min: $("f0Min"), f0Max: $("f0Max"),
    mainGain: $("mainGain"), mgVal: $("mgVal"),
    backupGain: $("backupGain"), bgVal: $("bgVal"),
    instGain: $("instGain"), igVal: $("igVal"),
    f0Autotune: $("f0Autotune"),
    generateBtn: $("generateBtn"), btnLabel: document.querySelector("#generateBtn .btn-label"),
    statusBar: $("statusBar"), statusText: $("statusText"), errorBox: $("errorBox"),
    emptyState: $("emptyState"), resultBody: $("resultBody"),
    coverPlayer: $("coverPlayer"), coverDownload: $("coverDownload"),
    stemConverted: $("stemConverted"), stemDereverb: $("stemDereverb"),
    stemBackup: $("stemBackup"), stemInstrumental: $("stemInstrumental"),
    modalBackdrop: $("modalBackdrop"), closeModal: $("closeModal"),
    modelList: $("modelList"),
    modelUrl: $("modelUrl"), modelName: $("modelName"), dlModelBtn: $("dlModelBtn"),
    zipUpload: $("zipUpload"), zipName: $("zipName"), upModelBtn: $("upModelBtn"),
    modalMsg: $("modalMsg"),
};

let uploadedFile = null;    // File object chosen by the user (upload tab)
let generating = false;

/* ============================================================
   THEME — light mode is the default. Saved preference wins.
   The user can still toggle dark mode manually.
   ============================================================ */
const THEME_KEY = "covergen-theme";

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    if (el.themeIcon) el.themeIcon.textContent = theme === "dark" ? "☀" : "☾";
    try { localStorage.setItem(THEME_KEY, theme); } catch (_) { /* private mode */ }
    // Sync the dark-mode overrides for color pickers when in dark theme
    syncDarkColorDefaults(theme);
}

function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (_) {}
    // LIGHT IS DEFAULT — only honor an explicit "dark" choice.
    if (saved !== "dark") saved = "light";
    applyTheme(saved);
}

el.themeToggle?.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    applyTheme(current === "dark" ? "light" : "dark");
});
initTheme();

/* ============================================================
   THEME CUSTOMIZER
   - All settings stored in localStorage under COVERGEN_THEME_KEY
   - Live preview: changes apply instantly, can be saved / reset
   - Export: build a standalone :root { ... } CSS snippet
   - Import: read a .css file and parse its :root block
   ============================================================ */
const THEME_DATA_KEY = "covergen-theme-data";
const CUSTOM_STYLE_ID = "covergen-custom-style";
const CUSTOM_CSS_KEY = "__custom_css__";   // marker key inside the data object

const DEFAULTS = {
    primary: "#f97316",
    primaryHover: "#ea580c",
    bg: "#f7f7f8",
    panel: "#ffffff",
    text: "#1f2937",
    border: "#e5e7eb",
    font: 'Source Sans Pro, "Source Sans Pro", system-ui, sans-serif',
    fontSize: 14,
    radius: 8,
    panelPad: 20,
    maxWidth: 1180,
    [CUSTOM_CSS_KEY]: "",
};

const PRESETS = {
    "gradio-soft": { ...DEFAULTS },
    "gradio-orange": {
        ...DEFAULTS,
        primary: "#ff7a00", primaryHover: "#e96900",
        bg: "#fff7ed", panel: "#ffffff", text: "#1c1917", border: "#fed7aa",
    },
    "ocean-blue": {
        ...DEFAULTS,
        primary: "#0ea5e9", primaryHover: "#0284c7",
        bg: "#f0f9ff", panel: "#ffffff", text: "#0c4a6e", border: "#bae6fd",
    },
    "forest-green": {
        ...DEFAULTS,
        primary: "#16a34a", primaryHover: "#15803d",
        bg: "#f0fdf4", panel: "#ffffff", text: "#14532d", border: "#bbf7d0",
    },
    "monochrome": {
        ...DEFAULTS,
        primary: "#111827", primaryHover: "#1f2937",
        bg: "#f5f5f5", panel: "#ffffff", text: "#111827", border: "#d4d4d8",
    },
    "rose-pink": {
        ...DEFAULTS,
        primary: "#e11d48", primaryHover: "#be123c",
        bg: "#fff1f2", panel: "#ffffff", text: "#4c0519", border: "#fecdd3",
    },
};

const cust = {
    backdrop: $("custBackdrop"),
    close: $("closeCust"),
    primary: $("custPrimary"), primaryText: $("custPrimaryText"),
    primaryHover: $("custPrimaryHover"), primaryHoverText: $("custPrimaryHoverText"),
    bg: $("custBg"), bgText: $("custBgText"),
    panel: $("custPanel"), panelText: $("custPanelText"),
    text: $("custText"), textText: $("custTextText"),
    border: $("custBorder"), borderText: $("custBorderText"),
    font: $("custFont"),
    fontSize: $("custFontSize"),
    radius: $("custRadius"),
    panelPad: $("custPanelPad"),
    maxWidth: $("custMaxWidth"),
    css: $("custCss"),
    presets: $("custPresets"),
    save: $("custSave"),
    apply: $("custApply"),
    export: $("custExport"),
    import: $("custImport"),
    importFile: $("custImportFile"),
    reset: $("custReset"),
    msg: $("custMsg"),
};

/* ---------- helpers ---------- */
function readCustomStyleEl() {
    let style = document.getElementById(CUSTOM_STYLE_ID);
    if (!style) {
        style = document.createElement("style");
        style.id = CUSTOM_STYLE_ID;
        document.head.appendChild(style);
    }
    return style;
}

function buildCssVars(data) {
    const v = { ...DEFAULTS, ...data };
    const isDark = document.documentElement.getAttribute("data-theme") === "dark";
    // Universal tokens (primary color, fonts, radii, spacing) apply to BOTH
    // themes — they look great either way and are the user's "global" choices.
    const universal = [
        `--c-primary: ${v.primary};`,
        `--c-primary-hover: ${v.primaryHover};`,
        `--c-primary-soft: ${hexToRgba(v.primary, isDark ? 0.16 : 0.12)};`,
        `--c-primary-contrast: ${contrastColor(v.primary)};`,
        `--font-body: ${v.font};`,
        `--font-head: ${v.font};`,
        `--fs-base: ${v.fontSize}px;`,
        `--radius: ${v.radius}px;`,
        `--radius-sm: ${Math.max(0, v.radius - 2)}px;`,
        `--radius-lg: ${Math.round(v.radius * 1.5)}px;`,
        `--maxw: ${v.maxWidth}px;`,
        `--panel-pad: ${v.panelPad}px;`,
    ];
    // Surface tokens (bg/panel/text/border) only apply to LIGHT mode. Dark
    // mode falls back to the dark theme block in styles.css so the user's
    // light surface colors never leak into a dark UI.
    const lightSurfaces = [
        `--c-bg: ${v.bg};`,
        `--c-panel: ${v.panel};`,
        `--c-text: ${v.text};`,
        `--c-border: ${v.border};`,
    ];
    return `/* universal tokens — same specificity as :root[data-theme=...] in styles.css
   so this block (loaded later) wins the cascade. Applies to both themes. */
:root[data-theme="light"], :root[data-theme="dark"] {
  ${universal.join("\n  ")}
}

/* surface colors — light theme only; dark theme uses defaults from styles.css */
:root[data-theme="light"] {
  ${lightSurfaces.join("\n  ")}
}`;
}

function applyThemeData(data) {
    const cssVars = buildCssVars(data);
    const userCss = (data && data[CUSTOM_CSS_KEY]) || "";
    readCustomStyleEl().textContent = `${cssVars}\n\n/* user custom CSS */\n${userCss}`;
}

function hexToRgba(hex, alpha) {
    if (!hex || !/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(hex)) return `rgba(0,0,0,${alpha})`;
    let h = hex.slice(1);
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function contrastColor(hex) {
    if (!hex || !/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(hex)) return "#ffffff";
    let h = hex.slice(1);
    if (h.length === 3) h = h.split("").map((c) => c + c).join("");
    const r = parseInt(h.slice(0, 2), 16);
    const g = parseInt(h.slice(2, 4), 16);
    const b = parseInt(h.slice(4, 6), 16);
    // YIQ luminance
    const yiq = (r * 299 + g * 587 + b * 114) / 1000;
    return yiq >= 140 ? "#1b1b1b" : "#ffffff";
}

function syncDarkColorDefaults(_theme) {
    /* The customizer lets the user pick light-theme colors. When they switch
       to dark mode, the same colors stay applied — only the soft/contrast
       helpers recompute. We re-issue the live style on theme toggle. */
    try {
        const data = JSON.parse(localStorage.getItem(THEME_DATA_KEY) || "null");
        if (data) applyThemeData(data);
    } catch (_) {}
}

/* ---------- form <-> data ---------- */
function gatherForm() {
    return {
        primary: cust.primary.value,
        primaryHover: cust.primaryHover.value,
        bg: cust.bg.value,
        panel: cust.panel.value,
        text: cust.text.value,
        border: cust.border.value,
        font: cust.font.value,
        fontSize: Number(cust.fontSize.value),
        radius: Number(cust.radius.value),
        panelPad: Number(cust.panelPad.value),
        maxWidth: Number(cust.maxWidth.value),
        [CUSTOM_CSS_KEY]: cust.css.value,
    };
}

function applyFormToUI(data) {
    const v = { ...DEFAULTS, ...data };
    // colors
    const setColor = (picker, text, val) => {
        picker.value = val;
        text.value = val;
    };
    setColor(cust.primary, cust.primaryText, v.primary);
    setColor(cust.primaryHover, cust.primaryHoverText, v.primaryHover);
    setColor(cust.bg, cust.bgText, v.bg);
    setColor(cust.panel, cust.panelText, v.panel);
    setColor(cust.text, cust.textText, v.text);
    setColor(cust.border, cust.borderText, v.border);
    // body style
    cust.font.value = v.font;
    cust.fontSize.value = v.fontSize;
    cust.radius.value = v.radius;
    cust.panelPad.value = v.panelPad;
    cust.maxWidth.value = v.maxWidth;
    cust.css.value = v[CUSTOM_CSS_KEY] ?? "";
}

/* ---------- persistence ---------- */
function saveThemeData(data) {
    try { localStorage.setItem(THEME_DATA_KEY, JSON.stringify(data)); } catch (_) {}
}

function loadThemeData() {
    try { return JSON.parse(localStorage.getItem(THEME_DATA_KEY) || "null"); } catch (_) { return null; }
}

function clearThemeData() {
    try { localStorage.removeItem(THEME_DATA_KEY); } catch (_) {}
    readCustomStyleEl().textContent = "";
}

/* ---------- wire customizer controls ---------- */
function bindColorPair(picker, text) {
    picker.addEventListener("input", () => {
        text.value = picker.value;
        livePreview();
    });
    text.addEventListener("change", () => {
        const v = text.value.trim();
        if (/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/.test(v)) {
            picker.value = v;
            livePreview();
        } else {
            text.value = picker.value;
        }
    });
}
bindColorPair(cust.primary, cust.primaryText);
bindColorPair(cust.primaryHover, cust.primaryHoverText);
bindColorPair(cust.bg, cust.bgText);
bindColorPair(cust.panel, cust.panelText);
bindColorPair(cust.text, cust.textText);
bindColorPair(cust.border, cust.borderText);

[cust.font, cust.fontSize, cust.radius, cust.panelPad, cust.maxWidth, cust.css].forEach((ctrl) => {
    ctrl.addEventListener("input", livePreview);
    ctrl.addEventListener("change", livePreview);
});

function livePreview() {
    applyThemeData(gatherForm());
}

/* preset chips */
cust.presets.addEventListener("click", (e) => {
    const btn = e.target.closest(".cust-preset");
    if (!btn) return;
    const preset = PRESETS[btn.dataset.preset];
    if (!preset) return;
    applyFormToUI(preset);
    livePreview();
    showCustMsg(`Loaded preset: ${btn.textContent.trim()}`, true);
});

/* action buttons */
cust.save.addEventListener("click", () => {
    const data = gatherForm();
    saveThemeData(data);
    showCustMsg("Theme saved. Re-applied automatically on next visits.", true);
});

cust.apply.addEventListener("click", () => {
    livePreview();
    showCustMsg("Preview applied (not yet saved).", true);
});

cust.reset.addEventListener("click", () => {
    clearThemeData();
    applyFormToUI(DEFAULTS);
    showCustMsg("Reset to defaults. Click Save to persist.", true);
});

cust.export.addEventListener("click", () => {
    const data = gatherForm();
    const css = buildExportCss(data);
    const blob = new Blob([css], { type: "text/css;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `covergen-theme-${Date.now()}.css`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    showCustMsg("Exported theme as CSS file.", true);
});

cust.import.addEventListener("click", () => cust.importFile.click());
cust.importFile.addEventListener("change", () => {
    const file = cust.importFile.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
        try {
            const parsed = parseImportCss(String(reader.result || ""));
            applyFormToUI(parsed);
            livePreview();
            showCustMsg(`Imported ${file.name}. Click Save to keep.`, true);
        } catch (err) {
            showCustMsg(`Import failed: ${err.message || err}`, false);
        } finally {
            cust.importFile.value = "";
        }
    };
    reader.onerror = () => showCustMsg("Could not read the file.", false);
    reader.readAsText(file);
});

function buildExportCss(data) {
    const v = { ...DEFAULTS, ...data };
    const lines = [
        `/* CoverGen custom theme — exported ${new Date().toISOString().slice(0, 19)} */`,
        `/* Universal tokens (primary color, fonts, radii, spacing) — applies to both themes. */`,
        `:root[data-theme="light"], :root[data-theme="dark"] {`,
        `  --c-primary: ${v.primary};`,
        `  --c-primary-hover: ${v.primaryHover};`,
        `  --c-primary-soft: ${hexToRgba(v.primary, 0.12)};`,
        `  --c-primary-contrast: ${contrastColor(v.primary)};`,
        `  --font-body: ${v.font};`,
        `  --font-head: ${v.font};`,
        `  --fs-base: ${v.fontSize}px;`,
        `  --radius: ${v.radius}px;`,
        `  --radius-sm: ${Math.max(0, v.radius - 2)}px;`,
        `  --radius-lg: ${Math.round(v.radius * 1.5)}px;`,
        `  --maxw: ${v.maxWidth}px;`,
        `  --panel-pad: ${v.panelPad}px;`,
        `}`,
        ``,
        `/* Surface colors — light theme only. Dark theme keeps its defaults from styles.css. */`,
        `:root[data-theme="light"] {`,
        `  --c-bg: ${v.bg};`,
        `  --c-panel: ${v.panel};`,
        `  --c-text: ${v.text};`,
        `  --c-border: ${v.border};`,
        `}`,
        ``,
        `/* Dark theme keeps the same primary accent for consistency. */`,
        `:root[data-theme="dark"] {`,
        `  --c-primary: ${v.primary};`,
        `  --c-primary-hover: ${v.primaryHover};`,
        `  --c-primary-soft: ${hexToRgba(v.primary, 0.16)};`,
        `  --c-primary-contrast: ${contrastColor(v.primary)};`,
        `}`,
        ``,
    ];
    if (v[CUSTOM_CSS_KEY] && v[CUSTOM_CSS_KEY].trim()) {
        lines.push(`/* user custom CSS overrides */`, v[CUSTOM_CSS_KEY], ``);
    }
    return lines.join("\n");
}

function parseImportCss(css) {
    const out = { ...DEFAULTS };
    // Collect ALL :root {...} blocks (universal, light, dark) and apply them
    // in order — later declarations override earlier ones, which is exactly
    // how CSS cascade works.
    const blockRe = /:root(?:\[data-theme="?([a-z]+)"?\])?\s*\{([\s\S]*?)\}/g;
    let m;
    while ((m = blockRe.exec(css)) !== null) {
        const scope = m[1] || "universal";
        const body = m[2];
        const decls = body.split(";").map((s) => s.trim()).filter(Boolean);
        for (const decl of decls) {
            const idx = decl.indexOf(":");
            if (idx === -1) continue;
            const key = decl.slice(0, idx).trim();
            const val = decl.slice(idx + 1).trim();
            switch (key) {
                case "--c-primary": out.primary = val; break;
                case "--c-primary-hover": out.primaryHover = val; break;
                case "--c-bg":
                    // Only take the light-theme bg; ignore the dark-theme default
                    if (scope === "light" || scope === "universal") out.bg = val;
                    break;
                case "--c-panel":
                    if (scope === "light" || scope === "universal") out.panel = val;
                    break;
                case "--c-text":
                    if (scope === "light" || scope === "universal") out.text = val;
                    break;
                case "--c-border":
                    if (scope === "light" || scope === "universal") out.border = val;
                    break;
                case "--font-body": out.font = val; break;
                case "--fs-base": out.fontSize = parseInt(val, 10) || 14; break;
                case "--radius": out.radius = parseInt(val, 10) || 8; break;
                case "--maxw": out.maxWidth = parseInt(val, 10) || 1180; break;
                case "--panel-pad": out.panelPad = parseInt(val, 10) || 20; break;
            }
        }
    }
    if (Object.keys(out).length === 0) {
        throw new Error("No :root { ... } block found in the CSS file.");
    }
    // grab the trailing user CSS (everything after the last :root block)
    const lastRootEnd = css.lastIndexOf("}");
    if (lastRootEnd !== -1 && lastRootEnd < css.length - 1) {
        const tail = css.slice(lastRootEnd + 1).trim();
        if (tail) out[CUSTOM_CSS_KEY] = tail;
    }
    return out;
}

function showCustMsg(text, ok) {
    cust.msg.textContent = text;
    cust.msg.classList.toggle("error", !ok);
    cust.msg.classList.toggle("ok", ok);
    cust.msg.classList.remove("hidden");
    clearTimeout(showCustMsg._t);
    showCustMsg._t = setTimeout(() => cust.msg.classList.add("hidden"), 4000);
}

/* open / close modal */
el.customizeTheme?.addEventListener("click", () => {
    cust.backdrop.classList.remove("hidden");
    // make sure the form reflects the currently applied theme (saved or default)
    const data = loadThemeData() || DEFAULTS;
    applyFormToUI(data);
    livePreview();
});
cust.close.addEventListener("click", () => cust.backdrop.classList.add("hidden"));
cust.backdrop.addEventListener("click", (e) => {
    if (e.target === cust.backdrop) cust.backdrop.classList.add("hidden");
});

/* on first load: re-apply saved theme data (so refreshes keep the user's style) */
(function bootThemeData() {
    const data = loadThemeData();
    if (data) applyThemeData(data);
})();

/* ============================================================
   Backend connection
   ============================================================ */
async function connect() {
    try {
        const res = await fetch("/api/models");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        el.connDot.classList.add("ok");
        el.connText.textContent = "Connected";
        await loadModels();
    } catch (err) {
        el.connDot.classList.add("err");
        el.connText.textContent = "Offline (demo mode)";
        // Show a friendly hint inside the inference panel instead of an error toast
        if (el.errorBox) {
            el.errorBox.textContent =
                "Backend is not reachable — UI is in demo mode. Start `python server.py` to enable generation.";
            el.errorBox.classList.remove("hidden");
        }
    }
}

/* ---------------- models ---------------- */
async function loadModels() {
    try {
        const res = await fetch("/api/models");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        const models = (data.models ?? []).filter(Boolean);
        el.voiceModel.innerHTML = "";
        if (models.length === 0) {
            el.voiceModel.innerHTML = '<option value="">— no models installed —</option>';
            el.modelHint.textContent = "No voice models found — open \u201CManage\u2026\u201D to add one.";
            el.generateBtn.disabled = true;
        } else {
            for (const m of models) {
                const opt = document.createElement("option");
                opt.value = m; opt.textContent = m;
                el.voiceModel.appendChild(opt);
            }
            el.modelHint.textContent = `${models.length} voice model(s) available.`;
            el.generateBtn.disabled = generating;
        }
        renderModelList(models);
    } catch (err) {
        showError(`Failed to load models: ${err.message ?? err}`);
    }
}

function renderModelList(models) {
    el.modelList.innerHTML = "";
    (models ?? []).forEach((m) => {
        const li = document.createElement("li");
        li.textContent = m;
        el.modelList.appendChild(li);
    });
}

el.refreshModels.addEventListener("click", () => loadModels());

/* ---------------- tabs ---------------- */
el.tabUrl.addEventListener("click", () => switchTab(true));
el.tabUpload.addEventListener("click", () => switchTab(false));

function switchTab(urlActive) {
    el.tabUrl.classList.toggle("active", urlActive);
    el.tabUpload.classList.toggle("active", !urlActive);
    el.bodyUrl.classList.toggle("hidden", !urlActive);
    el.bodyUpload.classList.toggle("hidden", urlActive);
}

/* main tabs: inference / download model */
$("tabInference")?.addEventListener("click", () => switchMainTab("inference"));
$("tabDownloadModel")?.addEventListener("click", () => switchMainTab("downloadModel"));
function switchMainTab(which) {
    $("tabInference").classList.toggle("active", which === "inference");
    $("tabDownloadModel").classList.toggle("active", which === "downloadModel");
    $("bodyInference").classList.toggle("hidden", which !== "inference");
    $("bodyDownloadModel").classList.toggle("hidden", which !== "downloadModel");
}

/* ---------------- upload / drag & drop ---------------- */
el.audioUpload.addEventListener("change", () => {
    uploadedFile = el.audioUpload.files[0] ?? null;
    if (uploadedFile) {
        el.dzMain.textContent = uploadedFile.name;
        el.dzSub.textContent = `${(uploadedFile.size / 1024 / 1024).toFixed(1)} MB — click to replace`;
    }
});

["dragover", "dragenter"].forEach((ev) =>
    el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach((ev) =>
    el.dropzone.addEventListener(ev, (e) => { e.preventDefault(); el.dropzone.classList.remove("dragover"); })
);
el.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file) {
        uploadedFile = file;
        el.dzMain.textContent = file.name;
        el.dzSub.textContent = `${(file.size / 1024 / 1024).toFixed(1)} MB — click to replace`;
    }
});

/* ---------------- slider badges ---------------- */
function bindSlider(input, badge, fmt = (v) => v) {
    input.addEventListener("input", () => { badge.textContent = fmt(input.value); });
}
bindSlider(el.pitch, el.pitchVal, (v) => (v > 0 ? `+${v}` : v));
bindSlider(el.crepeHop, el.hopVal);
bindSlider(el.indexRate, el.indexVal);
bindSlider(el.filterRadius, el.filterVal);
bindSlider(el.rmsMixRate, el.rmsVal);
bindSlider(el.protect, el.protectVal);
bindSlider(el.mainGain, el.mgVal, (v) => (v > 0 ? `+${v}` : v));
bindSlider(el.backupGain, el.bgVal, (v) => (v > 0 ? `+${v}` : v));
bindSlider(el.instGain, el.igVal, (v) => (v > 0 ? `+${v}` : v));

/* ---------------- error / status UI ---------------- */
function showError(message) {
    el.errorBox.textContent = message;
    el.errorBox.classList.remove("hidden");
}
function clearError() {
    el.errorBox.textContent = "";
    el.errorBox.classList.add("hidden");
}
function setStatus(text) {
    el.statusBar.classList.remove("hidden");
    el.statusText.textContent = text;
}
function clearStatus() {
    el.statusBar.classList.add("hidden");
    el.statusText.textContent = "";
}

/* ---------------- generate ---------------- */
el.generateBtn.addEventListener("click", generateCover);

function buildFormData() {
    const useUpload = el.tabUpload.classList.contains("active") && uploadedFile;
    const form = new FormData();
    form.append("voice_model", el.voiceModel.value);
    form.append("song_input", useUpload ? "" : el.songInput.value.trim());
    form.append("pitch", String(Number(el.pitch.value)));
    form.append("keep_files", String(el.keepFiles.checked));
    form.append("main_gain", String(Number(el.mainGain.value)));
    form.append("backup_gain", String(Number(el.backupGain.value)));
    form.append("inst_gain", String(Number(el.instGain.value)));
    form.append("index_rate", String(Number(el.indexRate.value)));
    form.append("filter_radius", String(Number(el.filterRadius.value)));
    form.append("rms_mix_rate", String(Number(el.rmsMixRate.value)));
    form.append("f0_method", el.f0Method.value);
    form.append("crepe_hop_length", String(Number(el.crepeHop.value)));
    form.append("protect", String(Number(el.protect.value)));
    form.append("output_format", el.outputFormat.value);
    form.append("f0autotune", String(el.f0Autotune.checked));
    form.append("f0_min", String(Number(el.f0Min.value)));
    form.append("f0_max", String(Number(el.f0Max.value)));
    if (useUpload) form.append("audio_file", uploadedFile);
    return form;
}

function revokeIfBlob(url) {
    if (url && url.startsWith("blob:")) URL.revokeObjectURL(url);
}

function parseStemHeader(headerValue) {
    if (!headerValue) return [];
    try {
        const parsed = JSON.parse(decodeURIComponent(headerValue));
        return Array.isArray(parsed) ? parsed : [];
    } catch (_) {
        return [];
    }
}

async function generateCover() {
    if (generating) return;

    const useUpload = el.tabUpload.classList.contains("active") && uploadedFile;
    if (!useUpload && !el.songInput.value.trim()) {
        showError("Provide a YouTube link / local path, or switch to the Upload tab.");
        return;
    }
    if (!el.voiceModel.value) {
        showError("Select a voice model first (or add one via \u201CManage\u2026\u201D).");
        return;
    }

    clearError();
    generating = true;
    el.generateBtn.disabled = true;
    el.btnLabel.textContent = "Generating…";
    setStatus("Starting pipeline…");
    el.emptyState.classList.add("hidden");

    const oldCover = el.coverPlayer.src;
    const oldStems = [el.stemConverted.src, el.stemDereverb.src, el.stemBackup.src, el.stemInstrumental.src];

    try {
        setStatus("Uploading & queuing — separation, voice conversion and mixing take a few minutes…");
        const response = await fetch("/api/cover", { method: "POST", body: buildFormData() });

        if (!response.ok) {
            let message = `Generation failed (HTTP ${response.status}).`;
            try {
                const errData = await response.json();
                if (errData?.message) message = errData.message;
            } catch (_) { /* body was not JSON */ }
            throw new Error(message);
        }

        const blob = await response.blob();
        if (!blob.type.startsWith("audio/")) {
            throw new Error(`Backend returned ${blob.type || "unknown type"} instead of audio.`);
        }
        const coverUrl = URL.createObjectURL(blob);
        const filename = decodeURIComponent(response.headers.get("X-Output-Filename") ?? "ai_cover");

        el.coverPlayer.src = coverUrl;
        el.coverDownload.href = coverUrl;
        el.coverDownload.setAttribute("download", filename);
        el.resultBody.classList.remove("hidden");

        const stems = parseStemHeader(response.headers.get("X-Stems"));
        const stemMap = [el.stemConverted, el.stemDereverb, el.stemBackup, el.stemInstrumental];
        stemMap.forEach((audioEl, i) => { audioEl.src = stems[i] ?? ""; });

        revokeIfBlob(oldCover);
        oldStems.forEach(revokeIfBlob);

        setStatus(`Done — ${filename}`);
        el.resultBody.scrollIntoView({ behavior: "smooth", block: "nearest" });
        el.coverPlayer.play().catch(() => { /* autoplay blocked until user interacts */ });
    } catch (err) {
        clearStatus();
        el.emptyState.classList.remove("hidden");
        el.resultBody.classList.add("hidden");
        showError(err.message ?? String(err));
    } finally {
        generating = false;
        el.generateBtn.disabled = !el.voiceModel.value;
        el.btnLabel.textContent = "Generate AI Cover";
        if (el.statusBar.classList.contains("hidden") === false && el.statusText.textContent.startsWith("Done")) {
            setTimeout(clearStatus, 4000);
        }
    }
}

/* ---------------- model manager modal ---------------- */
el.openModelMgr.addEventListener("click", () => {
    el.modalBackdrop.classList.remove("hidden");
    el.modalMsg.classList.add("hidden");
    loadModels();
});
el.closeModal.addEventListener("click", () => el.modalBackdrop.classList.add("hidden"));
el.modalBackdrop.addEventListener("click", (e) => {
    if (e.target === el.modalBackdrop) el.modalBackdrop.classList.add("hidden");
});

function showModalMsg(text, ok) {
    el.modalMsg.textContent = text;
    el.modalMsg.classList.toggle("error", !ok);
    el.modalMsg.classList.toggle("ok", ok);
    el.modalMsg.classList.remove("hidden");
}

el.dlModelBtn.addEventListener("click", async () => {
    const url = el.modelUrl.value.trim();
    const name = el.modelName.value.trim();
    if (!url || !name) { showModalMsg("Fill in both the download link and the model name.", false); return; }
    el.dlModelBtn.disabled = true;
    showModalMsg("Downloading… this can take a while depending on the file size.", true);
    try {
        const res = await fetch("/api/models/download", {
            method: "POST",
            headers: { "Content-Type": "application/x-www-form-urlencoded" },
            body: new URLSearchParams({ url, dir_name: name }),
        });
        const data = await res.json();
        showModalMsg(data.message ?? (data.ok ? "Model downloaded!" : "Download failed."), !!data.ok);
        if (data.ok) { el.modelUrl.value = ""; el.modelName.value = ""; }
        await loadModels();
    } catch (err) {
        showModalMsg(err.message ?? String(err), false);
    } finally {
        el.dlModelBtn.disabled = false;
    }
});

el.upModelBtn.addEventListener("click", async () => {
    const file = el.zipUpload.files[0];
    const name = el.zipName.value.trim();
    if (!file || !name) { showModalMsg("Choose a zip file and give the model a unique name.", false); return; }
    el.upModelBtn.disabled = true;
    showModalMsg("Uploading…", true);
    try {
        const res = await fetch(`/api/models/upload?dir_name=${encodeURIComponent(name)}`, {
            method: "POST",
            body: file,
        });
        const data = await res.json();
        showModalMsg(data.message ?? (data.ok ? "Model uploaded!" : "Upload failed."), !!data.ok);
        if (data.ok) { el.zipUpload.value = ""; el.zipName.value = ""; }
        await loadModels();
    } catch (err) {
        showModalMsg(err.message ?? String(err), false);
    } finally {
        el.upModelBtn.disabled = false;
    }
});

/* ---------------- boot ---------------- */
connect();
