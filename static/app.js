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

/* ---------------- theme (light / dark) ---------------- */
const THEME_KEY = "covergen-theme";

function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    if (el.themeIcon) el.themeIcon.textContent = theme === "dark" ? "☀" : "☾";
    try { localStorage.setItem(THEME_KEY, theme); } catch (_) { /* private mode */ }
}

function initTheme() {
    let saved = null;
    try { saved = localStorage.getItem(THEME_KEY); } catch (_) {}
    if (saved !== "light" && saved !== "dark") {
        // No stored preference: follow the OS setting (defaults to light).
        const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
        saved = prefersDark ? "dark" : "light";
    }
    applyTheme(saved);
}

el.themeToggle?.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    applyTheme(current === "dark" ? "light" : "dark");
});
initTheme();

/* ---------------- connection ---------------- */
async function connect() {
    try {
        const res = await fetch("/api/models");
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        el.connDot.classList.add("ok");
        el.connText.textContent = "Connected";
        await loadModels();
    } catch (err) {
        el.connDot.classList.add("err");
        el.connText.textContent = "Connection failed";
        showError(`Could not connect to the backend: ${err.message}`);
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
    // The X-Stems header carries a URL-encoded JSON array of stem URLs.
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

    // Release the previous blob URLs once new audio replaces them.
    const oldCover = el.coverPlayer.src;
    const oldStems = [el.stemConverted.src, el.stemDereverb.src, el.stemBackup.src, el.stemInstrumental.src];

    try {
        setStatus("Uploading & queuing — separation, voice conversion and mixing take a few minutes…");
        const response = await fetch("/api/cover", { method: "POST", body: buildFormData() });

        if (!response.ok) {
            // Error responses are small JSON messages with a human-readable reason.
            let message = `Generation failed (HTTP ${response.status}).`;
            try {
                const errData = await response.json();
                if (errData?.message) message = errData.message;
            } catch (_) { /* body was not JSON */ }
            throw new Error(message);
        }

        // === THE RESPONSE IS THE AUDIO ITSELF ===
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

        // Stems are real audio files served from /song_output/.
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
            body: file,               // raw zip bytes as the request body
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
