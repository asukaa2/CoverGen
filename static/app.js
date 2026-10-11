/* ============================================================
   CoverGen frontend logic
   Talks to the Python gradio.Server backend through the official
   Gradio JS client (@gradio/client), so every request goes through
   the backend's queueing engine (concurrency + SSE progress).
   ============================================================ */

import { Client, handle_file } from "https://cdn.jsdelivr.net/npm/@gradio/client/dist/index.min.js";

/* ---------------- element handles ---------------- */
const $ = (id) => document.getElementById(id);

const el = {
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

let client = null;          // gradio JS client
let uploadedFile = null;    // File object chosen by the user (upload tab)
let generating = false;

/* ---------------- connection ---------------- */
async function connect() {
    try {
        client = await Client.connect(window.location.origin);
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
function unwrapData(msg) {
    const data = msg?.data ?? msg ?? [];
    return Array.isArray(data[0]) ? data[0] : data;
}

async function loadModels() {
    if (!client) return;
    try {
        const result = await client.predict("/get_models", {});
        const models = unwrapData(result).filter(Boolean);
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

function buildPayload() {
    const useUpload = el.tabUpload.classList.contains("active") && uploadedFile;
    const payload = {
        voice_model: el.voiceModel.value,
        song_input: useUpload ? "" : el.songInput.value.trim(),
        pitch: Number(el.pitch.value),
        keep_files: el.keepFiles.checked,
        main_gain: Number(el.mainGain.value),
        backup_gain: Number(el.backupGain.value),
        inst_gain: Number(el.instGain.value),
        index_rate: Number(el.indexRate.value),
        filter_radius: Number(el.filterRadius.value),
        rms_mix_rate: Number(el.rmsMixRate.value),
        f0_method: el.f0Method.value,
        crepe_hop_length: Number(el.crepeHop.value),
        protect: Number(el.protect.value),
        output_format: el.outputFormat.value,
        f0autotune: el.f0Autotune.checked,
        f0_min: Number(el.f0Min.value),
        f0_max: Number(el.f0Max.value),
    };
    if (useUpload) payload.audio_file = handle_file(uploadedFile);
    return payload;
}

async function generateCover() {
    if (!client || generating) return;

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

    try {
        const job = client.submit("/generate_cover", buildPayload());

        for await (const msg of job) {
            if (msg.type === "status") {
                if (msg.stage === "error") {
                    const m = typeof msg.message === "string" ? msg.message : JSON.stringify(msg.message);
                    throw new Error(m || "Generation failed.");
                }
                if (msg.stage === "pending" && msg.position != null && msg.position >= 0) {
                    setStatus(`Queued — position ${msg.position}${msg.eta ? ` · eta ${Math.round(msg.eta)}s` : ""}`);
                } else if (msg.stage === "generating") {
                    setStatus("Processing… separating vocals, converting voice, mixing.");
                } else if (msg.stage === "complete") {
                    setStatus("Finalizing…");
                }
            } else if (msg.type === "data") {
                renderResults(msg.data);
            }
        }
        clearStatus();
    } catch (err) {
        clearStatus();
        el.emptyState.classList.remove("hidden");
        el.resultBody.classList.add("hidden");
        showError(err.message ?? String(err));
    } finally {
        generating = false;
        el.generateBtn.disabled = !el.voiceModel.value;
        el.btnLabel.textContent = "Generate AI Cover";
    }
}

function fileUrl(item) {
    return item?.url ?? (item?.path ? `/file=${item.path}` : null);
}

function renderResults(data) {
    const files = Array.isArray(data?.[0]) ? data[0] : (Array.isArray(data) ? data : []);
    if (files.length === 0) return;

    const [cover, converted, dereverb, backup, instrumental] = files;
    const coverUrl = fileUrl(cover);
    if (coverUrl) {
        el.coverPlayer.src = coverUrl;
        el.coverDownload.href = coverUrl;
        el.coverDownload.setAttribute("download", cover?.path?.split("/").pop() ?? "ai_cover");
        el.resultBody.classList.remove("hidden");
    }
    const stemMap = [
        [converted, el.stemConverted],
        [dereverb, el.stemDereverb],
        [backup, el.stemBackup],
        [instrumental, el.stemInstrumental],
    ];
    stemMap.forEach(([item, audioEl]) => {
        const url = fileUrl(item);
        audioEl.src = url ?? "";
    });
    el.resultBody.scrollIntoView({ behavior: "smooth", block: "nearest" });
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
        const result = await client.predict("/download_online_model", { url, dir_name: name });
        showModalMsg(unwrapData(result)[0] ?? "Model downloaded!", true);
        el.modelUrl.value = ""; el.modelName.value = "";
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
        const result = await client.predict("/upload_local_model", { zip_file: handle_file(file), dir_name: name });
        showModalMsg(unwrapData(result)[0] ?? "Model uploaded!", true);
        el.zipUpload.value = ""; el.zipName.value = "";
        await loadModels();
    } catch (err) {
        showModalMsg(err.message ?? String(err), false);
    } finally {
        el.upModelBtn.disabled = false;
    }
});

/* ---------------- boot ---------------- */
connect();
