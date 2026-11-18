"use strict";
const form = document.querySelector("#controls");
const status = document.querySelector("#status");
const error = document.querySelector("#error");
const renderButton = document.querySelector("#render");
const stopButton = document.querySelector("#stop");
const labels = {x:"X", y:"Y", roughness:"roughness", albedo:"albedo", light:"light", absorption:"σₐ", scattering:"σₛ"};
const sceneLabels = {cornell:"Cornell", bunny:"Bunny", volume:"Backlit fog", fog_cube:"Homogeneous fog cube"};
let defaults, busy = false, snapshot = null, activeId = null, stopped = false;

function settings() {
  const result = Object.fromEntries(new FormData(form));
  for (const key in result) if (!["scene", "integrator", "derivative", "method"].includes(key)) result[key] = Number(result[key]);
  return result;
}
function updateControls() {
  const finiteDifference = form.elements.method.value === "fd";
  document.querySelector("#fd-step-control").hidden = !finiteDifference;
  form.elements.fd_step.disabled = !finiteDifference;
  const volume = ["volume", "fog_cube"].includes(form.elements.scene.value);
  document.querySelector("#object-controls").hidden = volume;
  document.querySelector("#medium-controls").hidden = !volume;
  for (const select of [form.elements.integrator, form.elements.derivative]) {
    for (const option of select.options) {
      option.hidden = option.disabled = ["volume", "absorption", "scattering"].includes(option.value) !== volume;
    }
    if (select.selectedOptions[0]?.disabled) select.value = [...select.options].find(o => !o.disabled).value;
  }
  for (const input of form.querySelectorAll("input[type=range]")) {
    form.querySelector(`output[for=${input.id}]`).textContent = Number(input.value).toFixed(2);
    input.style.setProperty("--fill", `${100*(input.value-input.min)/(input.max-input.min)}%`);
  }
  if (!busy) {
    const dirty = snapshot && JSON.stringify(settings()) !== JSON.stringify(snapshot);
    document.body.classList.toggle("dirty", Boolean(dirty));
    status.textContent = dirty ? "Settings changed" : stopped ? "Stopped" : snapshot ? "Ready" : "Ready to render";
  }
}
function setControls(values) {
  for (const [key,value] of Object.entries(values)) form.elements.namedItem(key).value = value;
  updateControls();
}
async function request(url, options) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `HTTP ${response.status}`);
  return data;
}
function showPass(submitted, completed, result) {
  for (const kind of ["render", "gradient"]) {
    const img = document.querySelector(`#${kind}-image`);
    img.src = result[kind]; img.hidden = false;
    const save = document.querySelector(`#save-${kind}`);
    save.href = result[kind]; save.hidden = false;
    const suffix = kind === "gradient" ? `-${submitted.method || "ad"}` : "";
    save.download = `${submitted.scene}-${kind}-${submitted.derivative}${suffix}.png`;
  }
  snapshot = submitted;
  const colorbar = document.querySelector("#colorbar");
  colorbar.src = result.colorbar;
  colorbar.hidden = false;
  document.querySelector("#gradient-title").textContent = `∂L / ∂${labels[submitted.derivative]}`;
  document.querySelector("#gradient-method").textContent = submitted.method === "fd"
    ? `FD · h=${submitted.fd_step}` : "FORWARD AD";
  document.querySelector("#negative").textContent = `−${result.limit.toPrecision(3)}`;
  document.querySelector("#positive").textContent = `+${result.limit.toPrecision(3)}`;
  document.querySelector("#snapshot").textContent = `${sceneLabels[submitted.scene]} · ${submitted.resolution}² · pass ${completed}/${submitted.npass} · ${completed * submitted.spp} spp · seed ${submitted.seed}`;
  document.querySelector("#timing").textContent = `${result.seconds.toFixed(2)}s`;
}
async function render(event, existingJob = null) {
  event?.preventDefault();
  if (busy || !defaults || !form.reportValidity()) return;
  busy = true;
  stopped = false;
  error.hidden = true;
  renderButton.hidden = true;
  stopButton.hidden = false;
  stopButton.disabled = true;
  document.body.classList.add("busy");
  document.body.classList.remove("dirty");
  const submitted = settings();
  status.textContent = `Rendering · 0/${submitted.npass}`;
  try {
    const job = existingJob || await request("/api/render", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify(submitted)});
    activeId = job.id;
    stopButton.disabled = false;
    let completed = 0;
    while (true) {
      const data = await request(`/api/jobs/${job.id}?after=${completed}`);
      if (data.result) {
        completed = data.completed;
        showPass(submitted, completed, data.result);
        if (!stopButton.disabled) status.textContent = `Rendering · ${completed}/${submitted.npass}`;
      }
      if (data.state === "error") throw new Error(data.error);
      if (data.state === "done" || data.state === "stopped") {
        stopped = data.state === "stopped";
        break;
      }
    }
  } catch (failure) {
    error.textContent = failure.message; error.hidden = false;
  } finally {
    activeId = null; busy = false;
    stopButton.hidden = true; renderButton.hidden = false;
    document.body.classList.remove("busy"); updateControls();
    if (!error.hidden) status.textContent = "Render failed";
  }
}
stopButton.addEventListener("click", async () => {
  const id = activeId;
  if (id === null) return;
  stopButton.disabled = true;
  status.textContent = "Stopping…";
  try {
    await request(`/api/jobs/${id}/stop`, {method:"POST"});
  } catch (failure) {
    if (activeId === id) {
      stopButton.disabled = false;
      error.textContent = failure.message; error.hidden = false;
    }
  }
});
form.addEventListener("input", updateControls);
form.addEventListener("submit", render);
document.querySelector("#reset").addEventListener("click", () => setControls(defaults));
document.addEventListener("keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") render(event); });
request("/api/config").then(config => {
  defaults = config.defaults;
  // An already-running local worker may still expose the previous API.
  form.elements.method.disabled = !Object.hasOwn(defaults, "method");
  form.elements.method.parentElement.hidden = form.elements.method.disabled;
  setControls(config.active?.settings || defaults);
  if (config.active) return render(null, config.active);
  const scene = new URLSearchParams(location.search).get("scene");
  if (Object.hasOwn(sceneLabels, scene)) {
    form.elements.scene.value = scene;
    if (scene === "fog_cube") {
      form.elements.resolution.value = "256";
      form.elements.spp.value = "32";
    }
    updateControls();
  }
  return render();
}).catch(failure => {
  error.textContent = failure.message; error.hidden = false; status.textContent = "Disconnected";
});
