let selectedJobId = null;

async function api(path, opts) {
  const res = await fetch(path, opts);
  return res.json();
}

function getFormData(form) {
  const fd = new FormData(form);
  const out = {};
  for (const [k, v] of fd.entries()) out[k] = v;
  return out;
}

async function runTask(taskId, params) {
  const payload = { task_id: taskId, params };
  const data = await api("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!data.ok) {
    alert("Run failed: " + (data.error || "unknown error"));
    return;
  }
  selectedJobId = data.job_id;
  await refreshJobs();
  await refreshSelectedJob();
}

function renderJobs(jobs) {
  const host = document.getElementById("jobsList");
  host.innerHTML = "";
  if (!jobs.length) {
    host.textContent = "No jobs yet.";
    return;
  }

  for (const j of jobs) {
    const el = document.createElement("div");
    el.className = "job" + (j.id === selectedJobId ? " active" : "");
    el.innerHTML = `
      <div><strong>${j.task_id}</strong> <span class="status ${j.status}">${j.status}</span></div>
      <div class="meta">id=${j.id} pid=${j.pid || "-"} rc=${j.returncode ?? "-"}</div>
    `;
    el.onclick = async () => {
      selectedJobId = j.id;
      await refreshJobs();
      await refreshSelectedJob();
    };
    host.appendChild(el);
  }
}

async function refreshJobs() {
  const jobs = await api("/api/jobs");
  renderJobs(jobs);
}

async function refreshSelectedJob() {
  const stopBtn = document.getElementById("stopBtn");
  const log = document.getElementById("logOutput");

  if (!selectedJobId) {
    stopBtn.disabled = true;
    log.textContent = "Select a job to view logs.";
    return;
  }

  const j = await api(`/api/jobs/${selectedJobId}`);
  if (!j || j.error) {
    stopBtn.disabled = true;
    log.textContent = "Job not found.";
    return;
  }

  stopBtn.disabled = !(j.status === "running");
  const header = `[${j.task_id}] status=${j.status} rc=${j.returncode ?? "-"} pid=${j.pid ?? "-"}\n`;
  log.textContent = header + "\n" + (j.logs || []).join("\n");
  log.scrollTop = log.scrollHeight;
}

async function stopSelectedJob() {
  if (!selectedJobId) return;
  await api(`/api/jobs/${selectedJobId}/stop`, { method: "POST" });
  await refreshJobs();
  await refreshSelectedJob();
}

function bindForms() {
  document.querySelectorAll(".card").forEach((card) => {
    const taskId = card.dataset.taskId;
    const form = card.querySelector(".task-form");
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const params = getFormData(form);
      await runTask(taskId, params);
    });
  });
}

function bindButtons() {
  document.getElementById("refreshBtn").onclick = async () => {
    await refreshJobs();
    await refreshSelectedJob();
  };
  document.getElementById("stopBtn").onclick = stopSelectedJob;
}

async function init() {
  bindForms();
  bindButtons();
  await refreshJobs();
  setInterval(async () => {
    await refreshJobs();
    await refreshSelectedJob();
  }, 1500);
}

init();

