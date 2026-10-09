const STORAGE_KEY = "axiom-academy-attempts";
const SETS = [
  { id: "precalc-conics", title: "Precalculus: Conic sections" },
  { id: "calc3-vectors", title: "Calculus III: Vectors and planes" },
];

const state = {
  set: null,
  index: 0,
  started: Date.now(),
  draft: "",
  pyodide: null,
  checking: false,
};

function studentName() {
  return localStorage.getItem("axiom-academy-name") || "";
}

function attempts() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]");
  } catch {
    return [];
  }
}

function saveAttempt(record) {
  const all = attempts();
  all.push(record);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(all));
}

function renderMath(root) {
  if (!window.renderMathInElement || !root) return;
  window.renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
  });
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function setIdFromUrl() {
  return new URLSearchParams(location.search).get("set");
}

function exportDocument(setId) {
  const rows = attempts().filter((item) => !setId || item.set_id === setId);
  if (setId) {
    return {
      schema: "axiom-attempts/v1",
      set_id: setId,
      student_name: studentName(),
      attempts: rows.map(publicAttempt),
    };
  }
  const bySet = new Map();
  for (const row of rows) {
    if (!bySet.has(row.set_id)) bySet.set(row.set_id, []);
    bySet.get(row.set_id).push(publicAttempt(row));
  }
  return {
    schema: "axiom-attempts/v1",
    student_name: studentName(),
    exports: [...bySet.entries()].map(([id, items]) => ({ set_id: id, attempts: items })),
  };
}

function publicAttempt(row) {
  return {
    set_id: row.set_id,
    problem_id: row.problem_id,
    answer: row.answer,
    hint_level: row.hint_level,
    time_spent_ms: row.time_spent_ms,
    created_at: row.created_at,
  };
}

function downloadAttempts(setId) {
  const payload = exportDocument(setId);
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  const label = setId || "all";
  link.download = `axiom-attempts-${label}-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  URL.revokeObjectURL(link.href);
}

function catalog() {
  const app = document.querySelector("#app");
  const saved = attempts().length;
  app.innerHTML = `
    <p class="eyebrow">Axiom Academy</p>
    <h1>Choose your assignment</h1>
    <p class="lede">Open the link Steve sent, or pick a published set. Your answer is checked in this browser. Nothing is sent until you download or submit the record.</p>
    ${SETS.map((item) => `<a class="set-link" href="?set=${item.id}"><strong>${escapeHtml(item.title)}</strong><p class="note">${item.id}</p></a>`).join("")}
    <section class="card">
      <h2>Your records</h2>
      <p class="note">${saved} attempt${saved === 1 ? "" : "s"} saved on this device.</p>
      <div class="actions">
        <button class="ghost" id="download" type="button">Download attempts</button>
      </div>
    </section>`;
  app.querySelector("#download").addEventListener("click", () => downloadAttempts());
}

function practiceMarkup() {
  const problem = state.set.problems[state.index];
  const hints = problem.hints || [];
  const revealed = Number(sessionStorage.getItem(`hint:${problem.id}`) || 0);
  return `
    <p class="eyebrow">${escapeHtml(state.set.course || "Practice")}</p>
    <h1>${escapeHtml(state.set.title)}</h1>
    <p class="note"><a href="./">All assignments</a></p>
    <div class="name-row">
      <input id="name" type="text" placeholder="Your name" value="${escapeHtml(studentName())}" aria-label="Your name" />
    </div>
    <div class="dots" role="tablist" aria-label="Problems">
      ${state.set.problems
        .map(
          (item, index) =>
            `<button type="button" data-index="${index}" aria-current="${index === state.index}" aria-label="Problem ${index + 1}">${index + 1}</button>`,
        )
        .join("")}
    </div>
    <article class="card">
      <p class="note">${escapeHtml(problem.topic)} · problem ${state.index + 1} of ${state.set.problems.length}</p>
      <h2 class="question" id="prompt"></h2>
      <label for="answer">Your answer</label>
      <textarea id="answer" placeholder="Type math, for example (x-2)^2 + (y+3)^2 = 25">${escapeHtml(state.draft)}</textarea>
      <div class="preview"><p class="note">Preview</p><div id="preview"></div></div>
      <div id="hints"></div>
      <div class="actions">
        <button class="ghost" id="hint" type="button" ${revealed >= hints.length ? "disabled" : ""}>${revealed ? `Next hint (${revealed}/${hints.length})` : "Ask for a hint"}</button>
        <button class="primary" id="check" type="button">Check answer</button>
      </div>
      <div id="feedback"></div>
    </article>
    <div class="actions">
      <button class="ghost" id="download" type="button">Download this set</button>
      <button class="ghost" id="submit" type="button">Submit record</button>
    </div>
    <p class="note" id="status">The first check downloads the math engine. After that, checking stays in this browser.</p>`;
}

function showPrompt() {
  const problem = state.set.problems[state.index];
  const prompt = document.querySelector("#prompt");
  prompt.textContent = problem.prompt;
  renderMath(prompt);
}

function showHints() {
  const problem = state.set.problems[state.index];
  const revealed = Number(sessionStorage.getItem(`hint:${problem.id}`) || 0);
  const box = document.querySelector("#hints");
  box.innerHTML = (problem.hints || [])
    .slice(0, revealed)
    .map(
      (hint, index) =>
        `<div class="hint"><strong>Hint ${index + 1}${index === 2 ? " · worked solution" : ""}.</strong> <span class="hint-text"></span></div>`,
    )
    .join("");
  box.querySelectorAll(".hint-text").forEach((node, index) => {
    node.textContent = problem.hints[index].text;
  });
  renderMath(box);
}

function paintPreview() {
  const answer = document.querySelector("#answer");
  const preview = document.querySelector("#preview");
  if (!answer || !preview) return;
  preview.textContent = answer.value.trim() ? `$$${answer.value}$$` : "Your expression will show here.";
  renderMath(preview);
}

function renderPractice() {
  const app = document.querySelector("#app");
  app.innerHTML = practiceMarkup();
  showPrompt();
  showHints();
  paintPreview();
  const answer = app.querySelector("#answer");
  answer.addEventListener("input", () => {
    state.draft = answer.value;
    paintPreview();
  });
  app.querySelector("#name").addEventListener("change", (event) => {
    localStorage.setItem("axiom-academy-name", event.target.value.trim());
  });
  app.querySelectorAll("[data-index]").forEach((button) => {
    button.addEventListener("click", () => {
      state.index = Number(button.dataset.index);
      state.draft = "";
      state.started = Date.now();
      renderPractice();
    });
  });
  app.querySelector("#hint").addEventListener("click", () => {
    const problem = state.set.problems[state.index];
    const revealed = Number(sessionStorage.getItem(`hint:${problem.id}`) || 0);
    sessionStorage.setItem(`hint:${problem.id}`, String(Math.min((problem.hints || []).length, revealed + 1)));
    state.draft = answer.value;
    renderPractice();
  });
  app.querySelector("#check").addEventListener("click", checkAnswer);
  app.querySelector("#download").addEventListener("click", () => downloadAttempts(state.set.id));
  app.querySelector("#submit").addEventListener("click", submitRecord);
}

async function checker() {
  if (state.pyodide) return state.pyodide;
  const status = document.querySelector("#status");
  if (status) status.textContent = "Loading the math checker. The first check takes a moment.";
  if (!window.loadPyodide) {
    await new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = "https://cdn.jsdelivr.net/pyodide/v0.27.5/full/pyodide.js";
      script.onload = resolve;
      script.onerror = () => reject(new Error("Could not load the in-browser Python runtime. Check the network and try again."));
      document.head.appendChild(script);
    });
  }
  const pyodide = await window.loadPyodide();
  await pyodide.loadPackage("sympy");
  const source = await (await fetch("mathcheck.py")).text();
  pyodide.FS.writeFile("mathcheck.py", source);
  state.pyodide = pyodide;
  if (status) status.textContent = "Checker ready. It stays in this browser.";
  return pyodide;
}

async function checkAnswer() {
  if (state.checking) return;
  const problem = state.set.problems[state.index];
  const answer = document.querySelector("#answer").value;
  const feedback = document.querySelector("#feedback");
  const name = document.querySelector("#name").value.trim();
  state.draft = answer;
  if (!name) {
    feedback.className = "feedback invalid";
    feedback.textContent = "Add your name so Steve can match this record.";
    return;
  }
  state.checking = true;
  document.querySelector("#check").disabled = true;
  try {
    const pyodide = await checker();
    pyodide.globals.set(
      "axiom_payload",
      JSON.stringify({
        expected: problem.answer,
        given: answer,
        tolerance: problem.tolerance ?? 0.001,
        kind: problem.answer_kind || "auto",
      }),
    );
    const raw = pyodide.runPython(`
import json
from mathcheck import check_answer
payload = json.loads(axiom_payload)
result = check_answer(
    payload["expected"],
    payload["given"],
    tolerance=float(payload.get("tolerance") or 0.001),
    kind=payload.get("kind") or "auto",
)
json.dumps(result.as_dict())
`);
    const parsed = JSON.parse(String(raw));
    const record = {
      schema: "axiom-attempt/v1",
      set_id: state.set.id,
      problem_id: problem.id,
      student_name: name,
      answer: answer.trim(),
      is_correct: parsed.status === "correct",
      hint_level: Number(sessionStorage.getItem(`hint:${problem.id}`) || 0),
      time_spent_ms: Math.max(1000, Date.now() - state.started),
      error_category: parsed.error_category,
      feedback: parsed.message,
      created_at: new Date().toISOString(),
    };
    saveAttempt(record);
    localStorage.setItem("axiom-academy-name", name);
    feedback.className = `feedback ${parsed.status}`;
    feedback.textContent = parsed.message;
    state.started = Date.now();
  } catch (error) {
    feedback.className = "feedback invalid";
    feedback.textContent = error.message || "The checker could not run.";
  } finally {
    state.checking = false;
    const button = document.querySelector("#check");
    if (button) button.disabled = false;
  }
}

async function submitRecord() {
  const status = document.querySelector("#status");
  try {
    const result = await window.AxiomSync.submit(exportDocument(state.set ? state.set.id : null));
    status.textContent = result.synced ? "Sent." : result.reason;
  } catch (error) {
    status.textContent = error.message;
  }
}

async function boot() {
  const id = setIdFromUrl();
  if (!id) {
    catalog();
    return;
  }
  try {
    const response = await fetch(`sets/${encodeURIComponent(id)}.json`);
    if (!response.ok) throw new Error("That set is not published yet.");
    state.set = await response.json();
    state.index = 0;
    state.draft = "";
    state.started = Date.now();
    renderPractice();
  } catch (error) {
    document.querySelector("#app").innerHTML = `<h1>Set unavailable</h1><p class="lede">${escapeHtml(error.message)}</p><p><a href="./">Back to assignments</a></p>`;
  }
}

document.addEventListener("DOMContentLoaded", boot);
