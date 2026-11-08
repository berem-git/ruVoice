"use strict";

const STATUS = {
  ru_audio:   { label: "Русская озвучка",   hint: "В игре уже есть русская озвучка." },
  ru_audio_store: { label: "Озвучка есть в Steam", hint: "Официальная русская озвучка существует, но не скачана. Создавать её не нужно: в Steam откройте Свойства игры → Язык → Русский." },
  ru_text:    { label: "Только субтитры",   hint: "Русский текст есть, озвучки нет. Можно озвучить официальные русские субтитры — это лучший случай для качества." },
  ru_unknown: { label: "Есть русские файлы", hint: "Найдены файлы с пометкой русского языка, но непонятно — озвучка это или текст." },
  no_ru:      { label: "Нет русского",      hint: "Русского языка нет. Понадобится перевод и озвучка." },
  unknown:    { label: "Языки не найдены",  hint: "Не удалось найти языковые файлы. Возможно, игра на одном языке или хранит языки внутри архивов." },
  pending:    { label: "Анализ…",           hint: "Папка игры ещё анализируется." },
};
const SOURCES = { steam: "Steam", epic: "Epic", gog: "GOG", ubisoft: "Ubisoft", manual: "Вручную", folder: "Из папки" };

let games = [];
let dubs = {};  // озвучка RuVoice по играм: идёт ли сейчас, сколько готово, установлена ли (/api/dub-overview)
let filter = "all";
let query = "";
let openId = null;
let lastVersion = -1;

// ---------- утилиты ----------

function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

function fmtSize(bytes) {
  if (!bytes) return "—";
  const units = ["Б", "КБ", "МБ", "ГБ", "ТБ"];
  let i = 0, v = bytes;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i++; }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

async function api(method, url, body) {
  const res = await fetch(url, {
    method,
    headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Ошибка ${res.status}`);
  return data;
}

function placeholderColor(name) {
  let hash = 0;
  for (const ch of name) hash = (hash * 31 + ch.charCodeAt(0)) | 0;
  const hue = Math.abs(hash) % 360;
  return `linear-gradient(135deg, hsl(${hue} 45% 32%), hsl(${(hue + 40) % 360} 50% 18%))`;
}

function initials(name) {
  return name.replace(/[^\p{L}\p{N} ]/gu, "").split(/\s+/).filter(Boolean).slice(0, 2).map(w => w[0]).join("").toUpperCase() || "?";
}

function cover(g) {
  const box = h("div", { class: "cover" });
  const ph = h("div", { class: "ph" }, initials(g.name));
  box.style.background = placeholderColor(g.name);
  box.append(ph);
  if (g.cover) {
    const img = h("img", { src: g.cover, alt: "", loading: "lazy" });
    img.onload = () => ph.remove();
    img.onerror = () => img.remove();
    box.append(img);
  }
  return box;
}

const badge = s => h("span", { class: `badge ${s}` }, STATUS[s].label);

// ---------- список ----------

function matches(g, f) {
  if (g.hidden) return false;
  switch (f) {
    case "need": return !["ru_audio", "ru_audio_store", "pending"].includes(g.status);
    case "ru_text": return g.status === "ru_text";
    case "ru_audio": return g.status === "ru_audio" || g.status === "ru_audio_store";
    case "manual": return g.source === "manual" || g.source === "folder";
    default: return true;
  }
}

function render() {
  for (const chip of document.querySelectorAll("#filters .chip")) {
    chip.classList.toggle("active", chip.dataset.filter === filter);
    chip.querySelector("b").textContent = games.filter(g => matches(g, chip.dataset.filter)).length;
  }
  const q = query.trim().toLowerCase();
  const list = games
    .filter(g => matches(g, filter) && (!q || g.name.toLowerCase().includes(q)))
    .sort((a, b) => a.name.localeCompare(b.name, "ru"));

  const grid = document.getElementById("grid");
  grid.replaceChildren(...list.map(g => {
    const c = cover(g);
    c.append(h("span", { class: "src" }, SOURCES[g.source] || g.source));
    const meta = [g.engine || (g.analyzed ? "Движок не определён" : "…"), g.size ? fmtSize(g.size) : null]
      .filter(Boolean).join(" · ");
    return h("div", { class: "card", onclick: () => openDrawer(g.id) },
      c,
      h("div", { class: "card-body" },
        h("div", { class: "card-title", title: g.name }, g.name),
        h("div", { class: "card-meta" }, g.missing ? "⚠ Папка не найдена" : meta),
        badge(g.status),
        dubLine(dubs[g.id]) ||
        (g.analyzed && !["ru_audio", "ru_audio_store"].includes(g.status) &&
          h("div", { class: `support ${g.dub_support ? "yes" : "no"}` },
            g.dub_capability?.status === "partial" ? "🎙 доступна часть звука" : g.dub_support ? "🎙 можно озвучить найденный звук" : "🎙 нужен адаптер ресурсов"))));
  }));
  document.getElementById("empty").classList.toggle("hidden", games.some(g => !g.hidden));
  if (openId) renderDrawer();
}

async function loadGames() {
  const [list, ov] = await Promise.all([api("GET", "/api/games"), api("GET", "/api/dub-overview").catch(() => ({}))]);
  games = list;
  dubs = ov;
  render();
  watchDubs();
}

// Озвучка RuVoice на карточке игры: идёт сейчас (с прогрессом) или уже сделана/установлена
const DUB_TASK = { prepare: "Подготовка", extract: "Извлечение реплик", transcribe: "Распознавание речи", translate: "Перевод",
  speakers: "Кто говорит", samples: "Пробы голосов", generate: "Озвучка", judge: "Контролёр", enginebench: "Стенд движков", install: "Установка в игру",
  uninstall: "Удаление озвучки", export: "Экспорт", import: "Импорт", train_tts: "Обучение голоса", train_voice: "Обучение голоса" };

function dubLine(d) {
  if (!d || d.error && !d.running && !d.dubbed && !d.installed) return null;
  if (d.running || d.queued) {
    const pct = d.total ? Math.round(d.done / d.total * 100) : null;
    const what = DUB_TASK[d.task] || (d.running ? "Работаем" : "В очереди");
    const count = d.total ? ` ${d.done} из ${d.total}` : "";
    // своё сообщение задачи точнее («Режиссёр: подача реплик…», «Озвучка: 120/5207, осталось ~5 ч»)
    const msg = (d.message || "").replace(/\s*\([^)]*вариант[^)]*\)/, "").trim();
    const text = msg && d.running ? msg : `${what}${count}`;
    return h("div", { class: "dub-live", title: d.message || "" },
      h("div", { class: "dub-live-text" }, h("span", { class: "pulse" }), `${text}${pct != null ? ` · ${pct}%` : ""}`),
      pct != null ? h("div", { class: "progress mini" }, h("div", { style: `width:${pct}%` })) : "");
  }
  if (d.installed) {
    const n = d.installed_count || d.approved || d.dubbed;
    return h("div", { class: "dub-own installed" }, `✓ Озвучка RuVoice установлена${n ? ` · ${n} реплик` : ""}`);
  }
  if (d.dubbed) {
    return h("div", { class: "dub-own" }, `🎙 Озвучка RuVoice: готово ${d.dubbed}${d.to_dub ? ` из ${d.dubbed + d.to_dub}` : ""} — не установлена`);
  }
  return null;
}

// Пока где-то идёт озвучка — обновлять карточки раз в 5 секунд (только сводку, без пересканирования игр)
let dubTimer = null;
function watchDubs() {
  const active = Object.values(dubs).some(d => d && (d.running || d.queued));
  if (active && !dubTimer) {
    dubTimer = setInterval(async () => {
      dubs = await api("GET", "/api/dub-overview").catch(() => dubs);
      if (!document.getElementById("drawer").classList.contains("hidden") || document.hidden) return;
      render();
      if (!Object.values(dubs).some(d => d && (d.running || d.queued))) { clearInterval(dubTimer); dubTimer = null; }
    }, 5000);
  }
}

// ---------- карточка игры ----------

function openDrawer(id) {
  openId = id;
  renderDrawer();
  document.getElementById("drawer").classList.remove("hidden");
  document.getElementById("backdrop").classList.remove("hidden");
}

function closeDrawer() {
  openId = null;
  document.getElementById("drawer").classList.add("hidden");
  document.getElementById("backdrop").classList.add("hidden");
}

function langTable(g) {
  if (!g.languages.length) {
    return h("p", { class: "note" }, g.analyzed
      ? "Языковые файлы не найдены. Возможно, все языки упакованы в общие архивы — их разберём на этапе извлечения."
      : "Идёт анализ…");
  }
  const hasStore = g.languages.some(l => l.store_audio || l.store_text);
  const cell = v => h("td", { class: "num" }, v ? fmtSize(v) : h("span", { class: "no" }, "—"));
  return h("div", {},
    h("table", {},
      h("thead", {}, h("tr", {},
        h("th", {}, "Язык"), h("th", { class: "num" }, "Озвучка"), h("th", { class: "num" }, "Текст"),
        h("th", { class: "num" }, "Видео"), h("th", { class: "num" }, "Прочее"),
        hasStore && h("th", {}, "В Steam"), h("th", {}, ""))),
      h("tbody", {}, g.languages.map(l => h("tr", { class: l.code === "ru" ? "ru" : "" },
        h("td", {}, l.name),
        cell(l.audio), cell(l.text), cell(l.video), cell(l.other),
        hasStore && h("td", {}, l.store_audio ? h("span", { class: "yes" }, "озвучка + текст")
          : l.store_text ? "текст" : h("span", { class: "no" }, "—")),
        h("td", {}, l.code !== "ru" && l.size > 0 &&
          h("button", { class: "btn subtle small", title: "Удалить файлы этого языка", onclick: () => openLanguageDelete(g, l) }, "Удалить")))))),
    h("p", { class: "note" },
      "Размеры — то, что установлено на диске. «В Steam» — что заявлено в магазине (язык может быть не скачан: его включают в свойствах игры в Steam). ",
      "«Удалить» перемещает файлы языка в Корзину — их можно вернуть."));
}

function renderDrawer() {
  const g = games.find(x => x.id === openId);
  const drawer = document.getElementById("drawer");
  if (!g) { closeDrawer(); return; }
  const st = STATUS[g.status];
  const calloutClass = { ru_audio: "good", ru_audio_store: "good", ru_text: "warn", ru_unknown: "warn", no_ru: "bad" }[g.status] || "";
  const canDub = g.analyzed && !["ru_audio", "ru_audio_store"].includes(g.status);

  drawer.replaceChildren(
    h("button", { class: "btn icon drawer-close", onclick: closeDrawer }, "✕"),
    cover(g),
    h("div", { class: "drawer-body" },
      h("h2", {}, g.name, h("button", { class: "edit", title: "Переименовать", onclick: () => renameGame(g) }, "✎")),
      h("div", { class: "path" }, g.path),
      h("div", { class: "tags" },
        h("span", { class: "tag" }, "Источник: ", h("b", {}, SOURCES[g.source] || g.source)),
        h("span", { class: "tag" }, "Движок: ", h("b", {}, g.engine || (g.analyzed ? "не определён" : "…"))),
        g.middleware.map(m => h("span", { class: "tag" }, "Звук/видео: ", h("b", {}, m))),
        g.analyzed && h("span", { class: "tag" }, "Озвучка: ", h("b", {},
          { source: "поддерживается (Source)", loose: "поддерживается (отдельные файлы)", fromsoft: "поддерживается (FromSoftware)",
            unreal: g.dub_note && g.dub_note !== "поддерживается" ? `Unreal Engine: ${g.dub_note}` : "поддерживается (Unreal Engine)",
            wwise: "поддерживается (Wwise)", fmod: "поддерживается (FMOD)", reengine: "поддерживается (RE Engine)",
            "zip-audio": "поддерживается (ZIP с аудио)", "godot-pck": "поддерживается (открытый Godot PCK)",
            "renpy-rpa": "поддерживается (Ren’Py RPA)" }[g.dub_support]
          || "пока не поддерживается")),
        g.size != null && h("span", { class: "tag" }, "Размер: ", h("b", {}, fmtSize(g.size))),
        g.files != null && h("span", { class: "tag" }, "Файлов: ", h("b", {}, g.files.toLocaleString("ru"))),
        g.steam_language && h("span", { class: "tag" }, "Язык в Steam: ", h("b", {}, g.steam_language))),
      badge(g.status),
      h("div", { class: `callout ${calloutClass}` }, st.hint),
      g.missing && h("div", { class: "callout bad" }, "Папка игры не найдена. Возможно, игру перенесли или удалили."),
      g.truncated && h("div", { class: "callout warn" }, "В папке очень много файлов — проанализирована только часть."),
      g.analyzed && g.dub_capability && h("div", { class: `callout ${g.dub_capability.status === "available" ? "good" : "warn"}` },
        g.dub_capability.note, (g.dub_capability.blockers || []).filter(note => note !== g.dub_capability.note).map(note => h("p", {}, note))),
      (g.engine_candidates || []).length > 0 && h("details", {}, h("summary", {}, "Как определён движок"),
        g.engine_candidates.map(candidate => h("p", { class: "muted small" }, `${candidate.name}: ${candidate.evidence.join(", ")}`))),
      (g.engine || "").startsWith("Unreal") && aesKeyRow(g),
      h("div", { class: "drawer-actions" },
        g.analyzed && h("button", { class: "btn primary", onclick: () => openStudio(g.id) },
          canDub ? "🎙 Создать русскую озвучку" : "🎙 Студия озвучки"),
        h("button", { class: "btn", onclick: () => api("POST", `/api/games/${encodeURIComponent(g.id)}/open`).catch(alert) }, "📂 Открыть папку"),
        h("button", { class: "btn", onclick: () => reanalyze(g) }, "⟳ Переанализировать"),
        g.ru_installed > 0 && h("button", { class: "btn danger", title: "Вернуть игре оригинальную озвучку. Проект, переводы и обученные голоса остаются в RuVoice", onclick: () => uninstallDub(g) },
          `🗑 Удалить русскую озвучку из игры (${g.ru_installed} реплик)`),
        h("button", { class: "btn subtle", title: "Игра пропадёт из этого списка. Файлы игры и установленная озвучка не трогаются", onclick: () => removeGame(g) },
          "✕ Убрать игру из RuVoice")),
      h("div", { class: "section-title" }, "Языки озвучки и текста"),
      langTable(g),
      g.exes.length > 0 && h("div", {},
        h("div", { class: "section-title" }, "Исполняемые файлы"),
        h("div", { class: "path" }, g.exes.map(e => h("div", {}, e))))));
}

function aesKeyRow(g) {
  // Архивы многих игр на Unreal зашифрованы: ключ игры публикуют сообщества моддеров, программа его не распространяет
  const input = h("input", { type: "text", class: "aes-input", value: g.aes_key || "", placeholder: "0x… (64 hex) или base64", spellcheck: "false" });
  const save = async () => {
    try {
      await api("PATCH", `/api/games/${encodeURIComponent(g.id)}`, { aes_key: input.value.trim() });
      await loadGames(); renderDrawer();
    } catch (e) { alert(e.message); }
  };
  return h("div", { class: "aes-row" },
    h("span", { class: "muted small" }, "AES-ключ архивов (.pak/.utoc): "),
    input,
    h("button", { class: "btn", onclick: save }, "Сохранить"),
    g.dub_note && g.dub_note !== "поддерживается" ? h("span", { class: "muted small" }, `— ${g.dub_note}`) : "");
}

async function renameGame(g) {
  const name = prompt("Новое название игры:", g.name);
  if (!name || name.trim() === g.name) return;
  await api("PATCH", `/api/games/${encodeURIComponent(g.id)}`, { name }).catch(e => alert(e.message));
  loadGames();
}

async function uninstallDub(g) {
  if (!confirm(`Удалить русскую озвучку из «${g.name}» и вернуть оригинал? Проект и обученные голоса останутся.`)) return;
  try { await api("POST", `/api/projects/${encodeURIComponent(g.id)}/run`, { steps: ["uninstall"] }); }
  catch (e) { alert(e.message); return; }
  alert("Удаление запущено. Через несколько секунд игра вернётся к оригинальной озвучке.");
  setTimeout(loadGames, 4000);
}

async function reanalyze(g) {
  await api("POST", `/api/games/${encodeURIComponent(g.id)}/analyze`);
  pollStatus();
}

async function removeGame(g) {
  const manual = g.source === "manual" || g.source === "folder";
  const text = manual
    ? `Убрать «${g.name}» из списка RuVoice?\n\nФайлы игры и установленная озвучка не трогаются. Чтобы вернуть игре оригинальный звук, сначала нажмите «Удалить русскую озвучку из игры».`
    : `Убрать «${g.name}» из списка RuVoice?\n\nФайлы игры и установленная озвучка не трогаются; вернуть игру в список можно через меню ⋯. Чтобы вернуть игре оригинальный звук, сначала нажмите «Удалить русскую озвучку из игры».`;
  if (!confirm(text)) return;
  await api("DELETE", `/api/games/${encodeURIComponent(g.id)}`);
  closeDrawer();
  loadGames();
}

// ---------- добавление вручную ----------

function openAdd(tab = "single") {
  document.getElementById("modal").classList.remove("hidden");
  switchTab(tab);
  for (const id of ["add-error", "dir-error"]) document.getElementById(id).textContent = "";
  loadDirs();
  setTimeout(() => document.getElementById(tab === "single" ? "add-path" : "dir-path").focus(), 50);
}

function closeAdd() { document.getElementById("modal").classList.add("hidden"); }

function switchTab(tab) {
  for (const t of document.querySelectorAll(".tab")) t.classList.toggle("active", t.dataset.tab === tab);
  for (const b of document.querySelectorAll(".tab-body")) b.classList.toggle("hidden", b.dataset.body !== tab);
}

async function pickFolder(inputId) {
  try {
    const { path } = await api("POST", "/api/pick-folder");
    if (path) document.getElementById(inputId).value = path;
  } catch {
    alert("Окно выбора папки доступно только в приложении. Вставьте путь к папке вручную.");
  }
}

async function submitGame() {
  const err = document.getElementById("add-error");
  const path = document.getElementById("add-path").value.trim();
  const name = document.getElementById("add-name").value.trim();
  if (!path) { err.textContent = "Укажите папку игры"; return; }
  try {
    const g = await api("POST", "/api/games", { path, name: name || null });
    document.getElementById("add-path").value = "";
    document.getElementById("add-name").value = "";
    closeAdd();
    await loadGames();
    openDrawer(g.id);
    pollStatus();
  } catch (e) { err.textContent = e.message; }
}

async function loadDirs() {
  const dirs = await api("GET", "/api/library-dirs");
  const ul = document.getElementById("dir-list");
  ul.replaceChildren(...(dirs.length ? dirs.map(d => h("li", {},
    h("span", {}, d),
    h("button", { class: "btn danger", onclick: async () => {
      if (!confirm(`Убрать папку ${d}? Игры из неё пропадут из списка (файлы не удаляются).`)) return;
      await api("DELETE", "/api/library-dirs", { path: d });
      loadDirs(); loadGames();
    } }, "Убрать"))) : [h("li", {}, h("span", { class: "muted" }, "Пока нет"))]));
}

async function submitDir() {
  const err = document.getElementById("dir-error");
  const path = document.getElementById("dir-path").value.trim();
  if (!path) { err.textContent = "Укажите папку"; return; }
  try {
    await api("POST", "/api/library-dirs", { path });
    document.getElementById("dir-path").value = "";
    err.textContent = "";
    loadDirs();
    pollStatus();
  } catch (e) { err.textContent = e.message; }
}

// ---------- фоновые задачи ----------

let polling = false;
async function pollStatus() {
  if (polling) return;
  polling = true;
  try {
    while (true) {
      const s = await api("GET", "/api/status");
      const box = document.getElementById("status");
      box.classList.toggle("hidden", !s.busy);
      document.getElementById("status-text").textContent =
        s.message + (s.pending > 0 ? ` (в очереди: ${s.pending})` : "");
      if (s.version !== lastVersion) { lastVersion = s.version; await loadGames(); }
      if (!s.busy) break;
      await new Promise(r => setTimeout(r, 800));
    }
  } finally { polling = false; }
}

// ---------- события ----------

document.getElementById("search").addEventListener("input", e => { query = e.target.value; render(); });
document.getElementById("filters").addEventListener("click", e => {
  const chip = e.target.closest(".chip");
  if (chip) { filter = chip.dataset.filter; render(); }
});
document.getElementById("btn-add").onclick = () => openAdd();
const importBtn = document.getElementById("btn-import");
if (importBtn) importBtn.onclick = () => importPackage(null);
document.getElementById("btn-scan").onclick = async () => { await api("POST", "/api/scan"); pollStatus(); };
document.getElementById("btn-menu").onclick = e => { e.stopPropagation(); document.getElementById("menu").classList.toggle("hidden"); };
document.addEventListener("click", () => document.getElementById("menu").classList.add("hidden"));
document.getElementById("menu").addEventListener("click", async e => {
  const act = e.target.dataset.act;
  if (act === "rescan") { await api("POST", "/api/scan?force=true"); pollStatus(); }
  if (act === "unhide") { await api("POST", "/api/games/unhide"); loadGames(); }
  if (act === "dirs") openAdd("folder");
  if (act === "system") openSystem();
});
document.getElementById("backdrop").onclick = closeDrawer;
document.querySelector("[data-close]").onclick = closeAdd;
document.getElementById("modal").addEventListener("click", e => { if (e.target.id === "modal") closeAdd(); });
for (const t of document.querySelectorAll(".tab")) t.onclick = () => switchTab(t.dataset.tab);
for (const b of document.querySelectorAll("[data-pick]")) b.onclick = () => pickFolder(b.dataset.pick);
document.getElementById("add-submit").onclick = submitGame;
document.getElementById("dir-submit").onclick = submitDir;
document.getElementById("add-path").addEventListener("keydown", e => { if (e.key === "Enter") submitGame(); });
document.getElementById("add-name").addEventListener("keydown", e => { if (e.key === "Enter") submitGame(); });
document.getElementById("dir-path").addEventListener("keydown", e => { if (e.key === "Enter") submitDir(); });
document.addEventListener("keydown", e => {
  if (e.key !== "Escape") return;
  closeAdd(); closeDrawer();
  for (const id of ["system-modal", "lang-modal"]) document.getElementById(id).classList.add("hidden");
});

loadGames().then(pollStatus);
// loadSystem() вызывается из system.js — он подключается после этого файла
