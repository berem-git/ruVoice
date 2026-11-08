"use strict";
// Студия озвучки — один линейный поток:
//   1. Подготовка (само): реплики → русский текст → обучение голосов главных персонажей → пробы
//   2. Послушайте персонажей: нравится → озвучить всю игру; не нравится → попробовать иначе / начать заново
//   3. Озвучка всей игры → прослушать и одобрить
//   4. Установка в игру (и удаление)

const LINE_STATUS = {
  new: "Нет перевода", translated: "Ждёт озвучки", generated: "Озвучено — прослушайте",
  approved: "Одобрено", rejected: "Отклонено", skipped: "Пропущено", error: "Ошибка",
};
const STEP_NAMES = {
  extract: "Извлечение реплик", transcribe: "Распознавание речи", translate: "Перевод",
  generate: "Озвучка", judge: "Контролёр качества", train_voice: "Обучение тембра", engineexport: "Экспорт озвучки", import: "Импорт озвучки", speakers: "Кто говорит", redirect: "Переозвучка по указанию", direct: "Режиссёр", events_pass: "Смех и вздохи из оригинала", screen: "Отсев неречевых реплик",
  prepare: "Подготовка", samples: "Пробы голосов", install: "Установка в игру", uninstall: "Удаление озвучки",
  restore_originals: "Возврат оригиналов",
};
const QUALITY = { fast: "Быстро — 1 вариант на реплику", normal: "Обычно — 2 варианта, контролёр выбирает лучший",
  thorough: "Тщательно — 4 варианта (в 3–4 раза дольше)" };

const studio = { gid: null, info: null, view: "flow", timer: null, filters: { status: "", char: "", q: "", offset: 0 } };
const PAGE = 50;

// ---------- общий проигрыватель ----------

const player = new Audio();
let playingBtn = null;
function playUrl(btn, url) {
  if (playingBtn === btn && !player.paused) { player.pause(); return; }
  if (playingBtn) playingBtn.classList.remove("playing");
  player.src = url;
  player.play().catch(() => {});
  playingBtn = btn;
  btn.classList.add("playing");
}
player.onended = player.onpause = () => { if (playingBtn) playingBtn.classList.remove("playing"); };
const projUrl = path => `/api/projects/${encodeURIComponent(studio.gid)}${path}`;
const playBtn = (kind, id, label) =>
  h("button", { class: `btn play ${kind}`, title: kind === "gen" ? "Русская озвучка" : "Оригинал",
    onclick: e => playUrl(e.currentTarget, projUrl(`/audio?kind=${kind}&id=${encodeURIComponent(id)}&t=${Date.now()}`)) }, "▶ ", label);
const fileBtn = (file, label) =>
  h("button", { class: "btn play gen", onclick: e => playUrl(e.currentTarget, projUrl(`/audio?kind=file&file=${encodeURIComponent(file)}&t=${Date.now()}`)) }, "▶ ", label);

// ---------- открытие / обновление ----------

async function openStudio(gid) {
  if (studio.gid !== gid) { studio.step = null; studio.tool = null; }
  studio.gid = gid;
  studio.view = "flow";
  studio.filters = { status: "", char: "", q: "", offset: 0 };
  closeDrawer();
  document.getElementById("studio").classList.remove("hidden");
  await refreshStudio();
  clearInterval(studio.timer);
  studio.timer = setInterval(pollStudio, 1500);
}

function closeStudio() {
  clearInterval(studio.timer);
  player.pause();
  document.getElementById("studio").classList.add("hidden");
  studio.gid = null;
  loadGames();
}

async function refreshStudio() {
  studio.info = await api("GET", projUrl(""));
  renderStudio();
}

async function pollStudio() {
  if (!studio.gid) return;
  const info = await api("GET", projUrl("")).catch(() => null);
  if (!info) return;
  const st = info.status, old = studio.info?.status;
  const finished = old && (old.running || old.queued.length) && !st.running && !st.queued.length;
  const changed = studio.view === "lines" ? finished
    : !old || st.running !== old.running || st.task !== old.task || st.queued.length !== old.queued.length || st.paused !== old.paused
      || st.message !== old.message || JSON.stringify(info.stats) !== JSON.stringify(studio.info.stats)
      || JSON.stringify(info.samples?.at) !== JSON.stringify(studio.info.samples?.at)
      || info.installed !== studio.info.installed || info.prepared !== studio.info.prepared
      || JSON.stringify(info.review) !== JSON.stringify(studio.info.review)
      || JSON.stringify(info.previews) !== JSON.stringify(studio.info.previews);
  studio.info = info;
  if (finished) { voicesCache.rows = null; studio.step = null; }  // задача закончилась — раскрыть актуальный шаг
  renderJobBar();
  // Не перерисовываем, только пока пользователь печатает в текстовом поле (галочки и списки не в счёт)
  const typing = document.activeElement?.matches("textarea, input[type=text], input[type=search], input:not([type])");
  if (changed && !typing) renderStudioBody();
}

// Раскрытые панели и прокрутка переживают перерисовку
function renderStudioBodyPreserving(render, body) {
  const panelKey = d => d.dataset.panel || d.querySelector("summary")?.textContent;
  const panels = new Map(Array.from(body.querySelectorAll("details"), d => [panelKey(d), d.open]));
  const y = body.scrollTop;
  const result = render(body);
  const restore = () => {
    for (const d of body.querySelectorAll("details")) {
      const key = panelKey(d);
      if (panels.has(key)) d.open = panels.get(key);
    }
    body.scrollTop = y;
  };
  if (result && typeof result.then === "function") result.then(restore); else restore();
}

// ---------- каркас ----------

function renderStudio() {
  const g = games.find(x => x.id === studio.gid) || { name: studio.info?.game_name || "" };
  const root = document.getElementById("studio");
  root.replaceChildren(
    h("header", { class: "studio-head" },
      h("button", { class: "btn", onclick: () => studio.view === "lines" ? (studio.view = "flow", renderStudio()) : closeStudio() },
        studio.view === "lines" ? "← Назад" : "← Библиотека"),
      h("div", { class: "studio-title" },
        h("div", { class: "name" }, g.name),
        h("div", { class: "muted small" }, studio.info.adapter ? `Способ: ${studio.info.adapter.name}` : "Для этой игры пока нет способа")),
      powerControl(),
      h("div", { id: "jobbar", class: "jobbar" })),
    h("nav", { id: "stepper", class: "stepper" }),
    h("div", { id: "studio-body", class: "studio-body" }));
  renderJobBar();
  renderStudioBody();
}

// Чем считаем — решает программа; пользователю только показываем
function accelChip() {
  const a = window.systemInfo?.accel;
  if (!a?.variant) return "";
  const gpu = a.device === "cuda";
  return h("span", { class: `chip accel ${gpu ? "ok" : "warn"}`, title: gpu ? `${a.device_name}, ${a.vram_gb} ГБ видеопамяти · ${a.variant_name}` : "Видеокарта для нейросетей не найдена — озвучка недоступна" },
    gpu ? `⚡ ${a.device_name}` : "✖ Видеокарта не найдена");
}

function powerControl() {
  const p = window.systemInfo?.power;
  if (!p) return "";
  return h("label", { class: "power", title: "Сколько ресурсов компьютера отдавать озвучке. Можно менять в любой момент — подействует со следующей пачки работы." },
    accelChip(), " ",
    "Нагрузка на систему ",
    h("select", { onchange: async e => { await api("POST", "/api/system/power", { variant: e.target.value }); await loadSystem(); } },
      Object.entries(p.levels).map(([k, v]) => h("option", { value: k, selected: p.level === k, title: v.desc }, v.name))));
}

// Задачи озвучки: остановка удаляет то, что озвучено в этом запуске (ранее готовое остаётся)
const DUB_TASKS = ["generate", "judge", "redirect", "samples", "events_pass"];
function jobButtons(st) {
  const dubbing = DUB_TASKS.includes(st.task);
  const pauseBtn = st.paused
    ? h("button", { class: "btn", title: "Продолжить с места остановки", onclick: () => api("POST", projUrl("/resume")).then(pollStudio) }, "▶ Продолжить")
    : h("button", { class: "btn", title: "Замереть на ближайшей реплике; модели остаются в памяти", onclick: () => api("POST", projUrl("/pause")).then(pollStudio) }, "⏸ Пауза");
  const stopBtn = dubbing
    ? h("button", { class: "btn danger", title: "Остановить и вернуть реплики этого запуска в «ждут озвучки»; озвученное раньше и одобренное остаётся",
        onclick: () => confirm("Остановить озвучку и удалить реплики, озвученные в этом запуске? Озвученное раньше и одобренное останется.") && api("POST", projUrl("/stop-delete")).then(pollStudio) }, "■ Остановить и удалить озвучку")
    : h("button", { class: "btn danger", onclick: () => api("POST", projUrl("/cancel")).then(pollStudio) }, "■ Стоп");
  return [pauseBtn, stopBtn];
}

function renderJobBar() {
  const bar = document.getElementById("jobbar");
  if (!bar || !studio.info) return;
  const st = studio.info.status;
  const busy = Boolean(st.running || st.queued.length);
  const pct = st.total ? Math.round(st.done / st.total * 100) : null;
  bar.replaceChildren(
    busy ? h("div", { class: `job ${st.paused ? "paused" : ""}` },
      st.paused ? h("span", { class: "pause-mark" }, "⏸") : h("span", { class: "spinner" }),
      h("div", { class: "job-text" },
        h("div", {}, (STEP_NAMES[st.task] || "В очереди") + (st.paused ? " — пауза" : "")),
        h("div", { class: "muted small" }, st.message)),
      pct != null && h("div", { class: "progress" }, h("div", { style: `width:${pct}%` })),
      ...(st.running ? jobButtons(st) : [h("button", { class: "btn danger", onclick: () => api("POST", projUrl("/cancel")).then(pollStudio) }, "■ Убрать из очереди")]))
    : st.error ? h("div", { class: "job error" }, "⚠ ", st.error)
    : st.message ? h("div", { class: "job muted" }, "✓ ", st.message) : "");
}

function renderStudioBody() {
  const body = document.getElementById("studio-body");
  if (!body) return;
  renderStudioBodyPreserving(studio.view === "lines" ? renderLines : renderFlow, body);
}

async function run(path, body) {
  try { await api("POST", projUrl(path), body); } catch (e) { alert(e.message); }
  pollStudio();
}
const runSteps = (steps, ids) => run("/run", { steps, ids });

// ---------- поток ----------

// ---------- поток: пять шагов, раскрыт один ----------

const TOOLS = { voices: "Голоса", settings: "Настройки" };
const PREP_STAGES = ["prepare", "extract", "transcribe", "translate", "speakers", "train_voice", "samples", "restore_originals"];

// Состояние пяти шагов по сводке проекта: done / active / running / pending и строка-итог
function stepsModel(info) {
  const s = info.stats || { total: 0, by_status: {}, rework: 0, seconds: 0, with_official: 0 }, by = s.by_status || {};
  const n = k => by[k] || 0;
  const generatedAll = n("generated") + n("approved") + n("rejected");
  const busy = Boolean(info.status.running || info.status.queued.length);
  const stage = busy ? info.status.task : null;
  const left = n("translated");
  const reviewReady = Object.values(info.review || {}).filter(st => st.state === "ready").length;
  const plan = info.train_plan || { planned: [] };
  const trained = (plan.planned || []).filter(c => c.trained).length;
  const prepare = {
    n: 1, key: "prepare", title: "Подготовка",
    state: PREP_STAGES.includes(stage) ? "running" : info.prepared ? "done" : "active",
    summary: info.prepared ? `${s.total} реплик · ${(info.characters || []).length} персонажей · тембр у ${trained} из ${(plan.planned || []).length}`
      : info.extracted ? `Реплики извлечены (${s.total}), текст и пробы ещё не готовы` : "Реплики, текст, голоса и пробы",
  };
  const samples = {
    n: 2, key: "samples", title: "Пробы",
    state: !info.samples ? "pending" : generatedAll > 0 ? "done" : "active",
    summary: !info.samples ? "Появятся после подготовки" : generatedAll > 0 ? "Голоса одобрены — игра озвучивается этим способом" : `Готовы пробы ${(info.samples.chars || []).length} персонажей — послушайте`,
  };
  const dub = {
    n: 3, key: "dub", title: "Озвучка",
    state: stage === "generate" || stage === "judge" || stage === "redirect" ? "running"
      : generatedAll > 0 && !left ? "done" : generatedAll > 0 ? "active" : "pending",
    summary: generatedAll ? `Озвучено ${generatedAll} из ${s.total - n("skipped")}` + (left ? `, осталось ${left}` : "") : "Вся игра одним нажатием после проб",
  };
  const review = {
    n: 4, key: "review", title: "Проверка",
    state: (n("generated") || reviewReady) ? "active" : n("approved") && !n("generated") ? "done" : "pending",
    summary: generatedAll ? `Одобрено ${n("approved")} · на прослушивании ${n("generated")}` + (s.rework ? ` · слабых ${s.rework}` : "") : "Послушать и одобрить персонажей",
  };
  const install = {
    n: 5, key: "install", title: "Установка",
    state: stage === "install" || stage === "uninstall" ? "running" : info.installed ? "done" : n("approved") ? "active" : "pending",
    summary: info.installed ? `Установлено ${info.installed_count} реплик` : n("approved") ? `Готово к установке: ${n("approved")} одобренных` : "Нужны одобренные реплики",
  };
  const steps = [prepare, samples, dub, review, install];
  const current = steps.find(x => x.state === "running") || steps.find(x => x.state === "active") || steps[steps.length - 1];
  return { steps, current, busy, stage, s, n, generatedAll, left };
}

function renderStepper(model) {
  const nav = document.getElementById("stepper");
  if (!nav) return;
  const open = studio.step || model.current.n;
  nav.replaceChildren(
    ...model.steps.map(st => h("button", { class: `stepbtn ${st.state} ${open === st.n ? "open" : ""}`, title: st.summary,
        onclick: () => selectStep(st.n) },
      h("span", { class: "num" }, st.state === "done" ? "✓" : st.state === "running" ? h("span", { class: "spinner tiny" }) : st.n),
      h("span", { class: "t" }, st.title))),
    h("div", { class: "tools" },
      ...Object.entries(TOOLS).map(([k, name]) => h("button", { class: `btn small ${studio.tool === k ? "on" : ""}`,
        onclick: () => { studio.tool = studio.tool === k ? null : k; renderStudioBody(); } }, name)),
      h("button", { class: "btn small", onclick: openSystem }, "Система"),
      studio.info.installed && uninstallButton(model.busy)));
}

function selectStep(number) {
  studio.step = number;
  studio.tool = null;
  studio.view = "flow";
  renderStudioBody();
}

function stepRow(st) {
  return h("button", { type: "button", class: `steprow ${st.state}`, onclick: () => selectStep(st.n) },
    h("div", { class: "num" }, st.state === "done" ? "✓" : st.n),
    h("b", {}, st.title),
    h("span", { class: "sum muted" }, st.summary),
    h("span", { class: "muted small" }, "развернуть"));
}

function renderFlow(body) {
  const info = studio.info;
  if (!info.adapter) {
    const g = games.find(x => x.id === studio.gid) || {};
    const what = [g.engine, ...(g.middleware || [])].filter(Boolean).join(", ");
    document.getElementById("stepper")?.replaceChildren();
    if (g.status === "ru_audio" || g.status === "ru_audio_store") {
      body.replaceChildren(h("div", { class: "callout good" },
        h("p", {}, h("b", {}, "В этой игре уже есть официальная русская озвучка."),
          g.status === "ru_audio_store" ? " Она не скачана: включите русский язык в свойствах игры в лаунчере (Steam → Свойства → Язык), лаунчер докачает её."
            : " Включите русский язык озвучки в настройках самой игры — перевод нейросетью здесь не нужен."),
        h("p", { class: "muted" }, "RuVoice нужен для игр, у которых русской озвучки нет.")));
      return;
    }
    body.replaceChildren(h("div", { class: "callout bad" },
      h("p", {}, h("b", {}, "Эту игру пока нельзя озвучить."), " Отдельных звуковых файлов в её папке нет — звук упакован в архивы движка",
        what ? ` (${what})` : "", ", а такой формат программа пока не разбирает."),
      h("p", { class: "muted" }, "Доступны адаптеры Wwise и FMOD, отдельных звуковых файлов, обычных ZIP/PK3/PK4/IWD/LÖVE, открытых Godot PCK v1/v2 и Ren’Py RPA v2/v3, а также Unreal, RE Engine, Source и FromSoftware. Поддержка зависит от содержимого и версии конкретных архивов; определение движка само по себе не гарантирует извлечение и установку.")));
    return;
  }
  const model = stepsModel(info);
  const { s, n, generatedAll, busy, stage } = model;
  renderStepper(model);
  const open = studio.step || model.current.n;
  const cards = {
    prepare: () => stepPrepare(info, s, busy, stage),
    samples: () => info.samples ? stepSamples(info, generatedAll, busy, PREP_STAGES.includes(stage))
      : card(2, "Пробы", "Пробы голосов появятся после подготовки: по три реплики на персонажа, оригинал и русский вариант рядом.", null, "pending"),
    dub: () => stepDub(info, s, n, generatedAll, busy, stage),
    review: () => stepReview(info, s, n, generatedAll, busy),
    install: () => stepInstall(info, n("approved"), busy),
  };
  const tool = studio.tool ? { voices: () => voicesPanel(info, busy), settings: () => settingsPanel(info) }[studio.tool]() : "";
  if (tool && tool.tagName === "DETAILS") {
    tool.dataset.panel = studio.tool;
    tool.open = true;
  }
  body.replaceChildren(
    hardwareCard(info),
    adapterNotes(info.adapter),
    tool ? h("div", { class: "tool-drawer" }, tool) : "",
    ...model.steps.map(st => st.n === open ? cards[st.key]() : stepRow(st)));
}

function card(num, title, text, actions, state) {
  return h("div", { class: `step ${state === "done" ? "done" : ""} ${state === "active" ? "active" : ""}` },
    h("div", { class: "num" }, state === "done" ? "✓" : num),
    h("div", { class: "step-main" }, h("h3", {}, title),
      typeof text === "string" ? h("p", { class: "muted" }, text) : text,
      actions ? h("div", { class: "row" }, actions) : ""));
}

function stepPrepare(info, s, busy, stage) {
  const plan = info.train_plan || { planned: [], ready: false, level: "normal", skipped: [] };
  const running = ["prepare", "extract", "transcribe", "translate", "speakers", "train_voice", "samples"].includes(stage);
  if (running) {
    const st = info.status, pct = st.total ? Math.round(st.done / st.total * 100) : null;
    return card(1, "Подготовка идёт", h("div", {},
      h("p", {}, st.message || "Работаем…"),
      pct != null ? h("div", { class: "progress wide" }, h("div", { style: `width:${pct}%` })) : "",
      h("p", { class: "muted small" }, "Реплики → русский текст → кто говорит → тембр главных → пробы. Это долго; программу можно свернуть.",
        info.settings?.auto_install ? " Режим «без прослушивания»: после подготовки игра озвучится и установится сама." : ""),
      timbreLevelPicker(info, true)),
      jobButtons(info.status), "active");
  }
  if (!info.prepared) {
    const text = info.extracted
      ? `Реплики уже извлечены (${s.total}). Осталось подготовить текст, обучить тембр главных персонажей и сделать пробы.`
      : "Программа достанет реплики из игры, подготовит русский текст, разберётся, кто говорит, обучит тембр главных персонажей и покажет пробы. В игру ничего не записывается до вашего одобрения.";
    return card(1, "Подготовка", h("div", {}, h("p", { class: "muted" }, text),
      h("details", { class: "more" }, h("summary", {}, "Тембр главных персонажей и режим без прослушивания"), timbreLevelPicker(info, busy), autoInstallCheck(info))),
      [h("button", { class: "btn primary big", disabled: busy || modelsMissing(), title: modelsMissing() ? "Сначала скачайте набор нейросетей (шаг 0)" : "", onclick: () => run("/prepare") }, info.extracted ? "Продолжить подготовку" : "Начать"),
       modelsMissing() && h("span", { class: "muted small" }, "Сначала скачайте нейросети в шаге 0")], "active");
  }
  const mains = plan.planned || [];
  const trained = mains.filter(c => c.trained).length;
  const voicesNote = !plan.ready ? "Перенос тембра не установлен — все персонажи идут быстрым путём (клонирование с записи)."
    : mains.length ? `Тембр обучен у ${trained} из ${mains.length} главных персонажей; остальные — быстрым путём.` : "";
  const noSubNote = s.total - s.with_official > 0 ? ` Без русских субтитров ${s.total - s.with_official} — они останутся с оригинальным голосом.` : "";
  return card(1, "Подготовка завершена",
    h("div", {}, h("p", { class: "muted" }, `Реплик: ${s.total}, персонажей: ${info.characters.length}, минут речи: ${Math.round(s.seconds / 60)}. Официальный русский текст у ${s.with_official}.${noSubNote} ${voicesNote}`),
      timbreLevelPicker(info, busy)),
    [h("button", { class: "btn subtle small", disabled: busy, onclick: () => confirm("Подготовить заново (текст и пробы; обученные тембры останутся)?") && run("/restart") }, "Подготовить заново")], "done");
}

// Уровень обучения тембра: сколько главных персонажей, времени и места займёт каждый (оценка по этому компьютеру)
function timbreLevelPicker(info, busy) {
  const t = info.timbre;
  if (!t || !t.levels) return "";
  const cur = info.settings?.timbre_level || t.current || "normal";
  const save = async lvl => { studio.info.settings = await api("POST", projUrl("/settings"), { timbre_level: lvl }); refreshStudio(); };
  const fmtMin = m => m >= 90 ? `~${Math.round(m / 6) / 10} ч` : `~${m} мин`;
  const opt = (key, extra) => {
    const l = t.levels[key];
    if (!l) return "";
    const what = key === "off" ? "никого не обучать — все персонажи быстрым путём"
      : !l.chars ? (info.prepared ? "главных персонажей нет (меньше 2 минут речи у каждого)" : "главные персонажи появятся после подготовки")
      : `${l.chars} гл. персонажей, ${l.todo ? fmtMin(l.minutes) : "уже обучены"}, ~${l.disk_gb} ГБ на диске`;
    return h("label", { class: "radio", title: (l.names || []).join(", ") },
      h("input", { type: "radio", name: "timbre_level", checked: cur === key, disabled: busy, onchange: () => save(key) }),
      h("span", {}, h("b", {}, l.name), ` — ${what}`, extra ? h("span", { class: "muted small" }, extra) : ""));
  };
  const need = cur !== "off" && (t.levels[cur]?.todo || 0) > 0 && info.prepared && t.ready;
  return h("div", { class: "train-levels" },
    h("div", { class: "small" }, h("b", {}, "Тембр главных персонажей"),
      h("span", { class: "muted" }, t.ready ? " · перенос тембра установлен" : " · перенос тембра не установлен (⋯ → Система и ускорение → Среды движков)")),
    opt("off"), opt("fast"), opt("normal", " · рекомендуем"), opt("deep", " · если «обычно» нравится не до конца"),
    h("p", { class: "muted small" }, "Главный персонаж — от 2 минут речи и 12 реплик (уровень можно поменять вручную на панели «Голоса»). Тембр обучается на его записях и проверяется на репликах вне обучения: если голос не стал ближе к персонажу, читает хуже или ушёл по высоте — модель не используется, персонаж идёт быстрым путём. Время — оценка по этой видеокарте."),
    need ? h("button", { class: "btn small", disabled: busy, onclick: () => run("/prepare") }, `Обучить сейчас (${fmtMin(t.levels[cur].minutes)})`) : "");
}

// Галочка «на ночь»: после подготовки сразу озвучить всю игру и установить, не слушая пробы
function autoInstallCheck(info) {
  const on = !!info.settings?.auto_install;
  return h("label", { class: "radio auto-install" },
    h("input", { type: "checkbox", checked: on, onchange: async e => { studio.info.settings = await api("POST", projUrl("/settings"), { auto_install: e.target.checked }); } }),
    h("span", {}, "Сразу озвучить всю игру и установить в игру без прослушивания ",
      h("span", { class: "muted small" }, "— можно оставить на ночь; забракованные контролёром реплики всё равно будут переозвучены и установлены")));
}

function stepSamples(info, generatedAll, busy, preparing) {
  const sm = info.samples;
  const v = sm.variant || {};
  const decided = generatedAll > 0;
  const queued = (info.status.queued || []).includes("generate");
  const note = preparing ? h("p", { class: "muted small" }, "Подготовка ещё идёт: пробы появляются по мере обучения персонажей, уже готовые можно слушать. ",
    queued ? "Озвучка всей игры уже в очереди и начнётся сама после подготовки." : "Кнопки «Нравится — озвучить всю игру» и «Переделать пробы» появятся после подготовки.") : "";
  const blocks = (sm.chars || []).map(c => h("div", { class: "sample-char" },
    h("div", { class: "sample-head" }, h("b", {}, c.name),
      c.trained ? h("span", { class: "pill ok" }, "тембр обучен") : h("span", { class: "pill", title: c.reason || "" }, c.source ? `быстрый путь · ${c.source}` : "быстрый путь"),
      c.similarity != null && h("span", { class: "muted small" }, `похожесть ${c.similarity.toFixed(2)}`),
      c.error && h("span", { class: "error" }, c.error)),
    (c.items || []).map(it => h("div", { class: "sample-row" },
      h("div", { class: "sample-text" }, h("div", {}, it.ru), h("div", { class: "muted small" }, it.en)),
      playBtn("orig", it.id, "Оригинал"), fileBtn(it.file, "Русский"), speakerPicker(it.id, c.id, busy))),
    h("button", { class: "btn small", disabled: busy, onclick: () => run("/samples", { char: c.id }) }, "Обновить пробы этого голоса")));
  return card(2, "Послушайте персонажей", h("div", {},
    note,
    h("p", { class: "muted" }, "Слева оригинал, справа русский. Нравится — озвучим всю игру. Нет — переделайте пробы (другой прогон) или смените донора во вкладке «Голоса»."),
    h("div", { class: "samples" }, blocks)),
    preparing || (info.settings?.auto_install && (info.status.running || info.status.queued.length)) ? []
      : decided ? [h("span", { class: "muted small" }, "Игра уже озвучена этим способом.")]
      : [h("button", { class: "btn primary big", disabled: busy, onclick: () => runSteps(["generate"]) }, "👍 Нравится — озвучить всю игру"),
         h("button", { class: "btn", disabled: busy, title: "Другой прогон: другие случайные зёрна у тех же реплик", onclick: () => run("/samples", { next: true }) }, "🔁 Переделать пробы"),
         h("button", { class: "btn subtle", disabled: busy, onclick: () => confirm("Начать заново: текст и пробы пересоберутся, обученные голоса останутся. Продолжить?") && run("/restart") }, "↺ Начать заново")],
    decided ? "done" : "active");
}

// Персонажи, озвученные целиком: послушать, одобрить или переозвучить с комментарием — пока идут остальные
let reviewCache = { at: 0, data: null };
function reviewPanel(info, busy) {
  const ready = Object.entries(info.review || {}).filter(([, st]) => st.state === "ready");
  if (!ready.length) return "";
  const key = ready.map(([c, st]) => c + st.at).join("|");
  const box = h("div", { class: "review" },
    h("h4", {}, `Готовы целиком: ${ready.length} ${ready.length === 1 ? "персонаж" : "персонажа"} — послушайте и одобрите`),
    h("p", { class: "muted small" }, "Нравится — одобрить (пойдёт в установку). Нет — напишите, что поправить, и переозвучьте: комментарий станет указанием для нейросети."),
    h("div", { class: "review-list" }, h("span", { class: "muted small" }, "Загрузка…")));
  const fill = data => {
    const list = box.querySelector(".review-list");
    list.replaceChildren(...(data.ready || []).map(r => {
      const ta = h("textarea", { rows: 2, placeholder: "Комментарий для нейросети: например «говорить медленнее и ниже, спокойнее, без спешки» или «моложе, живее»" }, r.note || "");
      return h("div", { class: "sample-char" },
        h("div", { class: "sample-head" }, h("b", {}, r.name), h("span", { class: "muted small" }, `${r.lines} реплик`),
          r.rework ? h("span", { class: "pill warn", title: "Контролёр просит проверить на слух" }, `⚠ ${r.rework}`) : ""),
        (r.samples || []).map(it => h("div", { class: "sample-row" },
          h("div", { class: "sample-text" }, h("div", {}, it.ru), h("div", { class: "muted small" }, it.en || "")),
          playBtn("orig", it.id, "Оригинал"), playBtn("gen", it.id, "Русский"), it.score != null ? h("span", { class: "muted small" }, `${it.score}`) : "")),
        ta,
        h("div", { class: "row" },
          h("button", { class: "btn primary", disabled: busy, onclick: async () => { await api("POST", projUrl("/review/approve"), { char: r.char, note: ta.value }); reviewCache = { at: 0, data: null }; refreshStudio(); } }, "👍 Одобрить персонажа"),
          h("button", { class: "btn", onclick: async () => { if (!confirm(`Переозвучить все реплики «${r.name}» заново с учётом комментария?`)) return; await api("POST", projUrl("/review/redo"), { char: r.char, note: ta.value }); reviewCache = { at: 0, data: null }; pollStudio(); } }, "🔁 Переозвучить с комментарием"),
          h("button", { class: "btn subtle small", onclick: async () => { await api("POST", projUrl("/character-note"), { char: r.char, note: ta.value }); } }, "Сохранить комментарий")));
    }));
  };
  if (reviewCache.data && reviewCache.key === key) fill(reviewCache.data);
  else api("GET", projUrl("/review")).then(d => { reviewCache = { key, data: d }; fill(d); }).catch(() => {});
  return box;
}

function stepDub(info, s, n, generatedAll, busy, stage) {
  if (stage === "generate" || stage === "judge" || stage === "redirect") {
    const st = info.status, pct = st.total ? Math.round(st.done / st.total * 100) : null;
    return card(3, stage === "generate" ? "Озвучка идёт" : stage === "judge" ? "Контролёр проверяет озвучку" : "Переозвучка по указанию", h("div", {},
      h("p", {}, st.message),
      pct != null ? h("div", { class: "progress wide" }, h("div", { style: `width:${pct}%` })) : "",
      h("p", { class: "muted small" }, "Готовые персонажи появляются в шаге «Проверка», пока идут остальные. «Пауза» замирает на ближайшей реплике; «Остановить и удалить озвучку» возвращает реплики этого запуска в «ждут озвучки», озвученное раньше остаётся.")),
      jobButtons(info.status), "active");
  }
  const left = n("translated");
  if (!generatedAll) {
    return card(3, "Озвучка всей игры", info.samples ? "После одобрения проб вся игра озвучивается одним нажатием; готовые персонажи можно слушать по ходу."
      : "Станет доступна после подготовки и проб.",
      info.samples ? [h("button", { class: "btn primary big", disabled: busy, onclick: () => runSteps(["generate"]) }, "Озвучить всю игру")] : null,
      info.samples ? "active" : "pending");
  }
  const text = `Озвучено ${generatedAll} из ${s.total - n("skipped")} реплик.` + (left ? ` Ещё не озвучено: ${left}.` : "");
  return card(3, "Озвучка всей игры", text, [
    left ? h("button", { class: "btn primary", disabled: busy, onclick: () => runSteps(["generate"]) }, `Озвучить оставшиеся (${left})`) : "",
    h("button", { class: "btn subtle small", disabled: busy, onclick: () => selectStep(4) }, "К проверке →"),
  ], left ? "active" : "done");
}

function stepReview(info, s, n, generatedAll, busy) {
  if (!generatedAll && !Object.values(info.review || {}).some(st => st.state === "ready")) {
    return card(4, "Проверка", "После озвучки: послушать персонажей, одобрить хорошее, переозвучить слабое с комментарием.", null, "pending");
  }
  const text = (n("generated") ? `На прослушивании ${n("generated")}.` : "") +
    (s.rework ? ` Контролёр отметил слабых: ${s.rework} — их стоит послушать или переозвучить.` : " Контролёр замечаний не нашёл.") +
    (n("approved") ? ` Одобрено: ${n("approved")}.` : "");
  return card(4, "Проверка", h("div", {}, h("p", { class: "muted" }, text), reviewPanel(info, busy)), [
    h("button", { class: "btn primary", disabled: busy, onclick: () => { studio.view = "lines"; studio.filters = { status: s.rework ? "rework" : "generated", char: "", q: "", offset: 0 }; renderStudio(); } }, "Прослушать реплики"),
    h("button", { class: "btn ok", disabled: busy || !n("generated"), onclick: approveGood }, "✓ Одобрить все хорошие"),
    s.rework ? h("button", { class: "btn", disabled: busy, onclick: regenRework }, `⟳ Переозвучить слабые (${s.rework})`) : "",
  ], n("approved") && !n("generated") ? "done" : "active");
}

async function approveGood() {
  if (!confirm("Одобрить все озвученные реплики, которые контролёр оценил как хорошие (70 баллов и выше)? Остальные останутся на прослушивание.")) return;
  const r = await api("POST", projUrl("/mark"), { status: "approved", from_status: "generated", min_score: 70 });
  await refreshStudio();
  alert(`Одобрено: ${r.changed}`);
}

async function regenRework() {
  const r = await api("GET", projUrl(`/lines?rework=1&limit=5000`));
  await api("POST", projUrl("/regenerate"), { ids: r.items.filter(l => l.status !== "approved").map(l => l.id) });
  pollStudio();
}

function uninstallButton(busy) {
  return h("button", { class: "btn danger small", disabled: busy,
    onclick: () => confirm("Удалить русскую озвучку из игры и вернуть оригинал?") && runSteps(["uninstall"]) }, "Удалить озвучку");
}

function stepInstall(info, approved, busy) {
  const a = info.adapter || {};
  const method = a.adapter === "source"
    ? "Озвучка кладётся в отдельную папку игры (custom\\ruvoice). Файлы игры не меняются."
    : a.adapter === "unreal"
    ? `Озвучка собирается в отдельный pak-мод ${a.mod_path || "Content\\Paks\\~mods\\RuVoice_P.pak"}. Оригинальные архивы не меняются. После обновления игры мод может перестать подхватываться — тогда переустановите озвучку.`
    : a.adapter === "fromsoft"
    ? "Звуковые банки игры (sound\\*.fsb) пересобираются с заменёнными репликами и ставятся на место английских; оригиналы — в резервной копии, «Удалить озвучку» их возвращает. Игра запускается как обычно. Русская озвучка подменяет английскую дорожку: в настройках игры оставьте язык озвучки «English», текст — русский; отдельного пункта «Русский» в списке озвучек не появится."
    : a.adapter === "fmod" || a.adapter === "wwise"
    ? "Звуковые файлы игры с озвучкой пересобираются с заменёнными репликами и ставятся на место английских; оригиналы — в резервной копии, «Удалить озвучку» их возвращает. В игре оставьте английскую озвучку и русские субтитры. Проверка файлов в Steam вернёт английскую озвучку — тогда установите заново."
    : "Файлы игры заменяются, оригиналы сохраняются в резервную копию и возвращаются кнопкой «Удалить озвучку».";
  const mb = info.purge_estimate_mb || 0;
  const purgeNote = info.settings?.after_install !== "keep" && mb > 0
    ? ` После установки освободится ~${mb >= 1024 ? (mb / 1024).toFixed(1) + " ГБ" : mb + " МБ"}: удалятся оригиналы, пробы и стенд (их можно вернуть); переводы, голоса, готовая озвучка и резервная копия остаются.` : "";
  const purged = info.originals_purged;
  return card(5, info.installed ? "Озвучка установлена" : "Установка в игру",
    h("div", {}, h("p", { class: "muted" }, `${method} Установятся только одобренные реплики (${approved}).` + (info.installed ? ` Сейчас установлено: ${info.installed_count}.` : "") + purgeNote),
      !info.installed && h("p", { class: "muted small" }, "Озвучка RuVoice в игру не установлена."),
      purged ? h("p", { class: "muted small" }, `Оригиналы удалены после установки (освобождено ${(purged.bytes / 2 ** 30).toFixed(1)} ГБ). Для прослушивания оригиналов и переозвучки программа вернёт их сама; можно вернуть сейчас. `,
        h("button", { class: "btn small", disabled: busy, onclick: () => run("/restore-originals", {}) }, "Вернуть оригиналы")) : ""),
    [h("button", { class: "btn primary big", disabled: busy || !approved, onclick: () => confirm(`Установить ${approved} реплик в игру?`) && runSteps(["install"]) },
       info.installed ? "Обновить установку" : "Установить в игру"),
     uninstallButton(busy || !info.installed)],
    info.installed ? "done" : "active");
}

function adapterNotes(a) {
  // Пояснения адаптера: зашифрованные архивы (нужен AES-ключ), античит, как запускать игру с модом
  if (!a || (!a.warnings?.length && !a.notes?.length)) return "";
  return h("div", {},
    a.warnings?.length ? h("div", { class: "callout warn" }, a.warnings.map(w => h("p", {}, "⚠ ", w))) : "",
    a.notes?.length ? h("details", { class: "notes" }, h("summary", {}, "Как это работает для этой игры"),
      h("div", { class: "callout" }, a.notes.map(n => h("p", { class: "muted" }, n)))) : "");
}

// Шаг 0: на чём считать (видеокарта NVIDIA / AMD / процессор) и установлены ли нейросети под это железо.
// Показывается до первой подготовки и всегда, пока чего-то не хватает.
function modelsMissing() { const rec = window.systemInfo?.models?.recommendation; return !rec || !!rec.no_gpu || !!rec.missing.length; }

function hardwareCard(info) {
  const sys = window.systemInfo;
  if (!sys?.accel?.variant) return h("div", { class: "callout" }, h("span", { class: "spinner" }), " Определяем железо…");
  const a = sys.accel, m = sys.models, rec = m?.recommendation, dl = m?.download || {};
  const missing = rec ? rec.missing.length : 0;
  const chosen = a.preferred !== "auto" ? a.preferred : a.variant;
  const confirmed = a.preferred !== "auto";
  if (info.prepared && !missing) return "";
  const gpuNames = (a.gpus || []).join(", ") || "видеокарта не найдена";
  const using = a.device === "cuda" ? `${a.device_name}, ${a.vram_gb} ГБ видеопамяти` : `процессор, ОЗУ ${a.ram_gb ?? "?"} ГБ`;
  // Выбор «видеокарта или процессор» пользователю не нужен — программа решает сама: видеокарта, если она есть
  // и поддерживается; иначе процессор как запасной путь. Ручной переключатель остаётся в «Система и нейросети».
  const amdNoRocm = a.vendor === "amd" && !(a.installed || []).includes("rocm");
  if (a.device !== "cuda" && !amdNoRocm) {
    return h("div", { class: "callout bad hw-compact" },
      h("span", {}, `Видеокарта для нейросетей не найдена (${gpuNames}). Озвучка работает только на видеокарте: NVIDIA с 6 ГБ видеопамяти и больше или поддерживаемая AMD (ROCm). `),
      h("button", { class: "btn subtle small", onclick: openSystem }, "Подробнее…"));
  }
  if (!missing && !amdNoRocm) {
    return h("div", { class: "callout good hw-compact" },
      h("span", {}, `Считаем на видеокарте: ${a.device_name}, ${a.vram_gb} ГБ. Набор нейросетей «${rec?.name}» установлен. `),
      h("button", { class: "btn subtle small", onclick: openSystem }, "Подробнее…"));
  }
  if (amdNoRocm && !missing) {
    return h("div", { class: "callout warn hw-compact" },
      h("span", {}, a.amd?.family === "rdna2"
        ? `Найдена ${gpuNames}. ${a.amd.note} Установка ~3 ГБ. `
        : a.amd?.family === "unknown" ? `Найдена ${gpuNames}. ${a.amd.note} `
        : `Найдена видеокарта AMD (${gpuNames}). Для озвучки нужна сборка ROCm (~2 ГБ): Radeon RX 9000, часть RX 7000, Ryzen AI 300/400/Max. Без неё озвучка недоступна. `),
      h("button", { class: "btn primary small", disabled: sys.amd_install?.running, onclick: async () => { await api("POST", "/api/system/install-amd").catch(x => alert(x.message)); pollSystemUntilIdle(); } },
        sys.amd_install?.running ? (sys.amd_install.message || "Устанавливается…") : (a.amd?.family === "rdna2" ? "Установить ускорение AMD (RX 6000, экспериментально)" : "Установить ускорение AMD")),
      sys.amd_install?.message && !sys.amd_install.running ? h("span", { class: "muted small" }, sys.amd_install.message) : "",
      sys.amd_install?.error ? h("span", { class: "error" }, sys.amd_install.error) : "");
  }
  const pick = async v => {
    if (v === chosen && confirmed) return;
    if (!confirm(`Переключить на «${OPTION_NAMES[v]}»? Программа перезапустится (несколько секунд).`)) return;
    try {
      await api("POST", "/api/system/accel", { variant: v });
      await api("POST", "/api/system/restart");
    } catch (e) { alert(e.message); return; }
    document.body.replaceChildren(h("div", { class: "restarting" }, h("span", { class: "spinner" }), " Перезапуск программы…"));
  };
  const OPTION_NAMES = { cuda: "Видеокарта NVIDIA", rocm: "Видеокарта AMD", cpu: "Процессор и ОЗУ" };
  const installed = a.installed || [];
  const isAmd = a.vendor === "amd", isNvidia = a.vendor === "nvidia";
  const opt = (v, title, desc, extra) => {
    const avail = installed.includes(v);
    return h("div", { class: `hw-opt ${chosen === v ? "current" : ""} ${avail ? "" : "disabled"}`, onclick: () => avail && pick(v) },
      h("div", { class: "hw-head" }, h("input", { type: "radio", name: "hw", checked: chosen === v, disabled: !avail }), h("b", {}, title),
        chosen === v && h("span", { class: "pill" }, confirmed ? "выбрано" : "сейчас")),
      h("p", { class: "muted small" }, desc), extra || "");
  };
  const options = [];
  if (isNvidia || installed.includes("cuda") && !isAmd)
    options.push(opt("cuda", "Видеокарта NVIDIA (CUDA)", `${gpuNames}. Быстро: ~2 с на реплику, обучение голосов 5–60 мин на персонажа.`));
  if (isAmd)
    options.push(opt("rocm", "Видеокарта AMD (ROCm)",
      installed.includes("rocm") ? `${gpuNames}. Быстро: как на NVIDIA.`
        : `${gpuNames}. Нужна сборка ROCm (~2 ГБ). Поддерживаются Radeon RX 9000, часть RX 7000 и Ryzen AI 300/400/Max с драйвером AMD 26.2.2+. На других AMD — только процессор.`,
      !installed.includes("rocm") && h("button", { class: "btn primary small", disabled: sys.amd_install?.running,
        onclick: async e => { e.stopPropagation(); await api("POST", "/api/system/install-amd").catch(x => alert(x.message)); pollSystemUntilIdle(); } },
        sys.amd_install?.running ? (sys.amd_install.message || "Устанавливается…") : "Установить ускорение AMD")));
  options.push(opt("cpu", "Процессор и ОЗУ", `ОЗУ ${a.ram_gb ?? "?"} ГБ. Медленно: 10–40 с на реплику, голоса персонажей не обучаются (говорят похожие дикторы). Подходит, если видеокарта не поддерживается.`));
  const tier = rec && h("div", { class: "hw-models" },
    h("p", {}, `Нейросети под этот выбор — набор `, h("b", {}, `«${rec.name}»`), ` (${rec.why}): `,
      missing ? `не хватает ${missing} из ${rec.models.length} моделей, ${rec.missing_gb} ГБ.` : (rec.missing_optional?.length ? "всё нужное для озвучки установлено ✓ (без модели перевода)" : "установлен полностью ✓")),
    rec.warnings.map(w => h("p", { class: "small" }, "⚠ ", w)),
    dl.running ? h("div", { class: "row" }, h("span", { class: "spinner" }), h("span", {}, dl.message),
        dl.key && m.models[dl.key]?.progress != null && h("div", { class: "progress wide" }, h("div", { style: `width:${Math.round(m.models[dl.key].progress * 100)}%` })))
      : missing ? h("div", { class: "row" },
          h("button", { class: "btn primary", onclick: () => { installTier(rec.tier); } }, `Скачать набор «${rec.name}» (${rec.missing_gb} ГБ)`),
          h("button", { class: "btn subtle small", onclick: openSystem }, "Другие наборы…")) : "");
  return h("div", { class: "step active hw" },
    h("div", { class: "num" }, "0"),
    h("div", { class: "step-main" }, h("h3", {}, "Нейросети"),
      h("p", { class: "muted" }, `Найдено: ${gpuNames} · ОЗУ ${a.ram_gb ?? "?"} ГБ. Считаем на: ${using}.`),
      tier));
}

function modelsWarning() { return ""; }

// ---------- персонажи (свёрнуто) ----------

// Панели персонажей нет: единый порядок для всех игр — озвучивается всё, кто где говорит, программа
// определяет сама (speakers.py, шаг подготовки).

// ---------- настройки (свёрнуто) ----------

// Стенд движков: одни и те же реплики разными вариантами конвейера — цифры здесь, прослушивание на странице прогона
function settingsPanel(info) {
  if (!info.exists) return "";
  const s = info.settings || {};
  // Сохранить и перерисовать, не сворачивая панели (renderStudioBody хранит раскрытые <details>)
  const save = async patch => {
    studio.info.settings = await api("POST", projUrl("/settings"), patch);
    studio.info = await api("GET", projUrl("")).catch(() => studio.info);
    renderStudioBody();
  };
  const radio = (name, value, label) => h("label", { class: "radio" },
    h("input", { type: "radio", name, value, checked: s[name] === value, onchange: () => save({ [name]: value }) }), h("span", {}, label));
  const check = (name, label, dflt = false) => h("label", { class: "radio" },
    h("input", { type: "checkbox", checked: s[name] == null ? dflt : !!s[name], onchange: e => save({ [name]: e.target.checked }) }), h("span", {}, label));
  const glossary = h("textarea", { rows: 4, placeholder: "Alyx = Аликс\nGordon Freeman = Гордон Фримен\nCombine = Альянс" }, s.glossary || "");
  glossary.addEventListener("change", () => save({ glossary: glossary.value }));
  const sel = (name, options, parse = v => v) => h("select", { onchange: e => save({ [name]: parse(e.target.value) }) },
    options.map(([v, l]) => h("option", { value: v, selected: String(s[name]) === String(v) }, l)));
  // Настройка в строку: подпись слева, список справа
  const field = (label, control, hint) => h("label", { class: "field" }, h("span", {}, label, hint ? h("small", { class: "muted" }, hint) : ""), control);
  return h("details", { class: "panel" },
    h("summary", {}, "⚙ Настройки"),
    h("div", { class: "settings" },
      h("h3", {}, "Голос и исходное исполнение"),
      h("p", { class: "muted small" }, "Произношение рождается от русского донора, тембр персонажа переносит обученная модель; кто какой голос получает — на панели «Голоса». Сначала послушайте пробы, затем одобрите результат или укажите, что поправить."),
      field("После установки в игру", sel("after_install", [["minimum", "Оставить минимум — удалить оригиналы, пробы и стенд (их можно вернуть)"],
          ["keep", "Оставить всё"]]), "переводы, голоса, готовая озвучка и резервная копия остаются всегда"),
      h("h3", {}, "Акустика оригинала"),
      field("Эхо и окружение", sel("effects_mode", [["auto", "Автоматически — эхо и фон по хвосту каждой реплики"], ["engine", "Добавляет игра — озвучка чистая"],
          ["dry", "Записи чистые, без эффектов"], ["baked", "Записаны в файле — переносить даже короткое эхо"]]),
          "где в оригинале слышен хвост эха, русская реплика получает такой же"),
      check("preserve_performance", "Сохранять из оригинала изолированные вздохи, смех и кашель между словами", true),
      h("h3", {}, "Словарь имён и терминов"),
      h("p", { class: "muted small" }, "По одному на строку: «Английское = Русское». Переводчик будет использовать именно эти варианты."),
      glossary,
      h("h3", {}, "Сколько стараться"),
      Object.entries(QUALITY).map(([k, label]) => radio("quality", k, label)),
      h("h3", {}, "Откуда брать русский текст"),
      radio("text_source", "official", "Официальные русские субтитры (лучшее качество)"),
      radio("text_source", "llm", "Перевести всё нейросетью (долго: сначала распознаётся вся речь)"),
      check("translate_missing", "Реплики без русских субтитров переводить нейросетью (иначе они остаются с оригинальным голосом)"),
      h("div", { class: "row" },
        h("button", { class: "btn small", onclick: openSystem }, "Система и нейросети…"),
        h("button", { class: "btn small", title: "Сохранить готовую озвучку в файл .ruvoice — перенести на другой ПК или поделиться", onclick: () => runSteps(["export"]) }, "📦 Экспорт озвучки"),
        h("button", { class: "btn small", title: "Загрузить файл .ruvoice с готовой озвучкой этой игры", onclick: () => importPackage(studio.gid) }, "📥 Импорт озвучки"),
        info.last_export ? h("button", { class: "btn subtle small", onclick: () => api("POST", projUrl("/open-exports")) }, `Папка с пакетами (${info.last_export.mb} МБ)`) : "",
        info.installed && uninstallButton(busy),
        h("button", { class: "btn subtle small", onclick: async () => {
          if (!confirm("Удалить проект озвучки (переводы, обученные голоса и сгенерированный звук)? Игра не затрагивается.")) return;
          try { await api("DELETE", projUrl("/project")); closeStudio(); } catch (e) { alert(e.message); }
        } }, "✕ Удалить проект озвучки (переводы и голоса; игра не трогается)"))));
}

// ---------- голоса персонажей ----------

// Уровень (обучать тембр / быстрый путь), донор русской речи, состояние тембра, прослушивание оригинала и донора
let voicesCache = { gid: null, rows: null, donors: null };
const TIMBRE_STATE = { trained: ["ok", "тембр обучен"], weak: ["warn", "модель не принята"], training: ["", "ждёт контроля"], none: ["", "не обучен"] };
function voicesPanel(info, busy) {
  if (!(info.characters || []).length) return "";
  const box = h("details", { class: "panel" }, h("summary", {}, `Голоса — персонажей: ${info.characters.length}`),
    h("p", { class: "muted small" }, "Главные персонажи получают проверенный тембр поверх постоянного русского образца. Если реплика попала в чужую группу, укажите нужного персонажа в пробах или в списке реплик."),
    h("button", { class: "btn small", onclick: () => { studio.view = "lines"; studio.filters = { status: "", char: "", q: "", offset: 0 }; renderStudio(); } }, "Открыть все реплики"),
    h("div", { class: "voices-list" }, h("span", { class: "muted small" }, "Загрузка…")));
  const reload = async () => { voicesCache.rows = null; await refreshStudio(); };
  const fill = (rows, donors) => {
    const list = box.querySelector(".voices-list");
    const donorOpt = (row, d) => h("option", { value: d.id, selected: row.donor?.id === d.id },
      `${d.id} · ${d.gender === "male" ? "муж." : "жен."} ${d.age || ""} ${Math.round(d.f0 || 0)} Гц`);
    list.replaceChildren(h("table", { class: "chars voices" },
      h("thead", {}, h("tr", {}, h("th", {}, "Персонаж"), h("th", { class: "num" }, "Реплик"), h("th", {}, "Голос"), h("th", {}, "Уровень"), h("th", {}, "Донор"), h("th", {}, "Тембр"), h("th", {}, "Послушать"))),
      h("tbody", {}, rows.map(r => {
        const [cls, label] = TIMBRE_STATE[r.timbre?.state] || ["", r.timbre?.state];
        return h("tr", {},
          h("td", {}, h("b", {}, r.name)),
          h("td", { class: "num" }, `${r.lines} · ${Math.round(r.seconds / 60)} мин`),
          h("td", {}, r.voice_type || "—"),
          h("td", {}, h("select", { disabled: busy, onchange: async e => { await api("POST", projUrl("/voices/tier"), { char: r.id, tier: e.target.value }).catch(x => alert(x.message)); reload(); } },
            h("option", { value: "auto", selected: !r.override }, r.tier === "main" ? "главный (авто)" : "эпизодический (авто)"),
            h("option", { value: "train", selected: r.override === "train" }, "обучать тембр"),
            h("option", { value: "episodic", selected: r.override === "episodic" }, "быстрый путь"))),
          h("td", {}, r.tier === "main" || r.donor
            ? h("select", { disabled: busy, onchange: async e => { await api("POST", projUrl("/voices/donor"), { char: r.id, donor: e.target.value }).catch(x => alert(x.message)); reload(); } },
                !r.donor && h("option", { value: "", selected: true }, "— подберётся сам —"), donors.map(d => donorOpt(r, d)))
            : h("span", { class: "muted small" }, "клон с записи")),
          h("td", {}, h("span", { class: `pill ${cls}`, title: r.timbre?.reason || "" }, label),
            r.tier === "main" && r.timbre?.state !== "trained" && r.timbre?.reason !== "перенос тембра не установлен"
              ? h("button", { class: "btn subtle small", disabled: busy, title: r.timbre?.reason || "", onclick: async () => {
                  await api("POST", projUrl("/train-voice"), { char: r.id }).catch(x => alert(x.message)); pollStudio(); } }, "Обучить")
              : "",
            r.timbre?.reason && r.timbre.state !== "trained" ? h("div", { class: "muted small" }, r.timbre.reason) : ""),
          h("td", {}, h("div", { class: "row" }, r.samples.slice(0, 2).map((id, i) => playBtn("orig", id, `Оригинал ${i + 1}`)),
            r.donor && h("button", { class: "btn play gen", onclick: e => playUrl(e.currentTarget, `/api/donors/audio?id=${encodeURIComponent(r.donor.id)}&style=calm&t=${Date.now()}`) }, "▶ Донор"))));
      }))));
  };
  if (voicesCache.gid === studio.gid && voicesCache.rows && voicesCache.donors) fill(voicesCache.rows, voicesCache.donors);
  else Promise.all([api("GET", projUrl("/voices")), voicesCache.donors ? Promise.resolve(voicesCache.donors) : api("GET", "/api/donors")])
    .then(([rows, donors]) => { voicesCache = { gid: studio.gid, rows, donors }; fill(rows, donors); })
    .catch(e => box.querySelector(".voices-list").replaceChildren(h("span", { class: "error" }, e.message)));
  return box;
}

// ---------- прослушивание ----------

function speakerPicker(id, char, busy) {
  if (busy || studio.info.characters.length < 2) return "";
  return h("details", { class: "speaker-picker", "data-panel": `speaker-${id}` },
    h("summary", {}, "Сменить персонажа"),
    h("select", { "aria-label": "Персонаж", onchange: async e => {
      const target = e.target.value;
      if (target === char) return;
      e.target.disabled = true;
      try {
        await api("POST", projUrl("/line/speaker"), { id, char: target });
        voicesCache.rows = null;
        await refreshStudio();
      } catch (error) { alert(error.message); e.target.disabled = false; e.target.value = char; }
    } }, studio.info.characters.map(c => h("option", { value: c.id, selected: c.id === char }, c.name))));
}

function lineCard(l) {
  const ch = studio.info.characters.find(c => c.id === l.char);
  const flag = l.info?.flag === "long" ? "Русская фраза заметно длиннее оригинала — лучше сократить текст"
    : l.info?.flag === "short" ? "Русская фраза намного короче оригинала — проверьте" : null;
  const ta = h("textarea", { rows: 2, placeholder: "Русский текст" }, l.ru || "");
  ta.addEventListener("change", async () => {
    const upd = await api("PATCH", projUrl(`/line?id=${encodeURIComponent(l.id)}`), { ru: ta.value }).catch(e => alert(e.message));
    if (upd) { Object.assign(l, upd); card.replaceWith(lineCard(l)); }
  });
  const setStatus = async status => {
    const upd = await api("PATCH", projUrl(`/line?id=${encodeURIComponent(l.id)}`), { status });
    Object.assign(l, upd);
    card.replaceWith(lineCard(l));
  };
  const card = h("div", { class: `line st-${l.status}` },
    h("div", { class: "line-head" },
      h("span", { class: "dot", style: `background:${ch?.color || "#555"}` }),
      h("b", {}, ch?.name || l.char),
      speakerPicker(l.id, l.char, studio.info.status.running || studio.info.status.queued.length),
      h("span", { class: "muted small" }, l.id),
      h("span", { class: `lstatus st-${l.status}` }, LINE_STATUS[l.status] || l.status),
      l.ru_source === "official" && h("span", { class: "tag" }, "официальный текст"),
      l.ru_source === "user" && h("span", { class: "tag" }, "изменено вами"),
      l.info?.plan && h("span", { class: "tag", title: `источник ${l.info.plan.source}${l.info.plan.donor ? ", донор " + l.info.plan.donor : ""}` },
        l.info.plan.timbre ? "донор + тембр" : (l.info.plan.donor ? "донор" : "клон с записи")),
      judgeChip(l.info?.judge),
    (l.info?.fit?.kept || []).length ? h("span", { class: "muted small", title: "Взято из оригинала и вклеено в русскую реплику" }, ` · из оригинала: ${l.info.fit.kept.join(", ")}`) : "",
    l.info?.fit?.room ? h("span", { class: "muted small", title: `Эхо оригинала перенесено: спад ${l.info.fit.room.rt60} с, доля ${l.info.fit.room.wet}, фон ${l.info.fit.room.noise_db} дБ` }, ` · эхо ${l.info.fit.room.rt60} с`) : ""),
    h("div", { class: "line-grid" },
      h("div", {}, h("div", { class: "muted small" }, `Оригинал · ${(l.duration || 0).toFixed(1)} с`),
        h("div", { class: "en" }, l.en || h("i", { class: "muted" }, "текст не распознан")), playBtn("orig", l.id, "Оригинал")),
      h("div", {}, h("div", { class: "muted small" }, l.gen_duration ? `Русский · ${l.gen_duration.toFixed(1)} с${l.info?.speed > 1.01 ? ` · ускорено ×${l.info.speed}` : ""}` : "Русский"),
        ta,
        h("div", { class: "row" },
          l.gen && playBtn("gen", l.id, "Русский"),
          l.gen && l.status !== "approved" && h("button", { class: "btn ok", onclick: () => setStatus("approved") }, "✓ Одобрить"),
          l.gen && l.status !== "rejected" && h("button", { class: "btn danger", onclick: () => setStatus("rejected") }, "✗ Плохо"),
          l.ru && h("button", { class: "btn", onclick: () => api("POST", projUrl("/regenerate"), { ids: [l.id] }).then(pollStudio) }, l.gen ? "⟳ Заново" : "🎙 Озвучить"),
          l.status !== "skipped" ? h("button", { class: "btn subtle", title: "Не озвучивать эту реплику", onclick: () => setStatus("skipped") }, "Пропустить")
            : h("button", { class: "btn subtle", onclick: () => setStatus(l.ru ? "translated" : "new") }, "Вернуть")))),
    flag && h("div", { class: "flag" }, "⚠ ", flag),
    l.info?.delivery?.note ? h("div", { class: "muted small" }, "🎬 Режиссёр: ", l.info.delivery.note,
      l.info.spoken ? h("span", {}, ` · читается как «${l.info.spoken}»`) : "") : "",
    l.info?.fix?.plan ? h("div", { class: "muted small" }, "🔧 ", l.info.fix.plan) : "",
    noteInput(l),
    judgeDetails(l.info?.judge),
    l.error && ["error", "skipped"].includes(l.status) && h("div", { class: "flag" }, "⚠ ", l.error));
  return card;
}

function judgeChip(j) {
  if (!j || j.score == null) return "";
  const cls = j.verdict === "ok" ? (j.score >= 85 ? "good" : "mid") : "bad";
  return h("span", { class: `judge ${cls}`, title: "Оценка контролёра качества" }, j.verdict === "ok" ? "✓" : "⚠", ` контролёр: ${j.score}`);
}

function judgeDetails(j) {
  if (!j || !(j.reasons || []).length) return "";
  return h("div", { class: `judge-box ${j.verdict}` },
    h("ul", { class: "reasons" }, j.reasons.map(r => h("li", {}, r))),
    (j.problems || []).length ? h("ul", { class: "actions muted small" }, j.problems.map(pr => h("li", {}, "→ ", pr.action))) : "",
    j.heard && j.cer > 0.15 && h("div", { class: "muted small" }, "Распознано: «", j.heard, "»"),
    j.pron_heard && (j.pron_words || []).length ? h("div", { class: "muted small" }, "Как звучит на самом деле: «", j.pron_heard, "»") : "");
}

// Указание к реплике: что поправить в подаче — режиссёр превратит его в подачу и реплика переозвучится
function noteInput(l) {
  if (!l.ru || l.status === "skipped") return "";
  const inp = h("input", { class: "note-input", value: l.info?.user_note || "", placeholder: "Как сыграть иначе: «злее, почти крик», «тише, с паузой после обращения»… (Enter — переозвучить)" });
  inp.addEventListener("keydown", async e => {
    if (e.key !== "Enter" || !inp.value.trim()) return;
    await api("POST", projUrl("/line-note"), { id: l.id, note: inp.value }).catch(x => alert(x.message));
    pollStudio();
  });
  return h("div", { class: "row note-row" }, inp);
}

async function renderLines(body) {
  const f = studio.filters;
  const qs = new URLSearchParams({ limit: PAGE, offset: f.offset });
  if (f.status === "rework") qs.set("rework", "1");
  else if (f.status) qs.set("status", f.status);
  if (f.char) qs.set("char", f.char);
  if (f.q) qs.set("q", f.q);
  const data = await api("GET", projUrl(`/lines?${qs}`));
  const by = studio.info.stats.by_status;
  const statusSel = h("select", { onchange: e => { f.status = e.target.value; f.offset = 0; renderStudioBody(); } },
    h("option", { value: "" }, `Все (${studio.info.stats.total})`),
    h("option", { value: "rework", selected: f.status === "rework" }, `⚠ Нужна доработка (${studio.info.stats.rework || 0})`),
    Object.entries(LINE_STATUS).map(([k, v]) => h("option", { value: k, selected: f.status === k }, `${v} (${by[k] || 0})`)));
  const charSel = h("select", { onchange: e => { f.char = e.target.value; f.offset = 0; renderStudioBody(); } },
    h("option", { value: "" }, "Все персонажи"),
    studio.info.characters.map(c => h("option", { value: c.id, selected: f.char === c.id }, `${c.name} (${c.lines})`)));
  const search = h("input", { type: "search", placeholder: "Поиск по тексту…", value: f.q });
  search.addEventListener("change", () => { f.q = search.value; f.offset = 0; renderStudioBody(); });
  const pages = Math.ceil(data.total / PAGE), page = Math.floor(f.offset / PAGE) + 1;
  body.replaceChildren(
    h("div", { class: "filters-row" }, statusSel, charSel, search),
    h("div", { class: "row bulk" },
      h("button", { class: "btn ok", disabled: !(by.generated), onclick: approveGood }, "✓ Одобрить все хорошие"),
      h("button", { class: "btn ok", disabled: !(by.generated), onclick: () => bulkMark("approved") },
        f.char ? "✓ Одобрить всё озвученное у персонажа" : "✓ Одобрить всё озвученное"),
      h("button", { class: "btn", disabled: !(studio.info.stats.rework), onclick: regenRework }, `⟳ Переозвучить слабые (${studio.info.stats.rework || 0})`),
      h("button", { class: "btn", disabled: !(by.rejected), onclick: async () => {
        const r = await api("GET", projUrl(`/lines?status=rejected&limit=5000`));
        await api("POST", projUrl("/regenerate"), { ids: r.items.map(l => l.id) }); pollStudio();
      } }, `⟳ Переозвучить отклонённые (${by.rejected || 0})`)),
    h("p", { class: "muted small" }, "Совет: слушайте выборочно (особенно отмеченные контролёром), а потом одобряйте пачкой. Неудачные — «Плохо» и «Заново»; текст можно править прямо здесь."),
    data.items.length ? h("div", { class: "lines" }, data.items.map(lineCard)) : h("p", { class: "muted" }, "Нет реплик с такими условиями."),
    pages <= 1 ? "" : h("div", { class: "row pager" },
      h("button", { class: "btn", disabled: page <= 1, onclick: () => { f.offset -= PAGE; renderStudioBody(); } }, "← Назад"),
      h("span", {}, `Страница ${page} из ${pages}`),
      h("button", { class: "btn", disabled: page >= pages, onclick: () => { f.offset += PAGE; renderStudioBody(); } }, "Вперёд →")));
}

async function bulkMark(status) {
  const f = studio.filters;
  const ch = f.char ? studio.info.characters.find(c => c.id === f.char)?.name : null;
  if (!confirm(`Одобрить все озвученные реплики${ch ? ` персонажа «${ch}»` : ""}? Их можно будет отменить по одной.`)) return;
  const r = await api("POST", projUrl("/mark"), { status, char: f.char || null, from_status: "generated" });
  await refreshStudio();
  alert(`Одобрено: ${r.changed}`);
}


// Импорт пакета .ruvoice: файл уходит телом запроса, программа сама находит игру (или берёт открытую)
function importPackage(gid) {
  const inp = document.createElement("input");
  inp.type = "file"; inp.accept = ".ruvoice,.zip";
  inp.onchange = async () => {
    const f = inp.files[0]; if (!f) return;
    const qs = new URLSearchParams({ name: f.name }); if (gid) qs.set("gid", gid);
    try {
      const r = await fetch(`/api/import?${qs}`, { method: "POST", body: f, headers: { "Content-Type": "application/octet-stream" } });
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || r.statusText);
      if (!gid && typeof openStudio === "function") openStudio(j.game_id); else pollStudio();
    } catch (e) { alert(e.message); }
  };
  inp.click();
}
