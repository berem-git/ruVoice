"use strict";
// Система: железо → подходящий набор нейросетей → установка. Плюс менеджер языков.

window.systemInfo = null;
let sysRetry = 0;

async function loadSystem() {
  window.systemInfo = await api("GET", "/api/system").catch(() => null);
  if (window.systemInfo) window.systemInfo.storage = await api("GET", "/api/system/storage").catch(() => null);
  // Железо определяется в фоне пару секунд после старта — переспрашиваем, пока нет данных
  if (window.systemInfo && !window.systemInfo.accel?.variant && sysRetry++ < 20) setTimeout(loadSystem, 2000);
  renderBanner();
  // Студия показывает шаг «на чём считать» по этим данным — перерисовать, как только железо определилось
  if (typeof studio !== "undefined" && studio.gid && typeof renderStudioBody === "function") renderStudioBody();
  return window.systemInfo;
}

const ROLE_NAMES = { tts: "Голос", llm: "Перевод", asr: "Распознавание речи" };

function hardwareLine(a) {
  if (!a?.variant) return "Определяется…";
  const gpu = (a.gpus || []).join(", ") || "видеокарта не найдена";
  const use = a.device === "cuda" ? `${a.device_name}, ${a.vram_gb} ГБ видеопамяти` : "процессор";
  return `${gpu} · ОЗУ ${a.ram_gb ?? "?"} ГБ · используется: ${use}`;
}

function systemPanel(sys) {
  if (!sys) return h("p", { class: "muted" }, "Нет данных о системе");
  const a = sys.accel || {}, m = sys.models, rec = m.recommendation, dl = m.download;
  const isAmd = a.vendor === "amd";
  const variantSel = h("select", { onchange: async e => {
      await api("POST", "/api/system/accel", { variant: e.target.value });
      alert("Сохранено. Перезапустите программу, чтобы применить.");
    } },
    h("option", { value: "auto", selected: a.preferred === "auto" }, "Авто"),
    Object.entries(sys.variants).map(([k, v]) => h("option", { value: k, selected: a.preferred === k, disabled: !(a.installed || []).includes(k) },
      v + ((a.installed || []).includes(k) ? "" : " (не установлено)"))));

  const tierCard = ([id, t]) => {
    const missing = t.models.filter(k => !m.models[k].present);
    const missingGb = missing.reduce((s, k) => s + m.models[k].size_gb, 0);
    const isRec = rec?.tier === id;
    return h("div", { class: `tier ${isRec ? "recommended" : ""} ${t.installed ? "installed" : ""}` },
      h("div", { class: "tier-head" }, h("b", {}, t.name), isRec && h("span", { class: "pill" }, "Рекомендуем"), t.installed && h("span", { class: "pill ok" }, "✓ Установлен")),
      h("div", { class: "muted small" }, t.needs),
      h("p", {}, t.desc),
      h("ul", { class: "models" }, t.models.map(k => h("li", {}, m.models[k].present ? "✓ " : "○ ", m.models[k].name,
        h("span", { class: "muted small" }, ` · ${m.models[k].size_gb} ГБ`)))),
      h("div", { class: "row" },
        t.installed ? h("span", { class: "muted small" }, `Всего ${t.total_gb} ГБ на диске`)
          : h("button", { class: `btn ${isRec ? "primary" : ""}`, disabled: dl.running, onclick: () => installTier(id) }, `Установить (${missingGb.toFixed(1)} ГБ)`)));
  };

  const progress = dl.running && h("div", { class: "dl" },
    h("div", { class: "row" }, h("span", { class: "spinner" }), h("span", {}, dl.message),
      dl.queue.length ? h("span", { class: "muted small" }, `· в очереди ещё ${dl.queue.length}`) : ""),
    dl.key && m.models[dl.key].progress != null && h("div", { class: "progress wide" }, h("div", { style: `width:${Math.round(m.models[dl.key].progress * 100)}%` })));

  return h("div", { class: "system" },
    h("h3", {}, "Железо"),
    h("p", {}, hardwareLine(a)),
    a.torch && h("p", { class: "muted small" }, `Ускоритель: ${a.variant_name} · PyTorch ${a.torch}`),
    h("details", { class: "advanced" }, h("summary", { class: "muted small" }, "Принудительный режим (только если что-то не работает)"),
      h("label", {}, "Считать на ", variantSel)),
    isAmd && !(a.installed || []).includes("rocm") && h("div", { class: "callout warn" },
      h("p", {}, a.amd?.family === "rdna2"
        ? `Обнаружена ${(a.gpus || []).join(", ")}. ${a.amd.note}`
        : "Обнаружена видеокарта AMD. Для ускорения нужна сборка ROCm (около 2 ГБ). Работает на Radeon RX 9000, части RX 7000 и Ryzen AI 300/400/Max с драйвером AMD 26.2.2 или новее; для RX 6000 ставится ночная сборка TheRock (экспериментально). Без видеокарты озвучка недоступна."),
      h("button", { class: "btn primary", disabled: sys.amd_install?.running, onclick: async () => {
        await api("POST", "/api/system/install-amd"); pollSystemUntilIdle();
      } }, sys.amd_install?.running ? "Устанавливается…" : "Установить ускорение AMD"),
      sys.amd_install?.message && h("p", { class: "muted small" }, sys.amd_install.message),
      sys.amd_install?.error && h("p", { class: "error" }, sys.amd_install.error)),
    a.error && h("p", { class: "error" }, a.error),

    h("h3", {}, "Нейросети"),
    rec?.no_gpu ? h("div", { class: "callout bad" }, h("p", {}, "Видеокарта для нейросетей не найдена — озвучка недоступна."), rec.warnings.map(w => h("p", { class: "small" }, "⚠ ", w)))
    : rec ? h("div", { class: `callout ${rec.missing.length ? "warn" : "good"}` },
      h("p", {}, `Под ваше железо (${rec.why}) подходит набор `, h("b", {}, `«${rec.name}»`), ". ",
        rec.missing.length ? `Не хватает ${rec.missing.length} из ${rec.models.length} моделей — ${rec.missing_gb} ГБ.` : "Он установлен полностью."),
      rec.warnings.map(w => h("p", { class: "small" }, "⚠ ", w)),
      rec.missing.length > 0 && h("button", { class: "btn primary", disabled: dl.running, onclick: () => installTier(rec.tier) }, `Установить набор «${rec.name}»`))
      : h("p", { class: "muted" }, "Определяем железо…"),
    progress,
    dl.error && h("p", { class: "error" }, dl.error),
    h("div", { class: "tiers" }, Object.entries(m.tiers).map(tierCard)),
    Object.keys(m.addons || {}).length > 0 && h("h3", {}, "Дополнения"),
    h("div", { class: "tiers" }, Object.entries(m.addons || {}).map(([id, t]) => h("div", { class: `tier ${t.installed ? "installed" : ""}` },
      h("div", { class: "tier-head" }, h("b", {}, t.name), t.installed && h("span", { class: "pill ok" }, "✓ Установлено")),
      h("p", {}, t.desc),
      h("ul", { class: "models" }, t.models.map(k => h("li", {}, m.models[k].present ? "✓ " : "○ ", m.models[k].name, h("span", { class: "muted small" }, ` · ${m.models[k].size_gb} ГБ`)))),
      h("div", { class: "row" }, t.installed ? h("span", { class: "muted small" }, `${t.total_gb} ГБ на диске`)
        : h("button", { class: "btn", disabled: dl.running, onclick: () => installTier(id) }, `Установить (${t.missing_gb} ГБ)`))))),
    h("p", { class: "muted small" }, `Свободно на диске: ${m.disk_free_gb} ГБ. Модели лежат в папке models — при копировании программы они переносятся вместе с ней.`),

    h("details", {}, h("summary", {}, "Все модели по отдельности"),
      h("table", { class: "models-table" }, h("tbody", {}, Object.entries(m.models).map(([k, mm]) => h("tr", {},
        h("td", {}, mm.present ? "✓" : "○"), h("td", {}, mm.name), h("td", { class: "num" }, `${mm.size_gb} ГБ`),
        h("td", {}, mm.present
          ? h("button", { class: "btn subtle small", disabled: dl.running, onclick: () => deleteModel(k, mm.name) }, "Удалить")
          : h("button", { class: "btn small", disabled: dl.running, onclick: () => installKeys([k]) }, "Скачать"))))))),

    h("h3", {}, "Среды движков"),
    enginesPanel(sys.engine_envs),

    h("h3", {}, "Место на диске"),
    storagePanel(sys.storage),

    h("h3", {}, "Инструменты"),
    h("ul", { class: "models" }, Object.entries(sys.tools).map(([k, v]) => h("li", {}, v ? "✓ " : "✗ ", k))));
}

// Среды движков (перенос тембра RVC, OmniVoice, VoxCPM2): ставятся из программы, ход виден здесь
const ENV_DESC = { rvc: "Обученный тембр главных персонажей: Applio + претрейны, ~1,5 ГБ. Без него главные персонажи идут быстрым путём.",
  omnivoice: "Донорский голос с тегами смеха и вздохов; модель ~2,5 ГБ скачается следом.",
  voxcpm: "Клонирование с записи персонажа (только эпизодические); модель ~4,7 ГБ скачается следом." };
function enginesPanel(envs) {
  if (!envs) return h("p", { class: "muted small" }, "Нет данных");
  const anyRunning = Object.values(envs).some(e => e.running);
  return h("div", { class: "tiers" }, Object.entries(envs).map(([id, e]) => h("div", { class: `tier ${e.installed ? "installed" : ""}` },
    h("div", { class: "tier-head" }, h("b", {}, e.name), e.installed && h("span", { class: "pill ok" }, "✓ Установлена")),
    h("p", {}, ENV_DESC[id] || ""),
    e.running ? h("div", { class: "row" }, h("span", { class: "spinner" }), h("span", { class: "muted small" }, e.message)) : "",
    e.error ? h("p", { class: "error" }, e.error) : "",
    h("div", { class: "row" }, h("button", { class: `btn ${id === "rvc" && !e.installed ? "primary" : ""} small`, disabled: anyRunning,
      onclick: async () => { try { await api("POST", "/api/system/engines/install", { engine: id }); } catch (x) { alert(x.message); return; } pollSystemUntilIdle(); } },
      e.installed ? "Переустановить" : "Установить")))));
}

async function installTier(tier) {
  try { await api("POST", "/api/system/models/download", { tier }); } catch (e) { alert(e.message); return; }
  pollSystemUntilIdle();
}

async function installKeys(keys) {
  try { await api("POST", "/api/system/models/download", { keys }); } catch (e) { alert(e.message); return; }
  pollSystemUntilIdle();
}

async function deleteModel(key, name) {
  if (!confirm(`Удалить модель «${name}» с диска? Её можно будет скачать снова.`)) return;
  try { await api("DELETE", `/api/system/models/${encodeURIComponent(key)}`); } catch (e) { alert(e.message); return; }
  await loadSystem();
  rerenderSystem();
}

let sysPolling = false;
async function pollSystemUntilIdle() {
  if (sysPolling) return;
  sysPolling = true;
  try {
    while (true) {
      const s = await loadSystem();
      rerenderSystem();
      if (!s?.amd_install?.running && !s?.models?.download?.running && !Object.values(s?.engine_envs || {}).some(e => e.running)) break;
      await new Promise(r => setTimeout(r, 2000));
    }
  } finally { sysPolling = false; }
}

function rerenderSystem() {
  const modalBody = document.getElementById("system-body");
  if (modalBody && !document.getElementById("system-modal").classList.contains("hidden"))
    modalBody.replaceChildren(systemPanel(window.systemInfo));
  if (typeof studio !== "undefined" && studio.gid && (studio.tab === "settings" || studio.tab === "prep")) renderStudioBody();
  renderBanner();
}

async function openSystem() {
  document.getElementById("system-modal").classList.remove("hidden");
  document.getElementById("system-body").replaceChildren(h("p", { class: "muted" }, "Загрузка…"));
  await loadSystem();
  rerenderSystem();
  if (!window.systemInfo?.accel?.variant) setTimeout(() => { if (!document.getElementById("system-modal").classList.contains("hidden")) openSystem(); }, 2000);
  if (window.systemInfo?.models?.download?.running) pollSystemUntilIdle();
}

// Подсказка в библиотеке, пока не установлен подходящий набор нейросетей
function storagePanel(st) {
  if (!st) return h("p", { class: "muted small" }, "Считаем…");
  const orphans = (st.projects || []).filter(p => p.game_missing);
  const row = (label, gb, extra) => h("tr", {}, h("td", {}, label), h("td", { class: "num" }, `${gb} ГБ`), h("td", {}, extra || ""));
  return h("div", {},
    h("table", { class: "chars" }, h("tbody", {},
      row("Нейросети (models)", st.models_gb, h("span", { class: "muted small" }, "удалять по одной — в списке ниже")),
      row("Среда и PyTorch (runtime)", st.runtime_gb),
      row("Инструменты (tools)", st.tools_gb),
      row("Проекты озвучки (data\\projects)", st.projects_gb),
      row("Кэш", st.cache_gb, st.cache_gb > 0.05 && h("button", { class: "btn small", onclick: async () => { await api("DELETE", "/api/system/storage/cache"); await loadSystem(); rerenderSystem(); } }, "Очистить")))),
    h("p", { class: "muted small" }, `Свободно на диске: ${st.disk_free_gb} ГБ`),
    h("div", { class: `callout ${st.garbage_gb >= 0.5 ? "warn" : "good"}` },
      h("span", {}, st.garbage_gb >= 0.01 ? `Мусор: ${st.garbage_gb} ГБ — варианты реплик, рабочие папки обучения, неиспользуемые дообученные модели, кэш. Программа убирает его сама при запуске и после установки озвучки. `
        : "Мусора нет — программа убирает его сама при запуске и после установки озвучки. "),
      st.garbage_gb >= 0.01 ? h("button", { class: "btn small", onclick: async e => { e.target.disabled = true; const r = await api("POST", "/api/system/storage/cleanup"); alert(`Освобождено ${r.freed_gb} ГБ`); await loadSystem(); rerenderSystem(); } }, "Убрать сейчас") : ""),
    (st.projects || []).length ? h("div", {}, h("h4", {}, "Проекты озвучки"),
      h("table", { class: "chars" }, h("tbody", {}, st.projects.map(p => h("tr", {},
        h("td", {}, p.name, p.game_missing ? h("span", { class: "pill warn", title: "Папки игры больше нет на диске" }, " игра удалена") : "",
          h("div", { class: "muted small" }, Object.entries(p.parts_gb || {}).filter(([, v]) => v >= 0.01).map(([k, v]) => `${({ orig: "оригинал", gen: "озвучка", voices: "голоса", profiles: "профили", backup: "резервные копии", samples: "пробы", bench: "контрольная сцена" })[k] || k} ${v}`).join(" · ")),
          p.garbage_gb >= 0.01 ? h("div", { class: "muted small", title: (p.garbage || []).join("\n") }, `мусор ${p.garbage_gb} ГБ`) : ""),
        h("td", { class: "num" }, `${p.size_gb} ГБ`),
        h("td", {}, p.game_missing ? h("button", { class: "btn danger small", onclick: async () => {
            if (!confirm(`Удалить проект озвучки «${p.name}» (${p.size_gb} ГБ)? Игры на диске уже нет.`)) return;
            await api("DELETE", `/api/system/storage/project/${encodeURIComponent(p.dir)}`); await loadSystem(); rerenderSystem();
          } }, "Удалить") : h("span", { class: "muted small" }, p.installed_count ? `установлено ${p.installed_count} реплик` : ""))))))) : "");
}

function renderBanner() {
  const box = document.getElementById("banner");
  if (!box) return;
  const m = window.systemInfo?.models, rec = m?.recommendation, dl = m?.download;
  const orphans = (window.systemInfo?.storage?.projects || []).filter(p => p.game_missing);
  const garbage = window.systemInfo?.storage?.garbage_gb || 0;
  if (rec && !rec.missing.length && !dl?.running && !orphans.length && garbage >= 1) {
    box.classList.remove("hidden");
    box.replaceChildren(h("div", { class: "row" }, h("span", {}, `🧹 Накопилось ${garbage} ГБ мусора (варианты реплик, рабочие файлы обучения, кэш).`),
      h("button", { class: "btn small", onclick: async () => { const r = await api("POST", "/api/system/storage/cleanup"); alert(`Освобождено ${r.freed_gb} ГБ`); loadSystem(); } }, "Убрать")));
    return;
  }
  if (rec && !rec.missing.length && !dl?.running && orphans.length) {
    const gb = orphans.reduce((a, p) => a + p.size_gb, 0).toFixed(1);
    box.classList.remove("hidden");
    box.replaceChildren(h("div", { class: "row" },
      h("span", {}, `🧹 ${orphans.length === 1 ? "Игра удалена" : "Игры удалены"}: ${orphans.map(p => p.name).join(", ")} — ${orphans.length === 1 ? "её проект озвучки занимает" : "их проекты озвучки занимают"} ${gb} ГБ.`),
      h("button", { class: "btn small", onclick: openSystem }, "Освободить место…")));
    return;
  }
  if (!rec || (!rec.missing.length && !dl?.running)) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  box.replaceChildren(
    dl?.running
      ? h("div", { class: "row" }, h("span", { class: "spinner" }), h("span", {}, dl.message),
        dl.key && m.models[dl.key].progress != null && h("span", { class: "muted" }, `${Math.round(m.models[dl.key].progress * 100)}%`),
        h("button", { class: "btn small", onclick: openSystem }, "Подробнее"))
      : h("div", { class: "row" },
        h("span", {}, "🧠 Для озвучки нужны нейросети. Под ваше железо (", rec.why, ") подходит набор ", h("b", {}, `«${rec.name}»`), ` — ${rec.missing_gb} ГБ.`),
        h("button", { class: "btn primary small", onclick: () => installTier(rec.tier) }, "Установить"),
        h("button", { class: "btn small", onclick: openSystem }, "Выбрать другой")));
  if (dl?.running) pollSystemUntilIdle();
}

// ---------- менеджер языков ----------

async function openLanguageDelete(g, lang) {
  const modal = document.getElementById("lang-modal");
  const box = document.getElementById("lang-body");
  modal.classList.remove("hidden");
  box.replaceChildren(h("p", { class: "muted" }, "Поиск файлов…"));
  const data = await api("GET", `/api/games/${encodeURIComponent(g.id)}/language-files?code=${encodeURIComponent(lang.code)}`);
  const names = { audio: "Озвучка", video: "Видео", text: "Текст / субтитры", other: "Прочие файлы (тип не определён)" };
  const checks = {};
  const kinds = Object.entries(data.by_kind).filter(([, v]) => v.count);
  const steamWarn = g.source === "steam";
  const isGameLang = g.steam_language && lang.name && lang.code === ({ english: "en", russian: "ru", german: "de", french: "fr" }[g.steam_language]);
  box.replaceChildren(h("div", {},
    h("h3", {}, `Удалить язык: ${lang.name}`),
    h("p", { class: "muted" }, `Найдено файлов: ${data.total}. Они будут перемещены в Корзину Windows — пока Корзина не очищена, всё можно вернуть.`),
    kinds.map(([k, v]) => {
      const cb = h("input", { type: "checkbox", checked: k === "audio" || k === "video" });
      checks[k] = cb;
      return h("label", { class: "radio" }, cb, h("span", {}, h("b", {}, names[k]), ` — ${v.count} файлов, ${fmtSize(v.size)}`));
    }),
    isGameLang && h("div", { class: "callout bad" }, "Это язык, выбранный для игры в Steam! Без него игра может не запуститься или остаться без звука."),
    steamWarn && h("div", { class: "callout warn" }, "Steam может скачать удалённые файлы обратно при проверке целостности или обновлении игры."),
    h("details", {}, h("summary", {}, "Список файлов"),
      h("div", { class: "path filelist" }, data.files.map(f => h("div", {}, `${fmtSize(f.size).padStart(9)}  ${f.rel}`)))),
    h("div", { class: "error", id: "lang-error" }),
    h("div", { class: "modal-foot" },
      h("button", { class: "btn", onclick: () => modal.classList.add("hidden") }, "Отмена"),
      h("button", { class: "btn danger", onclick: async () => {
        const sel = Object.entries(checks).filter(([, cb]) => cb.checked).map(([k]) => k);
        if (!sel.length) return;
        try {
          const r = await api("POST", `/api/games/${encodeURIComponent(g.id)}/language-delete?code=${encodeURIComponent(lang.code)}`, { kinds: sel });
          modal.classList.add("hidden");
          alert(`Перемещено в Корзину: ${r.deleted} файлов, ${fmtSize(r.freed)}.` + (r.errors.length ? `\nНе удалось: ${r.errors.length}` : "") +
            "\nЧтобы освободить место, очистите Корзину.");
          await api("POST", `/api/games/${encodeURIComponent(g.id)}/analyze`);
          pollStatus();
        } catch (e) { document.getElementById("lang-error").textContent = e.message; }
      } }, "Удалить в Корзину"))));
}

loadSystem();
