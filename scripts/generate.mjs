import fs from "fs";
import * as cheerio from "cheerio";

const KEY = process.env.AI_API_KEY;
const MODEL = process.env.MODEL || "gemini-2.5-flash";
const ONLY = (process.env.FILE || "").trim();
const MAX_BATCHES = parseInt(process.env.MAX_BATCHES || "0", 10) || Infinity;
const BATCH_SIZE = 6;
const DELAY_MS = 6000;
const TIMEOUT_MS = 180000;

if (!KEY) { console.error("Missing AI_API_KEY secret"); process.exit(1); }

const GEN = fs.readFileSync("prompts/generate.txt", "utf8");
const REV = fs.readFileSync("prompts/review.txt", "utf8");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
const squash = (s) => s.replace(/\s+/g, " ").trim();

class QuotaError extends Error {}
class FatalError extends Error {}

async function callAI(prompt) {
  for (let attempt = 1; attempt <= 4; attempt++) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), TIMEOUT_MS);
    try {
      const res = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-goog-api-key": KEY },
          body: JSON.stringify({
            contents: [{ parts: [{ text: prompt }] }],
            generationConfig: { temperature: 0.3, responseMimeType: "application/json" },
          }),
          signal: ctrl.signal,
        }
      );
      clearTimeout(timer);
      if (res.status === 429 || res.status === 503) {
        if (attempt === 4) throw new QuotaError("limit reached");
        console.log(`Busy/limit (status ${res.status}), waiting...`);
        await sleep(attempt * 30000);
        continue;
      }
      if (!res.ok) {
        const body = (await res.text()).slice(0, 300);
        if ([400, 401, 403, 404].includes(res.status)) throw new FatalError(`API error status ${res.status}: ${body}`);
        throw new Error(`API error status ${res.status}: ${body}`);
      }
      const data = await res.json();
      return (data.candidates?.[0]?.content?.parts || []).map((p) => p.text || "").join("");
    } catch (e) {
      clearTimeout(timer);
      if (e instanceof QuotaError || e instanceof FatalError) throw e;
      const why = e.name === "AbortError" ? "timed out after 3 minutes" : `${e.message} ${e.cause?.code || e.cause?.message || ""}`;
      console.log(`Problem on try ${attempt}/4: ${why}`);
      if (attempt === 4) throw new Error(`failed: ${why}`);
      await sleep(attempt * 10000);
    }
  }
}

function parseJSON(text) {
  try {
    const clean = String(text).replace(/```json|```/g, "").trim();
    const v = JSON.parse(clean);
    return Array.isArray(v) ? v : [];
  } catch {
    return [];
  }
}

function cardText(node) {
  const c = node.clone();
  c.find("script,style").remove();
  c.find("br").replaceWith("\n");
  c.find("div,p,li,h1,h2,h3,h4,h5,h6,tr,section").append("\n");
  return c.text().replace(/[ \t\u00a0]+/g, " ").replace(/\s*\n\s*/g, "\n").trim();
}

function extractItems(html, month) {
  const $ = cheerio.load(html);
  $("script,style,noscript").remove();
  const FOOT = /^(.{2,100}?)\s*•\s*(\d{1,3}\.\d{1,3})$/;
  const feet = [];

  $("body *").each((_, el) => {
    const $el = $(el);
    if ($el.find("*").length > 8) return;
    const t = squash($el.text());
    if (t.length > 120) return;
    const m = t.match(FOOT);
    if (!m) return;
    const childHit = $el.children().toArray().some((c) => FOOT.test(squash($(c).text())));
    if (childHit) return;
    feet.push({ el, cat: m[1].replace(/^.*\bNotes\s+/i, "").trim(), num: m[2] });
  });
  console.log(`${month}: footer markers found: ${feet.length}`);
  feet.forEach((f) => $(f.el).attr("data-f", "1"));

  const items = [];
  const seenId = new Set();
  feet.forEach((f, i) => {
    let node = $(f.el);
    for (;;) {
      const p = node.parent();
      if (!p.length || p.is("body,html")) break;
      if (p.find("[data-f]").length !== 1) break;
      node = p;
    }
    const text = cardText(node);
    let id = `${month}-${f.num}`;
    if (seenId.has(id)) id += `-${i}`;
    seenId.add(id);
    if (text.length < 80) return;
    items.push({ id, cat: f.cat, text: text.slice(0, 3500) });
  });
  return items;
}

function overlap(line, source) {
  const w = norm(line).split(" ").filter(Boolean);
  if (!w.length) return 0;
  const src = new Set(norm(source).split(" "));
  return w.filter((x) => src.has(x)).length / w.length;
}

function valid(q, source) {
  if (!source) return false;
  if (!q || typeof q !== "object" || !q.options) return false;
  const keys = ["A", "B", "C", "D"];
  if (!keys.every((k) => typeof q.options[k] === "string" && q.options[k].trim())) return false;
  if (new Set(keys.map((k) => norm(q.options[k]))).size !== 4) return false;
  if (!keys.includes(q.answer)) return false;
  if (!q.question || q.question.trim().length < 15) return false;
  if (/all of the above|none of the above|both a and b/i.test(JSON.stringify(q.options))) return false;
  if (!q.source_line || overlap(q.source_line, source) < 0.8) return false;
  return true;
}

function shuffleOpts(q) {
  const keys = ["A", "B", "C", "D"];
  const vals = keys.map((k) => q.options[k]);
  if (vals.some((v) => /\d/.test(v))) return q;
  const correct = q.options[q.answer];
  for (let i = vals.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [vals[i], vals[j]] = [vals[j], vals[i]];
  }
  q.options = Object.fromEntries(keys.map((k, i) => [k, vals[i]]));
  q.answer = keys[vals.indexOf(correct)];
  return q;
}

// Quick check that the model name and key work before doing real work
try {
  await callAI('Reply with this JSON only: {"ok":true}');
  console.log(`Model check OK: ${MODEL}`);
} catch (e) {
  console.log(`MODEL CHECK FAILED for ${MODEL}: ${e.message}`);
  process.exit(1);
}

fs.mkdirSync("quizzes", { recursive: true });

const files = fs
  .readdirSync(".")
  .filter((f) => /^CA_.*_Card_Notes\.html$/.test(f))
  .filter((f) => !ONLY || f === ONLY);

let batchesRun = 0;
let stop = false;

for (const file of files) {
  if (stop) break;
  const month = file.match(/^CA_(.+)_Card_Notes\.html$/)[1];
  const outPath = `quizzes/Quiz_${month}.json`;
  const progPath = `quizzes/.progress_${month}.json`;

  let prog = { done: [], questions: [], rejected: 0, complete: false };
  if (fs.existsSync(progPath)) prog = JSON.parse(fs.readFileSync(progPath, "utf8"));
  if (prog.complete) {
    console.log(`${month}: already complete. To redo it, delete ${progPath} and run again.`);
    continue;
  }

  const items = extractItems(fs.readFileSync(file, "utf8"), month);
  console.log(`${month}: found ${items.length} news items`);
  if (items.length < 5) { console.log("Too few items found, skipping. Check HTML structure."); continue; }

  const batches = [];
  for (let i = 0; i < items.length; i += BATCH_SIZE) batches.push(items.slice(i, i + BATCH_SIZE));

  let failsInRow = 0;
  for (let b = 0; b < batches.length; b++) {
    if (prog.done.includes(b)) continue;
    if (batchesRun >= MAX_BATCHES) { stop = true; break; }
    const batch = batches[b];
    const byId = Object.fromEntries(batch.map((x) => [x.id, x]));
    const batchText = batch.map((x) => `[${x.id}] (${x.cat}) ${x.text}`).join("\n\n");
    try {
      const gen = parseJSON(await callAI(`${GEN}\n\nNEWS ITEMS:\n${batchText}`));
      await sleep(DELAY_MS);
      const rev = parseJSON(await callAI(`${REV}\n\nNEWS ITEMS:\n${batchText}\n\nQUESTIONS:\n${JSON.stringify(gen)}`));
      await sleep(DELAY_MS);
      const candidates = rev.length ? rev : gen;
      const seen = new Set(prog.questions.map((q) => norm(q.question)));
      let kept = 0;
      for (const q of candidates) {
        const item = q && byId[q.news_id];
        if (item && valid(q, item.text) && !seen.has(norm(q.question))) {
          q.topic = item.cat;
          prog.questions.push(shuffleOpts(q));
          seen.add(norm(q.question));
          kept++;
        } else prog.rejected++;
      }
      prog.done.push(b);
      batchesRun++;
      failsInRow = 0;
      console.log(`${month} batch ${b + 1}/${batches.length}: kept ${kept} of ${candidates.length}`);
    } catch (e) {
      if (e instanceof QuotaError) {
        console.log("Daily limit reached. Progress saved. Run again later to continue.");
        stop = true;
        break;
      }
      if (e instanceof FatalError) {
        console.log(`STOPPED: ${e.message}`);
        stop = true;
        process.exitCode = 1;
        break;
      }
      failsInRow++;
      console.log(`Batch ${b + 1} failed (${e.message}). Failures in a row: ${failsInRow}`);
      if (failsInRow >= 3) {
        console.log("3 failures in a row, stopping. Run again later.");
        stop = true;
        break;
      }
    }
    fs.writeFileSync(outPath, JSON.stringify(prog.questions, null, 1));
    fs.writeFileSync(progPath, JSON.stringify(prog));
  }

  prog.complete = prog.done.length === batches.length;
  fs.writeFileSync(outPath, JSON.stringify(prog.questions, null, 1));
  fs.writeFileSync(progPath, JSON.stringify(prog));
  fs.writeFileSync(
    `quizzes/report_${month}.md`,
    `# ${month}\n- Questions kept: ${prog.questions.length}\n- Rejected: ${prog.rejected}\n- Batches done: ${prog.done.length}/${batches.length}\n- Complete: ${prog.complete}\n`
  );
}
