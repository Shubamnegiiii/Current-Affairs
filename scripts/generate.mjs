import fs from "fs";
import * as cheerio from "cheerio";

const KEY = process.env.AI_API_KEY;
const MODEL = process.env.MODEL || "gemini-2.5-flash";
const ONLY = (process.env.FILE || "").trim();
const MAX_BATCHES = parseInt(process.env.MAX_BATCHES || "0", 10) || Infinity;
const BATCH_SIZE = 8;
const DELAY_MS = 6000;

if (!KEY) { console.error("Missing AI_API_KEY secret"); process.exit(1); }

const GEN = fs.readFileSync("prompts/generate.txt", "utf8");
const REV = fs.readFileSync("prompts/review.txt", "utf8");
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const norm = (s) => String(s).toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();

class QuotaError extends Error {}

async function callAI(prompt) {
  for (let attempt = 1; attempt <= 4; attempt++) {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": KEY },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
          generationConfig: { temperature: 0.3, responseMimeType: "application/json" },
        }),
      }
    );
    if (res.status === 429 || res.status === 503) {
      if (attempt === 4) throw new QuotaError("limit reached");
      console.log(`Busy/limit (status ${res.status}), waiting...`);
      await sleep(attempt * 30000);
      continue;
    }
    if (!res.ok) throw new Error(`API error status ${res.status}`);
    const data = await res.json();
    return (data.candidates?.[0]?.content?.parts || []).map((p) => p.text || "").join("");
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

function extractItems(html) {
  const $ = cheerio.load(html);
  $("script,style,nav").remove();
  for (const sel of [".card", ".news-card", ".news-item", "article", "section"]) {
    const els = $(sel);
    if (els.length >= 5) {
      const items = [];
      els.each((_, el) => {
        const t = $(el).text().replace(/\s+/g, " ").trim();
        if (t.length > 60) items.push(t);
      });
      if (items.length >= 5) return items;
    }
  }
  const items = [];
  let cur = "";
  $("body").find("h1,h2,h3,h4,p,li,td").each((_, el) => {
    const t = $(el).text().replace(/\s+/g, " ").trim();
    if (!t) return;
    if (/^h[1-4]$/.test(el.tagName.toLowerCase())) {
      if (cur.length > 60) items.push(cur);
      cur = t + ". ";
    } else cur += t + " ";
  });
  if (cur.length > 60) items.push(cur);
  return items;
}

function overlap(line, source) {
  const w = norm(line).split(" ").filter(Boolean);
  if (!w.length) return 0;
  const src = new Set(norm(source).split(" "));
  return w.filter((x) => src.has(x)).length / w.length;
}

function valid(q, source) {
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
  if (!ONLY && fs.existsSync(progPath)) prog = JSON.parse(fs.readFileSync(progPath, "utf8"));
  if (prog.complete && !ONLY) { console.log(`${month}: already complete, skipping`); continue; }

  const items = extractItems(fs.readFileSync(file, "utf8"));
  console.log(`${month}: found ${items.length} news items`);
  if (items.length < 5) { console.log("Too few items found, skipping. Check HTML structure."); continue; }

  const batches = [];
  for (let i = 0; i < items.length; i += BATCH_SIZE) batches.push(items.slice(i, i + BATCH_SIZE).map((t, j) => ({ id: `${month}-${i + j + 1}`, text: t.slice(0, 1500) })));

  for (let b = 0; b < batches.length; b++) {
    if (prog.done.includes(b)) continue;
    if (batchesRun >= MAX_BATCHES) { stop = true; break; }
    const batchText = batches[b].map((x) => `[${x.id}] ${x.text}`).join("\n\n");
    const source = batches[b].map((x) => x.text).join(" ");
    try {
      const gen = parseJSON(await callAI(`${GEN}\n\nNEWS ITEMS:\n${batchText}`));
      await sleep(DELAY_MS);
      const rev = parseJSON(await callAI(`${REV}\n\nNEWS ITEMS:\n${batchText}\n\nQUESTIONS:\n${JSON.stringify(gen)}`));
      await sleep(DELAY_MS);
      const candidates = rev.length ? rev : gen;
      const seen = new Set(prog.questions.map((q) => norm(q.question)));
      let kept = 0;
      for (const q of candidates) {
        if (valid(q, source) && !seen.has(norm(q.question))) {
          prog.questions.push(shuffleOpts(q));
          seen.add(norm(q.question));
          kept++;
        } else prog.rejected++;
      }
      prog.done.push(b);
      batchesRun++;
      console.log(`${month} batch ${b + 1}/${batches.length}: kept ${kept}`);
    } catch (e) {
      if (e instanceof QuotaError) {
        console.log("Daily limit reached. Progress saved. Run again later to continue.");
        stop = true;
        break;
      }
      console.log(`Batch ${b + 1} failed (${e.message}), will retry next run.`);
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
