# 03 — Rubric Locking Decision & Feasible Fine-Tuning Plan (i5-13400H + RTX 3050, Demo Timeframe)

## Part A — Locked Rubrics vs. Manual Entry: The Decision Is Mostly Made For You

This isn't purely a performance question — it's constrained by what CertiProof (the system this model is being integrated into next) already requires structurally.

### The performance argument (locked wins)
- Rubric criteria embeddings (`R` in the model) are recomputed on every request if entered manually/live. For a fixed exam, this is wasted, repeated compute for identical input — cache `R` once per certified rubric.
- Locked rubrics allow a **golden-set regression check** (compute-and-compare against known-good outputs) before an exam goes live — impossible if the rubric can change per request.

### The integrity argument (locked is close to mandatory)
CertiProof's existing design already **hash-commits the rubric before any submission exists** — "Teacher certifies a rubric (hashes rubric + both zones' model identity, locked before any submission exists)" is a direct quote from its own README. A rubric that could still be edited live would break the entire tamper-detection story this system is built around — there is no meaningful "prove this score came from this rubric" claim if the rubric itself isn't fixed at proving time.

### Recommendation
**Implement two modes, not one:**
1. **Dev/test mode** (current default in `Rcaj-x-mini-v1`, keep this) — rubrics can be entered or edited freely, useful for benchmarking and the Streamlit console during development.
2. **Locked/certified mode** (new, required before CertiProof integration) — once a rubric is certified, `R` (its embedding) is computed once, cached, and hash-committed. Any subsequent request against that rubric reuses the cached `R` rather than recomputing it, and any attempt to submit a *different* rubric text under the same rubric ID should fail loudly (hash mismatch), not silently re-embed new text.

Implement this now, even before the CertiProof integration plan runs, since it's a small change (a cache keyed by rubric hash + a mode flag) and it directly de-risks the integration work later — see `certiproof-integration-plan/00_INTEGRATION_OVERVIEW.md` for how this maps onto CertiProof's existing certification flow.

---

## Part B — Feasible Fine-Tuning & Negation Hardening on This Specific Hardware

### Why this is newly worth doing (it wasn't, before)
Earlier planning assumed CPU-only, unlimited-time evaluation and explicitly deferred encoder fine-tuning as out of scope. A dedicated RTX 3050 changes that calculus — BGE-small is 33M parameters, comfortably fine-tunable on 4GB+ VRAM in minutes, not hours. This is the single highest-leverage upgrade available given the hardware you actually have, and it's now affordable in demo time. Do this.

### Scoped Recipe (fits a focused work session, not unlimited time)
1. **Triplet construction** (30–45 min of human time, not GPU time): from your reviewed dataset (`02`), build `(criterion, correct_chunk, hard_negative_chunk)` triplets:
   - **Negation-flipped hard negatives** — highest priority, directly targets the negation blind spot: for every correct chunk, pair it with its `negation_flipped` variant as a forced-dissimilar negative.
   - **Confidently-wrong hard negatives** — same-topic, wrong-fact chunks, forced dissimilar.
   - Random negatives (chunks from unrelated questions) as easy negatives to stabilize early training — cheap to generate, don't hand-author these.
2. **Fine-tuning run**: use `sentence-transformers`' `MultipleNegativesRankingLoss` or `TripletLoss`, batch size 16–32 (fits comfortably in 4GB VRAM at this model size), 2–4 epochs, small learning rate (2e-5 to 5e-5). **Expected wall-clock time on this GPU: single-digit minutes** for a dataset in the 500–1000 triplet range — this is not an overnight job.
3. **Freeze immediately after.** From this point the fine-tuned encoder is treated exactly like the original frozen BGE — no further training, hash-committed once finalized (same pattern as the rest of the project).
4. **Before/after ablation (required, not optional):** re-run the existing benchmark suite with both the vanilla and fine-tuned encoder, compare specifically on `negation_flipped`, `confidently_wrong`, and `paraphrase` categories — these are where the gain should concentrate. Report both numbers; don't just swap the encoder and assume improvement.

### What NOT to spend demo-time budget on (explicitly deprioritized)
- **Do not** attempt to fine-tune or swap the whole cross-attention + scoring head architecture — only the frozen encoder gets fine-tuned; the attention/scoring layers are retrained from scratch on the upgraded data (`02`) as already planned, not "fine-tuned" separately.
- **Do not** attempt a full entropy-vs-spread ambiguity signal comparison in this pass — spread is already working structurally; this is a marginal-gain, non-demo-critical investigation.
- **Do not** attempt multi-head count sweeps beyond a quick 2 vs. 4 comparison — a full grid search is not worth GPU time here; pick 4 heads unless the quick comparison shows a clear win for a different count.
- **Do not** attempt LoRA/PEFT-style partial fine-tuning — at 33M params, full fine-tuning is already cheap enough that LoRA's memory savings aren't needed, and full fine-tuning is simpler to implement correctly in the time available.

### Priority Order If Time Runs Short
1. Score bounding fix (`01`) — do regardless, it's nearly free.
2. Data upgrade (`02`) — do as much as time allows; even partial progress (3 subjects instead of 5) is a real improvement over synthetic-only data.
3. Encoder fine-tuning (this file) — do this before attempting a large hyperparameter sweep; it's the higher-leverage use of limited GPU time.
4. Rubric-locking cache (Part A) — small, do it, it directly de-risks the next phase.

## Checkpoint
- Two rubric modes implemented (dev/manual, locked/certified), with the locked mode caching `R` per rubric hash.
- Fine-tuned encoder produced, hash-committed, with a documented before/after ablation showing measured (not assumed) improvement on `negation_flipped`, `confidently_wrong`, `paraphrase`.
- Total GPU time spent on fine-tuning documented (should be single-digit-to-low-double-digit minutes, not hours) — if it's taking much longer, something about batch size or data loading is likely misconfigured, not a sign that more time is needed.
