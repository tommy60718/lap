# W3 accepted package — plain-language analysis

This report looks at an **already accepted** training package. It does not retrain anything.
It only reads the sealed numbers and draws pictures so a beginner can see what those numbers mean.

## The big picture in one minute

Think of the verifier as a matching game: given a robot motion snippet, can it find the right language
description (and the other way around)? Random guessing would only succeed at the **exact chance** rates
below (pool size 1118). Anything clearly above chance is the model doing real work.

- Authority boundary: **offline only** (`recorded_data_offline_integration_only`). Not robot control. Not live candidate picking.
- Wrist / two-view benefit established? **no**.

## Did training settle down?

Training loss is how wrong the model is on the data it sees while learning. Validation loss is the same idea
on held-out data. Both should generally fall as epochs go on.

- Epochs: 0 → 49
- Train loss: 4.1628 → 2.0222
- Validation loss: 4.1184 → 3.4391

Figure: `figures/fig_loss_curves.png`

## Matching better than chance?

Top-1 means “the correct match is ranked first.” Top-5 means “it is in the first five.”
We compare each score to exact chance for this pool.

| Direction | Top-1 | Top-5 | Chance top-1 | Chance top-5 |
|---|---:|---:|---:|---:|
| Action → language | 0.0197 | 0.0993 | 0.000894 | 0.004472 |
| Language → action | 0.0250 | 0.1199 | 0.000894 | 0.004472 |

Action→language top-1 95% CI: [0.0123, 0.0275]

Language→action top-1 95% CI: [0.0143, 0.0379]

Figure: `figures/fig_retrieval_vs_chance.png`

## Are good pairs stronger than bad pairs?

A **margin** asks: does the correct pairing score higher than a bad pairing?
Shuffled = random wrong pairs. Nearby = hard near-miss pairs.
Positive mean with a 95% interval above zero means the advantage is statistically on the right side of zero.

- Shuffled margin mean 5.3806, 95% CI [5.0658, 5.7371] (fraction > 0: 0.948)
- Nearby margin mean 0.8617, 95% CI [0.5149, 1.2365] (fraction > 0: 0.670)

Figure: `figures/fig_margins_ci95.png`

## Eight conditions at a glance

The eval splits into eight shape×direction buckets. The heatmap shows action→language top-1 in each bucket.
Uneven cells mean some motion settings are harder than others — useful context, not a retune knob.

Figure: `figures/fig_conditions_heatmap.png`

## Two-view versus base-only (wrist camera)

Paired ablation asks: does adding the wrist view help **the same** protocol?
Bars show two-view minus base-only. Error bars are the paired 95% intervals.

Verdict: wrist benefit **was not established**. The sealed package still accepts offline deployment, but this particular gain is not proven.

Figure: `figures/fig_paired_ablation.png`

## Failures and caveats (read this before trusting a headline)

- Authority is offline-only: recorded-data integration, not robot or candidate-selection authority.
- Wrist benefit was not established (paired CI includes zero or fails the seal rule).
- This analysis is read-only visualization of the accepted package; it does not retrain or retune thresholds.
- Checkpoints were not copied into the analysis directory.

Machine-readable twin: `analysis_summary.json` (schema `osx_cover_w3_post_acceptance_analysis_v1`, content_hash `e4c0feb6470f69c17bd05a4ad39415038a2099c55c0853f77a89f080576d8480`).
