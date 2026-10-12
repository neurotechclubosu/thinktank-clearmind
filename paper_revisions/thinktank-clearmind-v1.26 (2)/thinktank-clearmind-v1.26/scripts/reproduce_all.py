#!/usr/bin/env python3
"""
Reproduce every number quoted in docs/CORRECTIONS.md, docs/CONSISTENCY_ANALYSIS.md
and the revised manuscript.

Usage:
    python scripts/reproduce_all.py [--R 2000] [--redraw-B 200] [--perm 2000]
                                    [--phoneme-source {fixture,cmudict}]

Default phoneme source is the hand-transcribed fixture
(data/fixtures/trial_phonemes.json) -- see its "_meta" field.  Pass
--phoneme-source cmudict once cmudict/g2p_en are installed to regenerate with an
independently sourced lexicon (recommended before the release is frozen).

Writes results/results.json (all numbers) and results/results_summary.md
(human-readable tables), and prints progress to stderr.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eeg.synthesis import EpochBank, plan_composite, materialize, plan_summary  # noqa: E402
from src.eeg import pipeline as P  # noqa: E402
from src.analysis import inference as I  # noqa: E402
from src.analysis import consistency as C  # noqa: E402
from src.eeg.EEG_NeurovascularVariables import VARIABLES  # noqa: E402


def get_lexicon(source):
    if source == "fixture":
        fx = json.load(open(ROOT / "data/fixtures/trial_phonemes.json"))
        return fx["trials"], fx["lexicon"], fx["_meta"]
    if source == "cmudict":
        import nltk
        from nltk.corpus import cmudict
        d = cmudict.dict()
        fx = json.load(open(ROOT / "data/fixtures/trial_phonemes.json"))
        lex = {}
        for k, t in fx["trials"].items():
            for w in t["response"].replace("—", " ").split():
                w = w.strip(".,!?;:\u2019'\"“”").lower()
                if not w:
                    continue
                if w in d:
                    lex[w] = [p.rstrip("012") for p in d[w][0]]
        return fx["trials"], lex, {"provenance": "nltk cmudict, first pronunciation"}
    raise ValueError(source)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=2000)
    ap.add_argument("--redraw-B", type=int, default=200)
    ap.add_argument("--perm", type=int, default=2000)
    ap.add_argument("--phoneme-source", default="fixture", choices=["fixture", "cmudict"])
    ap.add_argument("--out", default=str(ROOT / "results"))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    t0 = time.time()

    trials, lex, meta = get_lexicon(args.phoneme_source)
    look = lambda w: lex.get(w.lower(), [])  # noqa: E731
    bank = EpochBank(str(ROOT / "data/eeg_recordings"))
    wb = P.WindowBank(bank)

    results = {"phoneme_source": args.phoneme_source, "phoneme_meta": meta,
               "R": args.R, "generated_unix_time": t0, "mc": {}, "consistency": {}}

    # ---- Monte-Carlo microgap-vs-no-microgap comparison, per sentence
    for name, t in trials.items():
        text = t["response"]
        missing = [w.strip(".,!?;:\u2019'\"“”").lower() for w in text.split()
                   if w.strip(".,!?;:\u2019'\"“”").lower() not in lex]
        mc = I.run_mc(text, look, wb, R=args.R, base_seed=0)
        rows = I.summarise_mc(mc)
        plan2 = plan_composite(text, look, microgaps=False)
        ne = [I.effective_n(mc["win2"]), I.effective_n(mc["win1_seed0"])]
        results["mc"][name] = {
            "prompt": t["prompt"], "n_words": len(text.split()),
            "missing_lexicon_entries": missing,
            "no_microgap_summary": plan_summary(plan2),
            "n_windows_no_microgap": int(mc["n_win2"]),
            "n_windows_microgap_mean": float(mc["n_win1"].mean()),
            "n_eff_channels_no_microgap": ne[0], "n_eff_channels_microgap_seed0": ne[1],
            "variables": rows,
        }
        print(f"[mc] {name} done ({time.time()-t0:.1f}s)", file=sys.stderr)

    # ---- Legacy-defect impact (small R; slow path materialises composites)
    results["legacy_impact"] = {}
    for name, t in trials.items():
        mc = I.run_mc(t["response"], look, wb, R=100, legacy=True, bank=bank)
        rows = {r["variable"]: r for r in I.summarise_mc(mc)}
        results["legacy_impact"][name] = rows["CMRO2"]
        print(f"[legacy] {name} done ({time.time()-t0:.1f}s)", file=sys.stderr)

    # ---- Segment-unit and fixed-bounds sensitivity (CMRO2 only, smaller R)
    results["sensitivity"] = {}
    allg = np.concatenate([P.gamma_windows_partial(bank.epochs(n)[:19 * 256]) for n in range(44)])
    fixed_bounds = (float(np.percentile(allg, 1)), float(np.percentile(allg, 99)))
    results["sensitivity"]["fixed_bounds_1_99pct"] = fixed_bounds
    for unit in ("word_final", "phoneme"):
        for bname, bd in (("own", None), ("fixed", fixed_bounds)):
            key = f"{unit}_{bname}"
            results["sensitivity"][key] = {}
            for name, t in trials.items():
                mc = I.run_mc(t["response"], look, wb, R=300, unit=unit, bounds=bd)
                c = [r for r in I.summarise_mc(mc) if r["variable"] == "CMRO2"][0]
                results["sensitivity"][key][name] = c
            print(f"[sensitivity] {key} done ({time.time()-t0:.1f}s)", file=sys.stderr)

    # ---- Dependence among the eight variables (pooled across sentences/seeds)
    all_res = []
    for name, t in trials.items():
        plan = plan_composite(t["response"], look, microgaps=True, seed=0)
        g = wb.composite_gamma(plan)
        all_res.append(P.analyse_gamma(g))
    pooled = {v: np.concatenate([r[v].ravel() for r in all_res]) for v in VARIABLES}
    dep = I.dependence_report(pooled)
    results["dependence"] = {
        "m_eff": dep["m_eff"], "first_pc_share": dep["first_pc_share"],
        "eigenvalues": dep["eigenvalues"].tolist(),
        "spearman": {a: {b: float(dep["spearman"][i, j]) for j, b in enumerate(VARIABLES)}
                     for i, a in enumerate(VARIABLES)},
    }
    print(f"[dependence] done ({time.time()-t0:.1f}s)", file=sys.stderr)

    # ---- Consistency analysis
    ca = C.ConsistencyAnalysis(bank)
    loo = ca.real_reference_distribution()
    keys = ["logpsd_r", "logpsd_rmse_db", "relband_mad", "src_raw_rho", "src_dc_rho",
            "cmro2_profile_rho"]
    results["consistency"]["reference_bounds_1_99pct"] = ca.bounds
    results["consistency"]["real_vs_real_loo"] = {
        k: {"median": float(np.median([r[k] for r in loo])),
            "min": float(np.min([r[k] for r in loo])),
            "max": float(np.max([r[k] for r in loo]))}
        for k in keys}
    results["consistency"]["per_sentence"] = {}
    for name, t in trials.items():
        entry = {}
        for cond, mg in (("no_microgap", False), ("microgap_seed0", True)):
            plan = plan_composite(t["response"], look, microgaps=mg, seed=0)
            x = materialize(plan, bank)
            d = ca.synthetic_vs_real(x)
            pr = {k: I.summarise_mc  # placeholder replaced below
                  for k in []}
            pr = {k: C.percentile_rank(d[k], [r[k] for r in loo]) for k in keys}
            entry[cond] = {"stats": {k: d[k] for k in keys}, "percentile_in_real_vs_real": pr,
                           "n_rows": int(x.shape[0])}
        if args.redraw_B > 0:
            plan_nomg = plan_composite(t["response"], look, microgaps=False)
            red = ca.redraws(plan_nomg, B=args.redraw_B)
            entry["redraw_no_microgap"] = {
                k: {"mean": float(np.mean([r[k] for r in red])),
                    "sd": float(np.std([r[k] for r in red]))}
                for k in keys}
        results["consistency"]["per_sentence"][name] = entry
        print(f"[consistency] {name} done ({time.time()-t0:.1f}s)", file=sys.stderr)

    id_test = C.identification_test(bank, n_perm=args.perm)
    results["consistency"]["phoneme_identification"] = id_test
    print(f"[identification] done ({time.time()-t0:.1f}s)", file=sys.stderr)

    results["wallclock_seconds"] = time.time() - t0
    with open(out / "results.json", "w") as fh:
        json.dump(results, fh, indent=1)

    write_summary_md(results, out / "results_summary.md")
    print(f"DONE in {time.time()-t0:.1f}s -> {out}/results.json", file=sys.stderr)


def write_summary_md(r, path):
    lines = [f"# Reproduction results ({r['phoneme_source']} lexicon, R={r['R']})", ""]
    lines.append("## Microgap vs. no-microgap: CMRO2 (montage-mean effect, Monte Carlo over microgap draws)")
    lines.append("| sentence | no-mg | mg mean | effect | 95% CI | P(effect>0) | p_MC |")
    lines.append("|---|---|---|---|---|---|---|")
    for name, e in r["mc"].items():
        c = [v for v in e["variables"] if v["variable"] == "CMRO2"][0]
        lines.append(f"| {name} | {c['no_microgap']:.4f} | {c['microgap_mean']:.4f} | "
                     f"{c['effect_mean']:+.4f} | [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}] | "
                     f"{c['prob_positive']:.3f} | {c['p_mc']:.4f} |")
    lines.append("")
    lines.append("## Consistency: synthetic composite vs. real-vs-real leave-one-out baseline")
    b = r["consistency"]["real_vs_real_loo"]
    lines.append("| metric | real-vs-real median [range] |")
    lines.append("|---|---|")
    for k, v in b.items():
        lines.append(f"| {k} | {v['median']:.3f} [{v['min']:.3f}, {v['max']:.3f}] |")
    lines.append("")
    for name, e in r["consistency"]["per_sentence"].items():
        lines.append(f"### {name}")
        for cond in ("no_microgap", "microgap_seed0"):
            s = e[cond]["stats"]
            lines.append(f"* {cond}: logpsd_r={s['logpsd_r']:.3f}, "
                         f"src_raw_rho={s['src_raw_rho']:.3f}, "
                         f"cmro2_profile_rho={s['cmro2_profile_rho']:.3f}")
    lines.append("")
    lines.append(f"## Phoneme identification (epoch 0 vs. held-out epochs, 39-way)")
    idt = r["consistency"]["phoneme_identification"]
    lines.append(f"* accuracy = {idt['acc_epoch0_vs_heldout']:.3f} "
                 f"(chance = {idt['chance']:.3f}, real-data ceiling = {idt['acc_real_loo']:.3f}, "
                 f"permutation p = {idt['p_perm']:.4f}, n_perm={idt['n_perm']})")
    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
