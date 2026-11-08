"""Markdown tables from llama-mtp-replay JSON (examples/mtp-replay), 2026-11-07. Plain python, no torch.

  replay_report.py gate --eval-long EL.json --replay R.json [--replay-full RF.json]
      per depth, eval_long.py vs replay on the same rows: groups all/all -> bucket 'all', all/asst -> 'role:assistant'.
      eval_long top1_sub / acc_prob are compared with a --vocab-subset replay (the same 32k Ws), top1_full with a
      full-vocab replay (--replay-full) when given.
  replay_report.py buckets R.json [--circuit M0] [--min-n 500] [--metric top1,acc_prob,acc2] [--dims role,asst,...]
      rows = bucket, cols = depth, one table per metric, plus n and the sums (sumA, Eg, Ep, Ep2).
  replay_report.py diff A.json B.json [--circuit-a M0 --circuit-b M0] [--min-n 500] [--metric acc2]
      paired per-bucket per-depth difference B - A (same inputs and starts required).
"""
import argparse, json, sys


def load(p):
    return json.load(open(p))


def gate(a):
    el = load(a.eval_long)["res"]
    rp = load(a.replay)["results"]["M0"]
    rf = load(a.replay_full)["results"]["M0"] if a.replay_full else None
    K = max(int(k.split("/k")[1]) for k in el if "/k" in k)
    print("| rows | depth | n eval_long | n replay | eval_long top1_sub | replay top1 (subset) | d | eval_long acc_prob | replay acc_prob | d |"
          + (" eval_long top1_full | replay top1 (full) | d |" if rf else ""))
    print("|---|---|---|---|---|---|---|---|---|---|" + ("---|---|---|" if rf else ""))
    for grp, bk in (("all/all", "all"), ("all/asst", "role:assistant")):
        if bk not in rp or el[f"{grp}/k1"]["n"] == 0:
            continue
        for k in range(1, K + 1):
            e = el[f"{grp}/k{k}"]
            r = rp[bk][k - 1]
            line = (f"| {grp.split('/')[1]} | k{k} | {e['n']} | {r['n']} | {e['top1_sub']:.4f} | {r['top1']:.4f} | "
                    f"{r['top1'] - e['top1_sub']:+.4f} | {e['acc_prob']:.4f} | {r['acc_prob']:.4f} | {r['acc_prob'] - e['acc_prob']:+.4f} |")
            if rf and bk in rf:
                f = rf[bk][k - 1]
                line += f" {e['top1_full']:.4f} | {f['top1']:.4f} | {f['top1'] - e['top1_full']:+.4f} |"
            print(line)


def sel(res, dims, min_n):
    out = []
    for b, rows in res.items():
        d = b.split(":")[0]
        if b != "all" and dims and d not in dims:
            continue
        if rows[0]["n"] < min_n and b != "all":
            continue
        out.append(b)
    order = {d: i for i, d in enumerate(dims or [])}
    return sorted(out, key=lambda b: (b != "all", order.get(b.split(":")[0], 99), -res[b][0]["n"]))


def buckets(a):
    R = load(a.replay)
    res = R["results"][a.circuit]
    dims = a.dims.split(",") if a.dims else None
    bks = sel(res, dims, a.min_n)
    D = len(res["all"])
    for m in a.metric.split(","):
        print(f"\n{m} per depth, circuit {a.circuit}\n")
        print("| bucket | n k1 | " + " | ".join(f"k{k}" for k in range(1, D + 1)) + " | sum |")
        print("|---|---|" + "---|" * (D + 1))
        for b in bks:
            rows = res[b]
            vals = [r.get(m) for r in rows]
            s = sum(v for v in vals if v is not None)
            print(f"| {b} | {rows[0]['n']} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {s:.2f} |")
    print(f"\nexpected accepted length over depths 1..{D} (teacher-forced chain, no p_min stop, no cap below {D})\n")
    print("| bucket | n k1 | Eg (greedy) | Ep (temp 1) | Ep2 (prod-like) | tgt_top1 k1 |")
    print("|---|---|---|---|---|---|")
    for b in bks:
        rows = res[b]
        eg = sum(r["alive"] for r in rows)
        ep = sum(r["cum"] for r in rows)
        ep2 = sum((r.get("cum2") or 0) for r in rows)
        print(f"| {b} | {rows[0]['n']} | {eg:.2f} | {ep:.2f} | {ep2:.2f} | {rows[0]['tgt_top1']:.3f} |")


def diff(a):
    A, B = load(a.a)["results"][a.circuit_a], load(a.b)["results"][a.circuit_b]
    bks = [b for b in sel(A, None, a.min_n) if b in B]
    D = len(A["all"])
    print(f"| bucket | n | " + " | ".join(f"d k{k}" for k in range(1, D + 1)) + " |")
    print("|---|---|" + "---|" * D)
    for b in bks:
        print(f"| {b} | {A[b][0]['n']} | " + " | ".join(f"{B[b][k][a.metric] - A[b][k][a.metric]:+.4f}" for k in range(D)) + " |")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    g = sp.add_parser("gate"); g.add_argument("--eval-long", required=True); g.add_argument("--replay", required=True)
    g.add_argument("--replay-full")
    b = sp.add_parser("buckets"); b.add_argument("replay"); b.add_argument("--circuit", default="M0")
    b.add_argument("--min-n", type=int, default=500); b.add_argument("--metric", default="top1,acc_prob,acc2")
    b.add_argument("--dims")
    d = sp.add_parser("diff"); d.add_argument("a"); d.add_argument("b"); d.add_argument("--circuit-a", default="M0")
    d.add_argument("--circuit-b", default="M0"); d.add_argument("--min-n", type=int, default=500)
    d.add_argument("--metric", default="acc2")
    a = ap.parse_args()
    {"gate": gate, "buckets": buckets, "diff": diff}[a.cmd](a)


if __name__ == "__main__":
    main()
