"""Task-vector merges of three v3 specialists fine-tuned from one shared base (docs/merge_categories_protocol_v3.md):
Task Arithmetic (base + lam * sum of task vectors) or TIES-Merging (trim, elect sign, disjoint mean; base + lam * merged
vector). Every tensor (extractor, actor, critic) is merged by the same rule and the result is saved as a benchmark-v3
(unconditioned) checkpoint bundle, like arm E's merge; train.py --init_mode full then pads the preference columns with
zeros.

    python tools/merge_task_vectors_v3.py --method ta|ties --base base.pth --sources m.pth c.pth t.pth --lam 0.5 \\
        --out merged.pth [--keep 0.2]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "daniel"))
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--method", required=True, choices=["ta", "ties"])
ap.add_argument("--base", required=True)
ap.add_argument("--sources", nargs=3, required=True)
ap.add_argument("--lam", type=float, required=True)
ap.add_argument("--keep", type=float, default=0.2, help="TIES: fraction of each task vector kept by magnitude")
ap.add_argument("--out", required=True)
a = ap.parse_args()
sys.argv = sys.argv[:1]

import torch                                                # noqa: E402
from checkpointing import unwrap_checkpoint, validate_checkpoint_schema, file_sha256   # noqa: E402
from feature_schemas import SCHEMA_V3                       # noqa: E402
from warmstart import task_arithmetic, ties_merge           # noqa: E402

LABEL = {"ta": "Task Arithmetic (Ilharco et al., ICLR 2023)", "ties": "TIES-Merging (Yadav et al., NeurIPS 2023)"}


def load(path):
    sd, meta = unwrap_checkpoint(torch.load(path, map_location="cpu", weights_only=False))
    validate_checkpoint_schema(meta, SCHEMA_V3, sd)
    return sd, meta


def main():
    base, _ = load(a.base)
    srcs = [load(s) for s in a.sources]
    sds = [sd for sd, _ in srcs]
    bundle = dict(srcs[0][1])
    bundle["state_dict"] = (task_arithmetic(base, sds, a.lam) if a.method == "ta" else
                            ties_merge(base, sds, a.lam, a.keep))
    bundle["update"] = 0
    bundle["validation_metrics"] = {}
    bundle["selection_metric"] = float("nan")
    bundle["parents"] = ([{"path": str(Path(a.base).resolve()), "sha256": file_sha256(a.base), "role": "base"}] +
                         [{"path": str(Path(s).resolve()), "sha256": file_sha256(s), "role": "specialist"}
                          for s in a.sources])
    bundle["merge"] = {"method": a.method, "label": LABEL[a.method], "lambda": a.lam,
                       **({"keep": a.keep} if a.method == "ties" else {})}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(json.dumps({"out": str(out), "method": a.method, "lambda": a.lam, "sources": [Path(s).name for s in a.sources]}))


if __name__ == "__main__":
    main()
