"""Arm E (and E-shared): uniform average of three v3 specialists' full state dicts (extractor, actor, critic), saved as
a benchmark-v3 (unconditioned) checkpoint bundle. train.py --init_mode full then pads the preference columns with
zeros (docs/warmstart_protocol_v3.md). Label: "naive merge of independently trained specialists (no shared base);
expected to be weak" (arm E), unless --shared_base is given (arm E-shared).

    python tools/merge_specialists_v3.py --sources a.pth b.pth c.pth --out merged.pth [--shared_base]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "daniel"))
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("--sources", nargs=3, required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--shared_base", action="store_true")
a = ap.parse_args()
sys.argv = sys.argv[:1]

import torch                                                # noqa: E402
from checkpointing import unwrap_checkpoint, validate_checkpoint_schema, file_sha256   # noqa: E402
from feature_schemas import SCHEMA_V3                       # noqa: E402
from warmstart import uniform_merge                         # noqa: E402

LABEL = {False: "naive merge of independently trained specialists (no shared base); expected to be weak",
         True: "uniform merge of specialists fine-tuned from one shared base (E-shared)"}


def main():
    sds, metas = [], []
    for s in a.sources:
        sd, meta = unwrap_checkpoint(torch.load(s, map_location="cpu", weights_only=False))
        validate_checkpoint_schema(meta, SCHEMA_V3, sd)
        sds.append(sd); metas.append(meta)
    bundle = dict(metas[0])
    bundle["state_dict"] = uniform_merge(sds)
    bundle["update"] = 0
    bundle["validation_metrics"] = {}
    bundle["selection_metric"] = float("nan")
    bundle["parents"] = [{"path": str(Path(s).resolve()), "sha256": file_sha256(s), "weight": 1 / 3} for s in a.sources]
    bundle["merge"] = {"method": "uniform average of full state dicts", "label": LABEL[a.shared_base]}
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(json.dumps({"out": str(out), "label": LABEL[a.shared_base], "sources": [Path(s).name for s in a.sources]}))


if __name__ == "__main__":
    main()
