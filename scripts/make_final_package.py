"""Assemble the official final package (run on the machine that holds the outputs).

<team>_submission.zip
├── output/{matching_results.tsv, candidate_pairs.tsv}
├── code/business_entity_resolution/{src/er/*.py, README.md, requirements.txt, models/...}
└── Documentation_template.md   (filled-in methodology)

  python3 scripts/make_final_package.py --team Team_AMS --out-dir output_v4 --version v4 \
      --model artifacts/train/model_hyb5.txt artifacts/train/stage2_hyb5_ce.txt --rename model.txt stage2.txt \
      --encoder models/e5er_fp16 models/ce_fp16
"""
import argparse
import zipfile
from pathlib import Path

MODULES = ["__init__", "normalize", "prep", "blocking", "biencoder", "candidates", "features", "add_ambiguity",
           "decide", "predict", "train_full", "cross_encoder", "stage2", "exp_stage2", "predict_stage2",
           "exp_band", "exp_band_robust", "exp_band_fd", "exp_band_rival", "build_competitors", "build_extra_band",
           "export_band_pairs", "export_ce_pairs", "export_test_band_texts", "kaggle_train_ce", "redecide",
           "export_rival_band", "adapt_unseen_ce", "demote_swaps"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--team", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--model", nargs="*", default=[], help="optional model files shipped under models/ (not required by the rules)")
    ap.add_argument("--rename", nargs="*", default=None, help="target names for --model files")
    ap.add_argument("--encoder", nargs="*", default=[], help="transformer weight dirs to ship under models/")
    ap.add_argument("--dest", default=".")
    ap.add_argument("--doc", default="package/Documentation.md", help="filled-in methodology document")
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    zpath = Path(a.dest) / f"{a.team}_submission.zip"
    code = "code/business_entity_resolution"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in ("matching_results.tsv", "candidate_pairs.tsv"):
            z.write(Path(a.out_dir) / f, f"output/{f}")
        for m in MODULES:
            z.write(root / "src/er" / f"{m}.py", f"{code}/src/er/{m}.py")
        z.write(root / f"package/README_{a.version}.md", f"{code}/README.md")
        req = root / f"package/requirements_{a.version}.txt"
        z.write(req if req.exists() else root / "package/requirements.txt", f"{code}/requirements.txt")
        for i, m in enumerate(a.model):
            name = a.rename[i] if a.rename else Path(m).name  # noqa
            z.write(m, f"{code}/models/{name}")
        for enc in a.encoder:
            for p in sorted(Path(enc).rglob("*")):
                if p.is_file():
                    z.write(p, f"{code}/models/{Path(enc).name}/{p.relative_to(enc)}")
        z.write(root / a.doc, "Documentation_template.md")
    with zipfile.ZipFile(zpath) as z:
        bad = z.testzip()
        n = len(z.namelist())
    print(f"wrote {zpath} ({zpath.stat().st_size / 1e6:.0f} MB, {n} files, integrity {'OK' if bad is None else bad})")


if __name__ == "__main__":
    main()
