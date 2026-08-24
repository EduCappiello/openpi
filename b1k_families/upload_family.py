import argparse, pathlib, sys
from huggingface_hub import HfApi

_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_OPENPI_ROOT / "src"))
sys.path.insert(0, str(_OPENPI_ROOT / "b1k_families"))
import openpi.training.config as C
import family_ledger as ledger


DEFAULT_REPO_ID = "0Corvid0/pi05-b1k-families"


def family_folder_name(f):
    return f"{f['name']}_100ep"


def upload_family(name, repo_id=DEFAULT_REPO_ID):
    d = ledger.load()
    f = None
    for fam in d["families"]:
        if fam["name"] == name:
            f = fam
            break
    if f is None:
        print(f"FATAL: unknown family {name!r} — not in family_ledger.json")
        return 1

    cfg_name = f"pi05_b1k_{name}"
    if cfg_name not in C._CONFIGS_DICT:
        print(f"FATAL: config {cfg_name!r} not found (ledger may need a reload)")
        return 1
    cfg = C.get_config(cfg_name)

    final_step = cfg.num_train_steps - 1
    step_dir = cfg.checkpoint_dir / str(final_step)
    if not (step_dir / "params").exists():
        print(f"FATAL: family {name!r} incomplete — {step_dir}/params/ missing. Train first.")
        return 1

    folder = family_folder_name(f)
    api = HfApi()
    repo_exists = False
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        repo_exists = True
    except Exception:
        pass

    if not repo_exists:
        print(f"Creating model repo {repo_id}...")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    # Upload params/ and assets/ (exclude train_state/ to match existing schema)
    for subdir in ["params", "assets"]:
        src = step_dir / subdir
        if not src.exists():
            print(f"  skipping {subdir}/ — not present at {src}")
            continue
        size_gb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e9
        print(f"Uploading {folder}/{subdir}/ (~{size_gb:.1f} GB) to {repo_id}...")
        api.upload_folder(
            folder_path=str(src),
            repo_id=repo_id,
            repo_type="model",
            path_in_repo=f"{folder}/{subdir}",
        )

    # Update repo-root README with this family's row
    readme_md = None
    try:
        readme_bytes = api.hf_hub_download(
            repo_id=repo_id, filename="README.md", repo_type="model"
        )
        readme_path = pathlib.Path(readme_bytes)
        if readme_path.exists():
            readme_md = readme_path.read_text()
    except Exception:
        pass

    if readme_md is None or "## Family Folders" not in readme_md:
        table_header = "\n\n## Family Folders\n\n| Folder | Tasks | Demos/task | Steps |\n|---|---|---|---|\n"
        if readme_md is None:
            readme_md = "---\nlicense: other\n---\n\n# pi05-b1k-family-checkpoints\n" + table_header

    row_line = f"| `{folder}/` | {len(f['task_ids'])} | {f['demo_lo']}-{f['demo_hi']} | {final_step} |\n"
    if row_line.strip() not in readme_md:
        readme_md += row_line

    import tempfile
    tmp_path = pathlib.Path(tempfile.mkdtemp()) / "README.md"
    tmp_path.write_text(readme_md)
    api.upload_file(
        path_or_fileobj=str(tmp_path),
        repo_id=repo_id,
        repo_type="model",
        path_in_repo="README.md",
    )

    steps_url = f"https://huggingface.co/{repo_id}/blob/main/{folder}"
    print(f"OK: {name} uploaded as '{folder}' → {steps_url}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Upload a completed family checkpoint to a HF model repo.")
    ap.add_argument("family_name", help='Family name from family_ledger.json (e.g. backbone_foundation)')
    ap.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                    help=f"Target HF model repo (default: {DEFAULT_REPO_ID}). "
                         "Point at your gated repo e.g. 0Corvid0/pi05-b1k-families for owner-only storage.")
    args = ap.parse_args()
    sys.exit(upload_family(args.family_name, repo_id=args.repo_id))


if __name__ == "__main__":
    main()
