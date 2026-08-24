import argparse, pathlib, sys
from huggingface_hub import HfApi

_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_OPENPI_ROOT / "src"))
sys.path.insert(0, str(_OPENPI_ROOT / "b1k_waves"))
import openpi.training.config as C
import ledger


REPO_ID = "0Corvid0/pi05-b1k-waves"


def wave_folder_name(w):
    return f"{w['name']}_{w['demo_hi']}ep"


def upload_wave(name):
    d = ledger.load()
    w = None
    for wave in d["waves"]:
        if wave["name"] == name:
            w = wave
            break
    if w is None:
        print(f"FATAL: unknown wave {name!r} — not in episode_ledger.json")
        return 1

    cfg_name = f"pi05_b1k_{name}"
    if cfg_name not in C._CONFIGS_DICT:
        print(f"FATAL: config {cfg_name!r} not found (ledger may need a reload)")
        return 1
    cfg = C.get_config(cfg_name)

    final_step = cfg.num_train_steps - 1
    step_dir = cfg.checkpoint_dir / str(final_step)
    if not (step_dir / "params").exists():
        print(f"FATAL: wave {name!r} incomplete — {step_dir}/params/ missing. Train first.")
        return 1

    folder = wave_folder_name(w)
    api = HfApi()
    repo_exists = False
    try:
        api.repo_info(repo_id=REPO_ID, repo_type="model")
        repo_exists = True
    except Exception:
        pass

    if not repo_exists:
        print(f"Creating model repo {REPO_ID}...")
        api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    # Upload params/ and assets/ (exclude train_state/ to match existing schema)
    for subdir in ["params", "assets"]:
        src = step_dir / subdir
        if not src.exists():
            print(f"  skipping {subdir}/ — not present at {src}")
            continue
        size_gb = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / 1e9
        print(f"Uploading {folder}/{subdir}/ (~{size_gb:.1f} GB) to {REPO_ID}...")
        api.upload_folder(
            folder_path=str(src),
            repo_id=REPO_ID,
            repo_type="model",
            path_in_repo=f"{folder}/{subdir}",
        )

    # Update repo-root README with this wave's row
    readme_md = None
    try:
        readme_bytes = api.hf_hub_download(
            repo_id=REPO_ID, filename="README.md", repo_type="model"
        )
        readme_path = pathlib.Path(readme_bytes)
        if readme_path.exists():
            readme_md = readme_path.read_text()
    except Exception:
        pass

    if readme_md is None or "## Wave Folders" not in readme_md:
        now_table_header = (
            "\n\n## Wave Folders\n\n| Folder | Demos/task | Steps |\n|---|---|---|\n"
        )
        if readme_md is None:
            readme_md = "---\nlicense: other\n---\n\n# pi05-b1k-monolithic-model\n" + now_table_header

    folder_markdown = f"{folder}/"
    row_line = f"| `{folder_markdown}` | {w['demo_lo']}\u2013{w['demo_hi']} | {final_step} |\n"
    if row_line.strip() not in readme_md:
        readme_md += row_line

    import tempfile
    tmp_path = pathlib.Path(tempfile.mkdtemp()) / "README.md"
    tmp_path.write_text(readme_md)
    api.upload_file(
        path_or_fileobj=str(tmp_path),
        repo_id=REPO_ID,
        repo_type="model",
        path_in_repo="README.md",
    )

    steps_url = f"https://huggingface.co/{REPO_ID}/blob/main/{folder}"
    print(f"OK: {name} uploaded as '{folder}' → {steps_url}")
    return 0


def main():
    ap = argparse.ArgumentParser(description="Upload a completed wave checkpoint to the unified HF repo.")
    ap.add_argument("wave_name", help='Wave name from episode_ledger.json (e.g. wave6_d70_80)')
    args = ap.parse_args()
    sys.exit(upload_wave(args.wave_name))


if __name__ == "__main__":
    main()
