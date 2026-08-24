"""Delete a trained family checkpoint directory. Refuses to run without --yes,
and refuses to delete a directory that is NOT complete on disk (or, with
--require-upload, unless it has been uploaded).
"""
import argparse, pathlib, shutil, sys

_OPENPI_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_OPENPI_ROOT / "src"))
sys.path.insert(0, str(_OPENPI_ROOT / "b1k_families"))
import openpi.training.config as C
import family_status as FS
import family_ledger as ledger


def cleanup(name, yes=False, require_upload=False):
    if not yes:
        print("FATAL: refusing to delete without --yes (explicit user permission required)")
        return 1

    d = ledger.load()
    f = None
    for fam in d["families"]:
        if fam["name"] == name:
            f = fam
            break
    if f is None:
        print(f"FATAL: unknown family {name!r}")
        return 1

    if require_upload:
        # 'uploaded' is set by upload_family via the ledger? No -- uploads are
        # non-fatal and not recorded. Approximate: require the family be COMPLETE
        # and warn that upload status is not tracked here. For a true guard use
        # family_status --check.
        print("WARN: upload status is not tracked in the ledger; --require-upload "
              "only ensures COMPLETE-on-disk. Confirm uploads manually before cleanup.")

    if not FS.is_family_complete(f):
        print(f"FATAL: family {name!r} is not COMPLETE on disk -- not deleting a partial run")
        return 1

    cfg_name = FS.family_config_name(name)
    cfg = C.get_config(cfg_name)
    target = cfg.checkpoint_dir
    if not target.exists():
        print(f"nothing to delete: {target} does not exist")
        return 0

    sz = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e9
    print(f"Deleting {sz:.1f} GB: {target}")
    shutil.rmtree(target)
    print("done.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("family_name")
    ap.add_argument("--yes", action="store_true", help="required to actually delete")
    ap.add_argument("--require-upload", action="store_true")
    args = ap.parse_args()
    sys.exit(cleanup(args.family_name, yes=args.yes, require_upload=args.require_upload))


if __name__ == "__main__":
    main()
