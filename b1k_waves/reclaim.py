"""Delete video media not needed by any incomplete wave.

Safe by construction: computes the set of mp4 files required by every wave in the
ledger whose status is not COMPLETE, and deletes anything on disk outside that set.
A wave that is COMPLETE has already been trained on and its checkpoint saved, so its
exclusive media (frames no other pending wave needs) is pure disk cost with no further
use. Run this BEFORE staging the next wave, not after -- freeing space is what makes
the next stage's download fit.
"""
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import stage_wave as S
import ledger


def freeable_files():
    df = S.meta()

    def mp4s(lo, hi):
        paths, _ = S.wave_files(df, lo, hi)
        return {p for p in paths if p.endswith(".mp4")}

    d = ledger.load()
    need = set()
    for w in d["waves"]:
        if "COMPLETE" not in w.get("status", ""):
            need |= mp4s(w["demo_lo"], w["demo_hi"])
    on_disk = {p for p in mp4s(0, ledger.EPISODES_PER_TASK) if (S.ROOT / p).exists()}
    return on_disk - need


def main():
    free_before = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize / 1e9
    victims = freeable_files()
    freed = 0
    for p in victims:
        fp = S.ROOT / p
        freed += fp.stat().st_size
        fp.unlink()
    free_after = os.statvfs("/").f_bavail * os.statvfs("/").f_frsize / 1e9
    print(f"reclaim: deleted {len(victims)} files, freed {freed/1e9:.1f} GB "
          f"(disk free {free_before:.0f} GB -> {free_after:.0f} GB)")


if __name__ == "__main__":
    main()
