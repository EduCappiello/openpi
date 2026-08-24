import ledger, datetime
d = ledger._blank()
# Arm 0 (arm0_monolithic, 50k steps, the pretrained warm-start base): demos 0..29 per
# task, 0..26 train + 27..29 val. It is NOT retrained in the wave loop -- it is the
# static base that wave1 warm-starts from. In the full-FT clean-slate re-run its
# params are re-fetched from HF by run_waves.sh (fetch_arm0) before wave1 launches.
arm0 = {
    "name": "arm0_monolithic", "demo_lo": 0, "demo_hi": 30, "val_lo": 27,
    "seed": ledger.SEED, "n_per_task": 30, "val_per_task": 3,
    "train_episodes": [ledger.gidx(t, i) for t in range(100) for i in range(0, 27)],
    "val_episodes":   [ledger.gidx(t, i) for t in range(100) for i in range(27, 30)],
    "created": "2026-07-20T10:00:00+00:00",
    "status": "COMPLETE (step 49999)",
    "checkpoint": "ecappiell-as/pi05-b1k-arm0-monolithic",
    "note": "Warm-start base for wave1; params fetched from HF on demand (run_waves.sh).",
}
ledger.commit(arm0, d)
d = ledger.load()
print("ledger:", ledger.LEDGER)
print("waves recorded:", [w["name"] for w in d["waves"]])
print("consumed demos:", f"{min(ledger.consumed(d))}..{max(ledger.consumed(d))}")
print("next wave starts at demo:", ledger.next_lo(d))
print("train eps:", len(d["waves"][0]["train_episodes"]), "val eps:", len(d["waves"][0]["val_episodes"]))
