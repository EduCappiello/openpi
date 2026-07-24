import stage_wave as S, os, numpy as np, glob
df=S.meta()
V=S.ROOT/"videos"
mean={s:np.mean([os.path.getsize(f) for f in glob.glob(f"{V}/observation.rgb.{s}/*/*.mp4")])/1e6 for s in S.STREAMS}
overall=np.mean(list(mean.values()))
free=os.statvfs("/").f_bavail*os.statvfs("/").f_frsize/1e9
print(f"disk free now: {free:.0f} GB")
print(f"{'wave':>12} {'eps':>6} {'Mframes':>8} {'new files':>10} {'incr GB':>9}")
for hi in [34,36,38,40,42,46,50]:
    paths,sub=S.wave_files(df,30,hi)
    todo=[p for p in paths if not (S.ROOT/p).exists()]
    gb=sum(overall if p.endswith('.mp4') else 100 for p in todo)/1e3
    print(f"     30..{hi:<3d} {len(sub):>6} {sub.length.sum()/1e6:>8.1f} {len(todo):>10} {gb:>9.0f}")
print("\nheadroom needed: ~13 GB per checkpoint + ~13 GB arrow cache for the new subset")
