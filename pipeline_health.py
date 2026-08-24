#!/usr/bin/env python3
# pipeline_health.py - runs every ~10 mins, logs GPU + process health + wave/family status; restarts the orchestrator if dead.
import subprocess, time, random, os, textwrap

CWD = "/root/BEHAVIOR-1K/b1k-baselines/baselines/openpi"
LOG_PATH = "/root/pipeline_health.log"
PY = os.path.join(CWD, ".venv/bin/python")
TMUX_SESSION = "behavior"


def run(cmd, capture=True, shell=False):
    r = subprocess.run(cmd, cwd=CWD, capture_output=capture, text=capture, shell=shell)
    return (r.returncode, r.stdout or "", r.stderr or "")


def snapshot():
    lines = []

    # timestamp
    from datetime import datetime, timezone
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines.append(f"=== {ts} ===")

    # GPU health: utilization + memory used & free (training healthy ~90 %+ util & 52-70 GB/GPU while training)
    rc, out, err = run("nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.free --format=csv,noheader", shell=True)
    lines.append(out.strip() or "[GPU UNREACHABLE]")

    # key processes counting: orchestrator (+ stage_wave/train.py when relevant)
    rc, out, _ = run(f"pgrep -cf 'run_waves'") if True else (99, "", "")  # force using pgrep
    ow_count = int(out.strip())  if out.strip().isdigit() else 0
    lines.append(f"run-waves-count={ow_count} {'(RUNNING)' if ow_count>0 else '(DEAD/STOPPED)'}")

    rc2, o2,_=run(["pgrep","-f","stage_wave.py"], capture=True); stage_ok="running(stage)"if int(o2.strip()if o2.strip().isdigit()else 0)>0 else "no-stage-wave"
    lines.append(f"stage={stage_ok}")

    rc_train, outt,_=run("""ps auxww | grep scripts/train\.py | grep -vE grep""", shell=True); train_running="RUNNING"if outt.strip().strip()else "NOT_STARTED_YET";lines.append(f"train-status={train_running}")

    # wave_status.py (disk-synced ledger + filesystem fact; trust this)
    rc_waves, wo,_=run([PY,"b1k_waves/wave_status.py"]) if PY else (99,"","");lines.append("\n"+wo.strip()or("[wave-status FAILED]"))

    # next_wave_info: determine DONE or queued wave to run next
    rc_wave_done,wdone_err,_=run([PY,"b1k_waves/next_wave_info.py"]);wd_line=(wdone_err.strip().splitlines()[0])if wdone_err.strip()else "err";waves_complete="YES"if str.strip(wd_line).upper()=="DONE" else "NO"; lines.append(f"WAVES_COMPLETE={waves_complete}")

    # family_status + next_family_info (future-gated pipeline; run_families.sh blocked until ALL waves marked COMPLETE/next_wave==DONE)
    lines.append("---families-preview---")
    rc_fam, fstat_txt,_=run([PY,"b1k_families/family_status.py"]);lines.append((fstat_txt or "[family-status fail]").strip()[:800])

    return "\n".join(lines), waves_complete


def maybe_restart_orchestrator(waves_done):
    # If ALL waves DONE and neither orchestrator alive -> launch families stage inside SAME tmux session `behavior`. Otherwise simply restart run_waves if dead AND not done.
    rc_pw,opo,_=run(["pgrep","-f","run_waves"]) if True else (99,"","")
    pw_alive=int(opo.strip())if opo.strip().isdigit()else 0>0

    # If training already running OR orchestrator alive then do nothing now. Only touch the session when both conditions break AND still something needs to run depending on waves_completed status: if families not started yet and done->launch family_experts.sh in behavior tmux; else just restart run_waves again until DONE.
    cmd=""
    if pw_alive==False: # dead/missing: decide what to start now based on completion state of wave chain vs current situation with families/gate conditions for family stage launch readiness logic (depends solely off next_wave_info being literally DONE string): waves_done=='YES' triggers starting run_family_experts.sh instead in same session name behavior; else only restart the original waves loop again via tmux new-session(-d -s name) or if stuck/locked/corrupt -> kill existing session named behavior and re-create cleanly for new command execution.
        rc_pwfam,_oFamErr,_=run(["tmux","has","-t",TMUX_SESSION]);fam_alive=("YES")if int(rc_pwfam)==0 else ("NO"); if fam_alive=="YES": tmux_cmd_kill="-kill";cmd=f"tmux {tmux_cmd_kill} -t '{TMUX_SESSION}' >/dev/null 2>/dev/null; sleep 1s ; export B1K_STAGE_WORKERS=8 && cd {CWD} && {'bash b1k_families/run_family_experts.sh' if waves_done else ('export HF_HUB_OFFLINE=0 && rm -f b1k_waves/run_waves.lock >/dev/null; } bash run_waves.sh 2>&1 | tee ~/train_pipeline.log)'}"; subprocess.run(['tmux','new-session','-d',"-s",TMUX_SESSION,cmd], cwd=CWD,text=True,capture=False);return 

def loop():
    with open(LOG_PATH,"a")as fh: # open append at start per run to avoid race conditions on log writes after backgrounding
        while True:
            snap_txt,waves_complete=snapshot();fh.write(snap_txt+"\n---\n");fh.flush() 
            maybe_re...restart_orchestrator(waves_complete);time.sleep(random.randint(60,90)) # jittered wake ~7 to 10 min avg over many wakes


if __name__=="__main__":
    loop()
