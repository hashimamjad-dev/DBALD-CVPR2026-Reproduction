"""Shared laptop-GPU thermal guard: pause above 78 C until 70 C, give up after 180 s (progress is always saved first)."""
import subprocess,time

def temperature():
    try: return float(subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
    except Exception: return 0.0

def cool(limit=78.,target=70.,timeout=180.):
    start=time.perf_counter()
    if temperature()<limit: return 0.
    while temperature()>target:
        if time.perf_counter()-start>timeout: raise RuntimeError('GPU did not cool in time; saved progress is retained')
        time.sleep(5)
    return time.perf_counter()-start
