"""
one command for the whole study.
  python run_all.py --input Combined.csv            # everything
  python run_all.py --input Combined.csv --skip-shap
Each stage's console output is also saved in logs/.
"""
import argparse, subprocess, sys, time
import config as C

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--skip-shap", action="store_true")
a = ap.parse_args()
py = sys.executable
stages = [
    ("data_prep", [py, "data_prep.py", "--input", a.input]),
    ("train", [py, "-u", "train_autoencoder.py"]),
    ("evaluate", [py, "evaluate_and_report.py"]),
    ("analysis", [py, "analysis_figures.py"]),
    ("attribution", [py, "explain_attribution.py"]),
    ("latency", [py, "measure_latency.py"]),
]
if not a.skip_shap:
    stages.append(("kernelshap", [py, "-u", "explain_kernelshap.py"]))
for name, cmd in stages:
    t = time.time(); print(f"\n===== {name} =====", flush=True)
    with open(f"{C.LOG_DIR}/{name}.log", "w") as lf:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            sys.stdout.write(line); lf.write(line)
        p.wait()
    if p.returncode:
        sys.exit(f"Stage '{name}' failed (see logs/{name}.log)")
    print(f"----- {name} finished in {time.time()-t:.0f}s")
print("\nALL DONE. Results are in the 'results' folder.")
