"""
run_all.py
Menjalankan semua script analisis secara berurutan.
Output disimpan di logs/analysis_outputs/.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "analysis"
OUTPUT_DIR = ROOT / "logs" / "analysis_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCRIPTS = [
    "01_weather_statistics.py",
    "02_iv_validation_stats.py",
    "03_dataset_statistics.py",
    "04_metadata_verification.py",
    "05_physics_consistency.py",
    "06_split_verification.py",
    "07_baseline_metrics.py",
    "08_pinn_status.py",
    "09_baseline_vs_hybrid.py",
    "10_latency_analysis.py",
    "11_latency_vs_severity.py",
    "12_threshold_safety.py",
    "13_per_class_analysis.py",
    "14_control_log_verification.py",
    "15_environment_verification.py",
]

print("=" * 60)
print("  MENJALANKAN SEMUA SCRIPT ANALISIS")
print("=" * 60)

for script in SCRIPTS:
    script_path = ANALYSIS_DIR / script
    if not script_path.exists():
        print(f"\n[SKIP] {script} tidak ditemukan")
        continue

    print(f"\n{'='*60}")
    print(f"  Menjalankan: {script}")
    print(f"{'='*60}")

    output_name = script.replace('.py', '.txt')
    output_path = OUTPUT_DIR / output_name

    result = subprocess.run(
        [sys.executable, str(script_path)],
        capture_output=True, text=True, cwd=str(ROOT))

    with open(output_path, 'w') as f:
        f.write(result.stdout)
        if result.stderr:
            f.write("\n\n=== STDERR ===\n")
            f.write(result.stderr)

    print(f"Output disimpan: {output_path}")
    if result.returncode != 0:
        print(f"[ERROR] Script gagal: {script}")
        print(result.stderr[:500])

print(f"\n{'='*60}")
print(f"  SEMUA SCRIPT SELESAI")
print(f"  Output: {OUTPUT_DIR}")
print(f"{'='*60}")