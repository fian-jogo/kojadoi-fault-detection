"""
08_pinn_status.py
Konfirmasi status PINN dari ablation_results.json.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ABLATION_JSON = ROOT / "logs" / "ablation_results.json"

print("=" * 60)
print("  08. KONFIRMASI STATUS PINN")
print("=" * 60)

if not ABLATION_JSON.exists():
    print(f"[ERROR] File tidak ditemukan: {ABLATION_JSON}")
    sys.exit(1)

with open(ABLATION_JSON) as f:
    results = json.load(f)

print(f"Total konfigurasi: {len(results)}")

print(f"\n=== Semua Konfigurasi ===")
print(f"{'Nama':<25}{'F1':>8}{'Acc':>8}{'Rec':>8}{'Top1Loc':>10}{'IoU':>8}")
print("-" * 70)
for r in results:
    t = r['test']
    loc = r['localization']
    print(f"{r['name']:<25}{t['f1']:>8.4f}{t['accuracy']:>8.4f}"
          f"{t['recall']:>8.4f}{loc.get('top1_accuracy',0):>10.4f}"
          f"{loc.get('mean_iou',0):>8.4f}")

print(f"\n=== Konfirmasi Pasangan PINN vs non-PINN ===")
pairs = [
    ('LSTM+GAT', 'LSTM+GAT+PINN', '100%'),
    ('LSTM+GAT_50pct', 'LSTM+GAT+PINN_50pct', '50%'),
    ('LSTM+GAT_20pct', 'LSTM+GAT+PINN_20pct', '20%'),
]
for base_name, pinn_name, frac in pairs:
    base = next((r for r in results if r['name'] == base_name), None)
    pinn = next((r for r in results if r['name'] == pinn_name), None)
    if not base or not pinn:
        continue
    delta_f1 = pinn['test']['f1'] - base['test']['f1']
    delta_loc = (pinn['localization'].get('top1_accuracy', 0)
                 - base['localization'].get('top1_accuracy', 0))
    delta_iou = (pinn['localization'].get('mean_iou', 0)
                 - base['localization'].get('mean_iou', 0))
    print(f"\n  Fraksi {frac}:")
    print(f"    F1  : {base['test']['f1']:.4f} -> {pinn['test']['f1']:.4f} "
          f"(delta={delta_f1:+.4f})")
    print(f"    Loc : {base['localization']['top1_accuracy']:.4f} -> "
          f"{pinn['localization']['top1_accuracy']:.4f} "
          f"(delta={delta_loc:+.4f})")
    print(f"    IoU : {base['localization']['mean_iou']:.4f} -> "
          f"{pinn['localization']['mean_iou']:.4f} "
          f"(delta={delta_iou:+.4f})")
    if delta_f1 > 0.005:
        status = "POSITIF (PINN membantu)"
    elif delta_f1 > -0.005:
        status = "NETRAL (PINN setara)"
    else:
        status = "NEGATIF (PINN merugikan)"
    print(f"    Status: {status}")

print(f"\n=== Kesimpulan PINN ===")
pinn_100 = next((r for r in results if r['name'] == 'LSTM+GAT+PINN'), None)
gat_100 = next((r for r in results if r['name'] == 'LSTM+GAT'), None)
if pinn_100 and gat_100:
    print(f"Pada 100% data:")
    print(f"  F1  : {pinn_100['test']['f1']:.4f} "
          f"(vs GAT {gat_100['test']['f1']:.4f})")
    print(f"  IoU : {pinn_100['localization']['mean_iou']:.4f} "
          f"(vs GAT {gat_100['localization']['mean_iou']:.4f})")

pinn_50 = next((r for r in results if r['name'] == 'LSTM+GAT+PINN_50pct'), None)
gat_50 = next((r for r in results if r['name'] == 'LSTM+GAT_50pct'), None)
if pinn_50 and gat_50:
    print(f"\nPada 50% data:")
    print(f"  F1  : {pinn_50['test']['f1']:.4f} "
          f"(vs GAT {gat_50['test']['f1']:.4f})")
    if pinn_50['test']['f1'] > gat_50['test']['f1']:
        print(f"  -> PINN MENGALAHKAN GAT pada regime data menengah")

pinn_20 = next((r for r in results if r['name'] == 'LSTM+GAT+PINN_20pct'), None)
gat_20 = next((r for r in results if r['name'] == 'LSTM+GAT_20pct'), None)
if pinn_20 and gat_20:
    print(f"\nPada 20% data:")
    print(f"  F1  : {pinn_20['test']['f1']:.4f} "
          f"(vs GAT {gat_20['test']['f1']:.4f})")