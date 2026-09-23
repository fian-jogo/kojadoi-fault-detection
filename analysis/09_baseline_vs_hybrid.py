"""
09_baseline_vs_hybrid.py
Perbandingan baseline LSTM vs LSTM+GAT vs LSTM+GAT+PINN.
"""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASELINE_JSON = ROOT / "logs" / "lstm_baseline_metrics.json"
ABLATION_JSON = ROOT / "logs" / "ablation_results.json"

print("=" * 60)
print("  09. PERBANDINGAN BASELINE vs HYBRID")
print("=" * 60)

with open(BASELINE_JSON) as f:
    baseline = json.load(f)
with open(ABLATION_JSON) as f:
    hybrid = json.load(f)

b_f1 = baseline['test']['f1']
b_acc = baseline['test']['accuracy']
b_rec = baseline['test']['recall']
b_prec = baseline['test']['precision']
b_loc = baseline['localization']['top1_accuracy']
b_iou = baseline['localization']['mean_iou']

gat = next(r for r in hybrid if r['name'] == 'LSTM+GAT')
pinn = next(r for r in hybrid if r['name'] == 'LSTM+GAT+PINN')

print(f"\n{'Metrik':<25}{'Baseline':>12}{'LSTM+GAT':>12}{'Hybrid+PINN':>12}")
print("-" * 61)
print(f"{'Test F1':<25}{b_f1:>12.4f}{gat['test']['f1']:>12.4f}{pinn['test']['f1']:>12.4f}")
print(f"{'Test Accuracy':<25}{b_acc:>12.4f}{gat['test']['accuracy']:>12.4f}{pinn['test']['accuracy']:>12.4f}")
print(f"{'Test Precision':<25}{b_prec:>12.4f}{gat['test']['precision']:>12.4f}{pinn['test']['precision']:>12.4f}")
print(f"{'Test Recall':<25}{b_rec:>12.4f}{gat['test']['recall']:>12.4f}{pinn['test']['recall']:>12.4f}")
print(f"{'Top-1 Loc':<25}{b_loc:>12.4f}{gat['localization']['top1_accuracy']:>12.4f}{pinn['localization']['top1_accuracy']:>12.4f}")
print(f"{'Top-3 Loc':<25}{'--':>12}{gat['localization']['top3_accuracy']:>12.4f}{pinn['localization']['top3_accuracy']:>12.4f}")
print(f"{'Mean IoU':<25}{b_iou:>12.4f}{gat['localization']['mean_iou']:>12.4f}{pinn['localization']['mean_iou']:>12.4f}")
print(f"{'Parameter':<25}{baseline['config']['n_params']:>12,}{gat['n_params']:>12,}{pinn['n_params']:>12,}")

print(f"\n=== Kontribusi GAT ===")
delta_f1 = gat['test']['f1'] - b_f1
delta_loc = gat['localization']['top1_accuracy'] - b_loc
delta_iou = gat['localization']['mean_iou'] - b_iou
delta_params = gat['n_params'] - baseline['config']['n_params']
print(f"  Delta F1      : {delta_f1:+.4f}  ({100*delta_f1/b_f1:+.1f}%)")
print(f"  Delta Loc     : {delta_loc:+.4f}  ({100*delta_loc/b_loc:+.1f}%)")
print(f"  Delta IoU     : {delta_iou:+.4f}")
print(f"  Delta Params  : {delta_params:+,} ({100*delta_params/baseline['config']['n_params']:+.1f}%)")

print(f"\n=== Kontribusi PINN ===")
delta_f1_p = pinn['test']['f1'] - gat['test']['f1']
delta_loc_p = pinn['localization']['top1_accuracy'] - gat['localization']['top1_accuracy']
delta_iou_p = pinn['localization']['mean_iou'] - gat['localization']['mean_iou']
print(f"  Delta F1      : {delta_f1_p:+.4f}")
print(f"  Delta Loc     : {delta_loc_p:+.4f}")
print(f"  Delta IoU     : {delta_iou_p:+.4f}")