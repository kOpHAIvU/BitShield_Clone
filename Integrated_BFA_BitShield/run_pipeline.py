import os
import argparse
import torch
from data_iotid20 import build_iotid20_loaders
from model_iotid20 import MLPClassifier
from custom_models import CustomModel1, CustomModel2
from attack_bfa_random import RandomBitFlip
from defense_bitshield import wrap_with_dig, calc_dig_range

@torch.no_grad()
def evaluate(model, loader, device='cpu'):
    model.eval()
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

def select_model(name, in_features, n_classes):
    if name == 'mlp':
        return MLPClassifier(in_features, n_classes)
    if name == 'custom1':
        return CustomModel1(input_size=in_features, output_size=n_classes)
    if name == 'custom2':
        return CustomModel2(input_size=in_features, output_size=n_classes)
    raise ValueError('Unknown model: ' + name)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', type=str, required=True)
    ap.add_argument('--weights', type=str, required=True)
    ap.add_argument('--bitshield-root', type=str, required=True)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--nflips', type=int, default=10)
    ap.add_argument('--model', type=str, default='mlp', choices=['mlp', 'custom1', 'custom2'])
    ap.add_argument('--source-csv', type=str, default=None)
    ap.add_argument('--download-url', type=str, default=None)
    ap.add_argument('--results-dir', type=str, default='results')
    ap.add_argument('--enable-dig', action='store_true')
    ap.add_argument('--disable-dig', dest='enable_dig', action='store_false')
    ap.set_defaults(enable_dig=True)
    args = ap.parse_args()

    train_loader, test_loader, in_features, n_classes = build_iotid20_loaders(args.data_root, source_csv=args.source_csv, download_url=args.download_url)
    model = select_model(args.model, in_features, n_classes)
    state = torch.load(args.weights, map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model.to(args.device)

    # Results setup
    os.makedirs(args.results_dir, exist_ok=True)
    import csv, json
    summary_rows = []

    print('=' * 60)
    print(f'Model: {args.model} | Device: {args.device} | DIG: {"ON" if args.enable_dig else "OFF"}')
    print('-' * 60)

    base_acc = evaluate(model, test_loader, args.device)
    print(f'[Baseline] Accuracy: {base_acc:.2f}%')
    summary_rows.append({'phase': 'baseline', 'strength': '', 'nflips': 0, 'accuracy': round(base_acc, 4), 'drop': 0.0, 'dig_detect_rate': ''})

    # Prepare DIG (compute clean suspicious score range BEFORE attacks)
    # Ensure DIG uses final classifier head explicitly
    model_fc = getattr(model, 'classifier', None)
    dig_range = [0.0, 0.0]
    if args.enable_dig:
        protected_clean = wrap_with_dig(model, args.bitshield_root, model_fc=model_fc)
        dig_range = calc_dig_range(protected_clean, train_loader, args.device, n_batches=20)
        print(f'[DIG] Clean suspicious score range: [{dig_range[0]:.2f}, {dig_range[1]:.2f}]')

    # Attack options: random bit flips + strength-based gaussian perturbation like BitShield
    # 1) Random bit flips on Linear/Conv1d/CustomBlock
    attacker = RandomBitFlip(model)
    flip_logs = []
    for i in range(args.nflips):
        info = attacker.flip_one_bit()
        if info:
            flip_logs.append(info)
    attacked_acc = evaluate(model, test_loader, args.device)
    print(f'[BitFlip x{len(flip_logs)}] Accuracy: {attacked_acc:.2f}%   Drop: {base_acc - attacked_acc:.2f}%')
    summary_rows.append({'phase': 'bitflip', 'strength': '', 'nflips': len(flip_logs), 'accuracy': round(attacked_acc, 4), 'drop': round(base_acc - attacked_acc, 4), 'dig_detect_rate': ''})

    # 2) Strength-based gaussian perturbation sweep
    strengths = [0.1, 0.2, 0.5, 1.0]
    for s in strengths:
        with torch.no_grad():
            for p in model.parameters():
                p.add_(torch.randn_like(p) * s)
        acc_after = evaluate(model, test_loader, args.device)
        print(f'[Strength {s}] Accuracy: {acc_after:.2f}%   Drop: {base_acc - acc_after:.2f}%')
        summary_rows.append({'phase': 'gaussian', 'strength': s, 'nflips': '', 'accuracy': round(acc_after, 4), 'drop': round(base_acc - acc_after, 4), 'dig_detect_rate': ''})

    # Defense: DIG (evaluate detection AFTER attacks using clean range)
    if args.enable_dig:
        protected = wrap_with_dig(model, args.bitshield_root, model_fc=model_fc)
        detected = 0
        total = 0
        protected.to(args.device).eval()
        for x, _ in test_loader:
            x = x.to(args.device)
            x.requires_grad_(True)
            try:
                s = protected.calc_sus_score(x).item()
                if not (dig_range[0] <= s <= dig_range[1]):
                    detected += 1
            except Exception:
                pass
            x.requires_grad_(False)
            total += 1
        dig_rate = 100.0 * detected / max(total, 1)
        print(f'[DIG] Detection rate (post-attack): {dig_rate:.2f}%')
        if summary_rows:
            summary_rows[-1]['dig_detect_rate'] = round(dig_rate, 4)

    # Save results
    csv_file = os.path.join(args.results_dir, f'summary_{args.model}.csv')
    json_file = os.path.join(args.results_dir, f'summary_{args.model}.json')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['phase','strength','nflips','accuracy','drop','dig_detect_rate'])
        writer.writeheader()
        writer.writerows(summary_rows)
    with open(json_file, 'w') as f:
        json.dump(summary_rows, f, indent=2)

    print('-' * 60)
    print(f'Saved results to: {csv_file} and {json_file}')
    print('=' * 60)


if __name__ == '__main__':
    main()

