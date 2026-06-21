"""
run_simulation.py
Batch run V2V routing simulations across multiple packet loss conditions.

Usage:
    python run_simulation.py --config njnyc.sumocfg
                             --packet_loss 0 0.2 0.4 0.6
                             --n_runs 3
"""

import argparse
import json
import statistics
from v2v_routing import run_simulation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',       default='njnyc.sumocfg')
    parser.add_argument('--packet_loss',  type=float, nargs='+',
                        default=[0.0, 0.2, 0.4, 0.6])
    parser.add_argument('--n_runs',       type=int, default=3)
    parser.add_argument('--output',       default='results/simulation_results.json')
    args = parser.parse_args()

    all_results = {}

    for pl in args.packet_loss:
        print(f"\nPacket loss = {pl*100:.0f}%")
        run_stats = []

        for run in range(args.n_runs):
            print(f"  Run {run+1}/{args.n_runs}...")
            stats = run_simulation(args.config, packet_loss=pl)
            run_stats.append(stats)
            print(f"    mean={stats['mean']:.1f}s  n={stats['n']}")

        means = [s['mean'] for s in run_stats]
        all_results[f"packet_loss_{pl}"] = {
            'packet_loss':    pl,
            'mean_journey':   statistics.mean(means),
            'std_journey':    statistics.stdev(means) if len(means) > 1 else 0,
            'individual_runs': run_stats,
        }

    import os
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {args.output}")
    print("\nSUMMARY (mean journey time in seconds):")
    print(f"{'Packet Loss':>15}  {'Mean (s)':>10}  {'Std':>8}")
    print("-" * 38)
    for key, res in all_results.items():
        print(f"  {res['packet_loss']*100:>12.0f}%  "
              f"{res['mean_journey']:>10.1f}  "
              f"{res['std_journey']:>8.1f}")


if __name__ == '__main__':
    main()
