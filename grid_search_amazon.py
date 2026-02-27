#!/usr/bin/env python3
"""
Grid Search for Amazon Target Domain.
Find which hyperparameters are most sensitive for Amazon target domain.
"""
import os
import json
import subprocess
import time
from itertools import product
from datetime import datetime
from pathlib import Path


class GridSearch:
    """Grid search for hyperparameter tuning."""

    def __init__(self, base_output_dir="./grid_search_amazon"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        self.results_file = self.base_output_dir / "results.json"
        self.results = []

        # Load existing results if any
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                self.results = json.load(f)

    def parse_final_accuracy(self, log_path):
        """Parse final accuracy from log file."""
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            # Find the final summary line
            for line in content.split('\n')[::-1]:
                if 'Final:' in line and ('Peak accuracy:' in line or 'Peak:' in line):
                    # Extract final accuracy
                    parts = line.split('Final:')[1].split('%')[0].strip()
                    return float(parts)
            return None
        except Exception as e:
            print(f"Error parsing log: {e}")
            return None

    def run_experiment(self, source, target, params, exp_name):
        """Run a single experiment."""
        output_dir = self.base_output_dir / exp_name
        output_dir.mkdir(exist_ok=True)

        cmd = [
            "python", "main.py",
            "--source", source,
            "--target", target,
            "--output_dir", str(output_dir),
            "--target_epochs", str(params.get("target_epochs", 20)),
            "--target_batch_size", str(params.get("target_batch_size", 64)),
            "--buffer_capacity", str(params.get("buffer_capacity", 300)),
        ]

        # Add optional hyperparameters
        if "proto_temp" in params:
            cmd.extend(["--proto_temp", str(params["proto_temp"])])
        if "proto_ema" in params:
            cmd.extend(["--proto_ema", str(params["proto_ema"])])
        if "reliability_threshold" in params:
            cmd.extend(["--reliability_threshold", str(params["reliability_threshold"])])
        if "class_detect_alpha" in params:
            cmd.extend(["--class_detect_alpha", str(params["class_detect_alpha"])])
        if "lambda_pl" in params:
            cmd.extend(["--lambda_pl", str(params["lambda_pl"])])
        if "lambda_proto" in params:
            cmd.extend(["--lambda_proto", str(params["lambda_proto"])])
        if "lambda_dist" in params:
            cmd.extend(["--lambda_dist", str(params["lambda_dist"])])

        print(f"\n{'='*60}")
        print(f"Running: {exp_name}")
        print(f"Params: {params}")
        print(f"{'='*60}")

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                cwd="/root/code4",
                capture_output=True,
                text=True,
                timeout=1800  # 30 min timeout
            )

            elapsed = time.time() - start_time

            log_path = output_dir / "e3p_run.log"
            final_acc = self.parse_final_accuracy(log_path)

            result_data = {
                "exp_name": exp_name,
                "params": params,
                "final_acc": final_acc,
                "elapsed_time": elapsed,
                "success": final_acc is not None,
                "timestamp": datetime.now().isoformat()
            }

            self.results.append(result_data)
            self.save_results()

            print(f"✓ Completed: {exp_name}")
            print(f"  Final Accuracy: {final_acc:.2f}%" if final_acc else "  Failed to parse accuracy")
            print(f"  Time: {elapsed/60:.1f}min")

            return result_data

        except subprocess.TimeoutExpired:
            print(f"✗ Timeout: {exp_name}")
            return None
        except Exception as e:
            print(f"✗ Error: {exp_name} - {e}")
            return None

    def save_results(self):
        """Save results to JSON."""
        with open(self.results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

    def print_summary(self):
        """Print summary of all results."""
        print(f"\n{'='*60}")
        print("GRID SEARCH SUMMARY")
        print(f"{'='*60}")

        if not self.results:
            print("No results yet.")
            return

        # Sort by final accuracy
        sorted_results = sorted(
            [r for r in self.results if r['final_acc'] is not None],
            key=lambda x: x['final_acc'],
            reverse=True
        )

        print(f"\nTop 5 configurations:")
        for i, r in enumerate(sorted_results[:5]):
            print(f"  {i+1}. {r['exp_name']}: {r['final_acc']:.2f}%")
            print(f"     Params: {r['params']}")

        # Sensitivity analysis per parameter
        self._analyze_sensitivity()

    def _analyze_sensitivity(self):
        """Analyze sensitivity of each parameter."""
        print(f"\n{'='*60}")
        print("PARAMETER SENSITIVITY ANALYSIS")
        print(f"{'='*60}")

        # Group results by parameter
        param_groups = {}
        for r in self.results:
            for key, value in r['params'].items():
                if key not in param_groups:
                    param_groups[key] = {}
                if value not in param_groups[key]:
                    param_groups[key][value] = []
                if r['final_acc'] is not None:
                    param_groups[key][value].append(r['final_acc'])

        # Calculate range for each parameter
        sensitivities = {}
        for param, values in param_groups.items():
            if len(values) > 1:
                avg_accs = {v: sum(accs)/len(accs) for v, accs in values.items()}
                min_acc = min(avg_accs.values())
                max_acc = max(avg_accs.values())
                range_acc = max_acc - min_acc
                sensitivities[param] = {
                    "range": range_acc,
                    "min": min_acc,
                    "max": max_acc,
                    "best_value": max(avg_accs, key=avg_accs.get)
                }

        # Sort by sensitivity (range)
        sorted_sens = sorted(sensitivities.items(), key=lambda x: x[1]['range'], reverse=True)

        print("\nParameter Sensitivity (higher range = more sensitive):")
        for param, stats in sorted_sens:
            print(f"  {param:25s}: range={stats['range']:5.2f}%, "
                  f"min={stats['min']:.2f}%, max={stats['max']:.2f}%, "
                  f"best={stats['best_value']}")


def run_sensitivity_analysis():
    """Run sensitivity analysis for Amazon target domain."""

    searcher = GridSearch()

    # Define parameter grids for different experiments
    # Each experiment varies one parameter while keeping others fixed

    base_params = {
        "target_epochs": 20,
        "target_batch_size": 64,
        "buffer_capacity": 300,
    }

    experiments = []

    # Exp 1: Target epochs (training intensity)
    for epochs in [10, 20, 30, 40]:
        params = base_params.copy()
        params["target_epochs"] = epochs
        experiments.append(("epochs", params))

    # Exp 2: Batch size
    for bs in [32, 64, 128, 256]:
        params = base_params.copy()
        params["target_batch_size"] = bs
        experiments.append(("batch_size", params))

    # Exp 3: Buffer capacity
    for buf in [150, 300, 500, 800]:
        params = base_params.copy()
        params["buffer_capacity"] = buf
        experiments.append(("buffer", params))

    # Exp 4: Class detection threshold
    for alpha in [1, 3, 5, 10, 15]:
        params = base_params.copy()
        params["class_detect_alpha"] = alpha
        experiments.append(("class_detect", params))

    # Exp 5: Reliability threshold
    for rel_th in [0.1, 0.3, 0.5, 0.7]:
        params = base_params.copy()
        params["reliability_threshold"] = rel_th
        experiments.append(("reliability", params))

    # Exp 6: Loss weights - lambda_pl
    for lp in [0.5, 1.0, 2.0, 3.0]:
        params = base_params.copy()
        params["lambda_pl"] = lp
        experiments.append(("lambda_pl", params))

    # Exp 7: Loss weights - lambda_proto
    for lpr in [0.05, 0.1, 0.2, 0.5]:
        params = base_params.copy()
        params["lambda_proto"] = lpr
        experiments.append(("lambda_proto", params))

    # Exp 8: Prototype temperature
    for pt in [0.05, 0.07, 0.1, 0.15]:
        params = base_params.copy()
        params["proto_temp"] = pt
        experiments.append(("proto_temp", params))

    source = "dslr"  # dslr -> amazon
    target = "amazon"

    for exp_type, params in experiments:
        # Generate run name
        param_str = "_".join(f"{k}={v}" for k, v in params.items())
        run_name = f"{exp_type}_{param_str.replace('.', '_')}"

        # Skip if already run
        existing = [r for r in searcher.results if r['exp_name'] == run_name]
        if existing:
            print(f"Skipping existing: {run_name}")
            continue

        searcher.run_experiment(source, target, params, run_name)

    searcher.print_summary()


if __name__ == "__main__":
    run_sensitivity_analysis()
