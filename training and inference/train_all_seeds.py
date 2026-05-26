"""
Run all models with multiple seeds - WITH VISIBLE OUTPUT
"""

import subprocess
import sys
from datetime import datetime

SEEDS = [42, 123, 456]
MODELS = ["mlp_baseline", "gru", "lstm", "transformer", "mamba", "mamba2"]

def run_training(model, seed):
    """Run a single training job with output visible"""
    print(f"\n{'='*80}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Starting: {model} with seed {seed}")
    print(f"{'='*80}\n")
    
    cmd = [
        sys.executable,  # Use same python as current environment
        "train.py",
        "--model", model,
        "--seed", str(seed),
        "--output_dir", f"outputs/{model}/seed_{seed}",
    ]
    
    try:
        # Run with output streaming to console
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            stdout=sys.stdout,  # Show output in real-time
            stderr=sys.stderr,
        )
        
        print(f"\n✓ Completed: {model} (seed {seed})")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed: {model} (seed {seed})")
        print(f"Error: {e}")
        return False
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)

def main():
    print("="*80)
    print("TRAINING ALL MODELS")
    print("="*80)
    print(f"Models: {', '.join(MODELS)}")
    print(f"Seeds: {SEEDS}")
    print(f"Total runs: {len(MODELS) * len(SEEDS)}")
    print("="*80)
    
    input("\nPress ENTER to start training...")
    
    results = {}
    start_time = datetime.now()
    
    for model in MODELS:
        results[model] = {}
        for seed in SEEDS:
            success = run_training(model, seed)
            results[model][seed] = "✓" if success else "✗"
    
    # Summary
    end_time = datetime.now()
    duration = end_time - start_time
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(f"Total time: {duration}")
    print()
    
    for model in MODELS:
        status = " ".join(results[model].values())
        print(f"{model:15s}: {status}")
    
    # Count successes
    total = len(MODELS) * len(SEEDS)
    success = sum(1 for m in results.values() for s in m.values() if s == "✓")
    
    print(f"\nCompleted: {success}/{total} runs")
    print("="*80)

if __name__ == "__main__":
    main()
