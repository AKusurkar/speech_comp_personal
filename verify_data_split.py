import pandas as pd
import json

def verify_data_split(train_file, val_file):
    print("Loading datasets...")
    # Load the saved jsonl files
    with open(train_file, 'r') as f:
        train_df = pd.DataFrame([json.loads(line) for line in f])
    with open(val_file, 'r') as f:
        val_df = pd.DataFrame([json.loads(line) for line in f])

    # 1. Check for Child ID Leakage
    print("\n--- Child Overlap Check ---")
    train_children = set(train_df['child_id'].unique())
    val_children = set(val_df['child_id'].unique())
    overlap = train_children.intersection(val_children)
    
    if len(overlap) == 0:
        print("✅ SUCCESS: 0 overlapping children between train and validation sets.")
    else:
        print(f"❌ WARNING: Found {len(overlap)} overlapping children!")
        print(f"Overlapping IDs: {overlap}")

    # 2. Check the Percentiles in the Validation Set
    print("\n--- Validation Duration Percentiles ---")
    
    # Your required targets
    targets = {
        'p5': 0.61,
        'p25': 1.05,
        'p50': 1.89,
        'p75': 3.29,
        'p95': 6.52,
        'p99': 12.51
    }
 
    # Calculate the actual quantiles on the saved validation data
    quantiles = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    actual_vals = val_df['audio_duration_sec'].quantile(quantiles).values
    
    # Print a formatted table comparing the results
    print(f"{'Percentile':<12} | {'Target (sec)':<14} | {'Actual (sec)':<14} | {'Difference'}")
    print("-" * 60)
    
    for (p_label, target), actual in zip(targets.items(), actual_vals):
        diff = actual - target
        print(f"{p_label:<12} | {target:<14.2f} | {actual:<14.2f} | {diff:<+12.2f}")

if __name__ == "__main__":
    
    # Point these to the location where your SplitData class saved the files
    train_file = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/train_data_comb.jsonl"
    val_file = "/home/epochvipc1/Documents/Speech_comp_temp/dicts_combined/val_data_comb.jsonl"
    
    verify_data_split(train_file, val_file)