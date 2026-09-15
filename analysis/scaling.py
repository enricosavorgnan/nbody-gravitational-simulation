import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def extract_ranks(methods):
    ranks = []
    for m in methods:
        # Match '1 Rank', '2 Ranks', etc. or just try to find the first integer
        match = re.search(r'(\d+)', str(m))
        if match:
            ranks.append(int(match.group(1)))
        else:
            ranks.append(0)
    return ranks

def main():
    parser = argparse.ArgumentParser(description="Merge two scaling CSVs and plot them together")
    parser.add_argument("--source1", required=True, help="Path to first CSV")
    parser.add_argument("--source2", required=True, help="Path to second CSV")
    parser.add_argument("--label1", default="Source 1", help="Label for first source in legend")
    parser.add_argument("--label2", default="Source 2", help="Label for second source in legend")
    parser.add_argument("--type", choices=["strong", "weak"], required=True, help="Type of scaling plot")
    parser.add_argument("--metric", choices=["real_time", "force_time"], default="real_time", 
                        help="Which metric to plot (real_time uses OS wallclock, force_time uses internal compute)")
    parser.add_argument("--xlabel", default="Processing Units (Ranks/Threads)", help="Label for the X-axis")
    parser.add_argument("--output", default="merged_scaling.png", help="Output filename")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.source1):
        print(f"Error: {args.source1} does not exist.")
        return
    if not os.path.exists(args.source2):
        print(f"Error: {args.source2} does not exist.")
        return

    df1 = pd.read_csv(args.source1)
    df2 = pd.read_csv(args.source2)
    
    ranks1 = extract_ranks(df1['method'])
    ranks2 = extract_ranks(df2['method'])
    
    # Fallback to sequential powers of 2 if parsing fails
    if not all(ranks1):
        ranks1 = [2**i for i in range(len(df1))]
    if not all(ranks2):
        ranks2 = [2**i for i in range(len(df2))]
        
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    # Decide which column to use for speedup/efficiency
    col = 'scaling_speedup' if args.metric == 'real_time' else 'speedup_vs_baseline'
    
    if args.type == "strong":
        # Strong scaling speedup
        speedups1 = df1[col]
        speedups2 = df2[col]
        
        ax.plot(ranks1, speedups1, marker='o', color="#2563EB", linewidth=2, label=args.label1)
        ax.plot(ranks2, speedups2, marker='s', color="#DC2626", linewidth=2, label=args.label2)
        
        # Ideal speedup based on first rank of source 1
        ideal_speedup = [r / ranks1[0] for r in ranks1]
        ax.plot(ranks1, ideal_speedup, linestyle='--', color="#4B5563", linewidth=2, label="Ideal Speedup")
        
        ax.set_yscale('log', base=2)
        ax.set_yticks(ranks1)
        ax.set_yticklabels([str(r) for r in ranks1])
        ax.set_ylabel(f"Speedup ({args.metric})", fontsize=11, fontweight="bold")
        ax.set_title("Merged Strong Scaling Speedup", fontsize=12, fontweight="bold", pad=12)
        
    elif args.type == "weak":
        # Weak scaling efficiency
        eff1 = df1[col]
        eff2 = df2[col]
        
        ax.plot(ranks1, eff1, marker='o', color="#059669", linewidth=2, label=args.label1)
        ax.plot(ranks2, eff2, marker='s', color="#D97706", linewidth=2, label=args.label2)
        ax.axhline(1.0, linestyle='--', color="#4B5563", linewidth=2, label="Ideal Efficiency")
        
        ax.set_ylabel(f"Efficiency ({args.metric})", fontsize=11, fontweight="bold")
        ax.set_title("Merged Weak Scaling Efficiency", fontsize=12, fontweight="bold", pad=12)
        
        # Adjust Y limits if needed
        max_eff = max(max(eff1), max(eff2))
        ax.set_ylim(0, max(1.2, max_eff * 1.1))

    ax.set_xscale('log', base=2)
    ax.set_xticks(ranks1)
    ax.set_xticklabels([str(r) for r in ranks1])
    ax.set_xlabel(args.xlabel, fontsize=11, fontweight="bold")
    
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best")
    
    fig.tight_layout()
    plt.savefig(args.output)
    print(f"[SUCCESS] Saved merged plot to {args.output}")

if __name__ == "__main__":
    main()
