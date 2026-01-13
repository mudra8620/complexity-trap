"""
generate_figures.py
====================

Regenerate publication figures from result data.

Usage:
    python figures/generate_figures.py

Author: Mudra Chaudhary
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

def generate_complexity_grid():
    """Generate the complexity grid figure showing all representations."""
    results = pd.read_csv('results/synthetic_results.csv')

    fig, ax = plt.subplots(figsize=(10, 6))

    tiers = ['Linear', 'Conditional', 'Loop', 'Recursion']
    representations = ['Char-Level', 'BPE', 'Positional', 'Pure']
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']

    x = np.arange(len(tiers))
    width = 0.2

    for i, rep in enumerate(representations):
        data = results[results['representation'] == rep]
        drops = [data[data['tier'] == t]['drop'].values[0] for t in tiers]
        ax.bar(x + i*width, drops, width, label=rep, color=colors[i])

    ax.set_xlabel('Complexity Tier', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('The Complexity Trap: Robustness vs Code Complexity', fontsize=14)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(tiers)
    ax.legend(title='Representation')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('figures/complexity_grid_regenerated.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: complexity_grid_regenerated.png")

def generate_humaneval_comparison():
    """Generate HumanEval results comparison figure."""
    results = pd.read_csv('results/humaneval_results.csv')

    fig, ax = plt.subplots(figsize=(8, 5))

    tiers = results['tier'].values
    bpe_drops = results['bpe_drop'].values
    ast_drops = results['ast_drop'].values

    x = np.arange(len(tiers))
    width = 0.35

    ax.bar(x - width/2, bpe_drops, width, label='BPE', color='#f39c12')
    ax.bar(x + width/2, ast_drops, width, label='AST', color='#2ecc71')

    ax.set_xlabel('Problem Type', fontsize=12)
    ax.set_ylabel('Accuracy Drop (%)', fontsize=12)
    ax.set_title('HumanEval Validation: The Complexity Trap in Real Code', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.legend()
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('figures/humaneval_regenerated.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Generated: humaneval_regenerated.png")

def main():
    """Generate all publication figures."""
    print("Generating publication figures...")
    print("-" * 40)

    # Change to repo root directory
    import os
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    generate_complexity_grid()
    generate_humaneval_comparison()

    print("-" * 40)
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
