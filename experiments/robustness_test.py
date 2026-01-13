#!/usr/bin/env python3
"""
AST-JEPA Robustness Test: The Money Plot
=========================================

THE SMOKING GUN EXPERIMENT

Train both models on clean code, then test on MUTATED code.
The AST model sees identical structure. The char model sees alien tokens.

Expected Result:
- Char-Level: Accuracy COLLAPSES (90% -> 15%)
- AST:        Accuracy UNCHANGED (90% -> 90%)

This is the billion-dollar proof that structure > surface.
"""

import ast
import random
import math
import time
import string
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Import from our modules
from mutation_engine_v1 import VariableRenamingEngine, generate_n_mutations
from ast_jepa_phase0 import (
    CompressedASTVisitor, to_compressed_ast, verify_round_trip,
    DataGenerator, CharTokenizer, ASTTokenizer, CodeDataset,
    TinyTransformer, Trainer, SEED
)

# ============================================================================
# ROBUSTNESS TEST FRAMEWORK
# ============================================================================

class RobustnessEvaluator:
    """
    Evaluates model robustness to variable renaming mutations.

    The key insight: If a model truly understands STRUCTURE,
    renaming variables should have ZERO impact on predictions.
    """

    def __init__(self, char_tokenizer: CharTokenizer, ast_tokenizer: ASTTokenizer):
        self.char_tokenizer = char_tokenizer
        self.ast_tokenizer = ast_tokenizer
        self.mutation_engine = VariableRenamingEngine()

    def evaluate_on_mutated(self,
                            char_model: nn.Module,
                            ast_model: nn.Module,
                            test_codes: List[str],
                            device: torch.device,
                            n_mutations: int = 5) -> Dict:
        """
        Evaluate both models on original and mutated test data.

        Args:
            char_model: Trained character-level model
            ast_model: Trained AST model
            test_codes: List of original test code samples
            device: torch device
            n_mutations: Number of mutations per test sample

        Returns:
            Dictionary with accuracy metrics
        """
        char_model.eval()
        ast_model.eval()

        results = {
            'char_original_acc': [],
            'char_mutated_acc': [],
            'ast_original_acc': [],
            'ast_mutated_acc': [],
            'mutation_details': []
        }

        with torch.no_grad():
            for code in test_codes:
                # Evaluate on original
                char_orig_acc = self._evaluate_single(char_model, code,
                                                       self.char_tokenizer, device, is_ast=False)
                ast_repr = to_compressed_ast(code)
                ast_orig_acc = self._evaluate_single(ast_model, ast_repr,
                                                      self.ast_tokenizer, device, is_ast=True)

                results['char_original_acc'].append(char_orig_acc)
                results['ast_original_acc'].append(ast_orig_acc)

                # Generate mutations and evaluate
                char_mut_accs = []
                ast_mut_accs = []

                for seed in range(n_mutations):
                    try:
                        engine = VariableRenamingEngine(seed=seed * 1000)
                        mutated_code, rename_map = engine.mutate(code)

                        # Char model on mutated code
                        char_mut_acc = self._evaluate_single(char_model, mutated_code,
                                                              self.char_tokenizer, device, is_ast=False)
                        char_mut_accs.append(char_mut_acc)

                        # AST model on mutated code (should be identical structure!)
                        mutated_ast = to_compressed_ast(mutated_code)
                        ast_mut_acc = self._evaluate_single(ast_model, mutated_ast,
                                                             self.ast_tokenizer, device, is_ast=True)
                        ast_mut_accs.append(ast_mut_acc)

                    except Exception as e:
                        # Skip samples that can't be mutated
                        continue

                if char_mut_accs:
                    results['char_mutated_acc'].append(sum(char_mut_accs) / len(char_mut_accs))
                    results['ast_mutated_acc'].append(sum(ast_mut_accs) / len(ast_mut_accs))

        # Compute summary statistics
        summary = {
            'char_original_mean': sum(results['char_original_acc']) / len(results['char_original_acc']),
            'char_mutated_mean': sum(results['char_mutated_acc']) / len(results['char_mutated_acc']),
            'ast_original_mean': sum(results['ast_original_acc']) / len(results['ast_original_acc']),
            'ast_mutated_mean': sum(results['ast_mutated_acc']) / len(results['ast_mutated_acc']),
        }

        # The key metric: robustness ratio
        summary['char_robustness'] = summary['char_mutated_mean'] / max(summary['char_original_mean'], 0.01)
        summary['ast_robustness'] = summary['ast_mutated_mean'] / max(summary['ast_original_mean'], 0.01)

        results['summary'] = summary
        return results

    def _evaluate_single(self, model: nn.Module, text: str,
                         tokenizer, device: torch.device, is_ast: bool) -> float:
        """Evaluate model on a single sample, return token-level accuracy."""
        if text is None:
            return 0.0

        tokens = tokenizer.encode(text)

        if len(tokens) < 3:  # Need at least SOS + 1 token + EOS
            return 0.0

        # Create input/target
        input_seq = torch.tensor(tokens[:-1], dtype=torch.long).unsqueeze(0).to(device)
        target_seq = torch.tensor(tokens[1:], dtype=torch.long).unsqueeze(0).to(device)

        # Forward pass
        padding_mask = (input_seq == tokenizer.pad_idx)
        logits = model(input_seq, padding_mask)

        # Compute accuracy
        preds = logits.argmax(dim=-1)
        mask = target_seq != tokenizer.pad_idx
        correct = ((preds == target_seq) & mask).sum().item()
        total = mask.sum().item()

        return correct / total if total > 0 else 0.0


def run_robustness_experiment(n_train: int = 3000,
                               n_test: int = 500,
                               n_epochs: int = 30,
                               batch_size: int = 64,
                               n_mutations: int = 5,
                               device_str: str = 'auto',
                               verbose: bool = True) -> Dict:
    """
    Run the full robustness experiment: train on clean, test on mutated.

    This produces THE MONEY PLOT.
    """

    # Setup device
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    if verbose:
        print("=" * 70)
        print("AST-JEPA ROBUSTNESS TEST: THE MONEY PLOT")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Train samples: {n_train}, Test samples: {n_test}")
        print(f"Mutations per test: {n_mutations}")
        print()
        print("HYPOTHESIS:")
        print("  - Char model will COLLAPSE on mutated code")
        print("  - AST model will be UNFAZED (identical structure)")
        print("=" * 70)

    # Generate training data
    if verbose:
        print("\n[1/5] Generating training data...")

    random.seed(SEED)
    torch.manual_seed(SEED)

    generator = DataGenerator(seed=SEED)
    train_data = generator.generate_dataset(n_train, tiers=['linear', 'conditional'], depths=[1, 2])

    if verbose:
        print(f"  Generated {len(train_data['code'])} training samples")

    # Generate SEPARATE test data (to avoid data leakage)
    if verbose:
        print("\n[2/5] Generating test data...")

    test_generator = DataGenerator(seed=SEED + 9999)  # Different seed
    test_data = test_generator.generate_dataset(n_test, tiers=['linear', 'conditional'], depths=[1, 2])

    if verbose:
        print(f"  Generated {len(test_data['code'])} test samples")

    # Build tokenizers on training data only
    if verbose:
        print("\n[3/5] Building tokenizers...")

    char_tokenizer = CharTokenizer()
    char_tokenizer.fit(train_data['code'])

    ast_tokenizer = ASTTokenizer()
    ast_tokenizer.fit(train_data['ast'])

    # CRITICAL: Extend char vocab to handle mutated variable names
    # This simulates a more realistic scenario where the tokenizer has seen
    # more diverse tokens (but not the specific mutations we'll test)
    extended_chars = set()
    for c in string.ascii_lowercase + string.digits + '_':
        extended_chars.add(c)
    for c in extended_chars:
        if c not in char_tokenizer.char_to_idx:
            idx = len(char_tokenizer.char_to_idx)
            char_tokenizer.char_to_idx[c] = idx
            char_tokenizer.idx_to_char[idx] = c
    char_tokenizer.vocab_size = len(char_tokenizer.char_to_idx)

    if verbose:
        print(f"  Char vocab size: {char_tokenizer.vocab_size}")
        print(f"  AST vocab size:  {ast_tokenizer.vocab_size}")

    # Encode training data
    char_sequences = [char_tokenizer.encode(c) for c in train_data['code']]
    ast_sequences = [ast_tokenizer.encode(a) for a in train_data['ast']]

    # Create datasets
    char_dataset = CodeDataset(char_sequences, char_tokenizer.pad_idx)
    ast_dataset = CodeDataset(ast_sequences, ast_tokenizer.pad_idx)

    char_loader = DataLoader(char_dataset, batch_size=batch_size, shuffle=True)
    ast_loader = DataLoader(ast_dataset, batch_size=batch_size, shuffle=True)

    # Create models
    if verbose:
        print("\n[4/5] Training models on CLEAN data...")

    char_model = TinyTransformer(
        vocab_size=char_tokenizer.vocab_size,
        d_model=128, nhead=4, num_layers=2, dim_feedforward=256
    )

    ast_model = TinyTransformer(
        vocab_size=ast_tokenizer.vocab_size,
        d_model=128, nhead=4, num_layers=2, dim_feedforward=256
    )

    char_trainer = Trainer(char_model, char_tokenizer, device)
    ast_trainer = Trainer(ast_model, ast_tokenizer, device)

    # Training loop
    start_time = time.time()

    for epoch in range(n_epochs):
        char_loss, char_acc = char_trainer.train_epoch(char_loader, epoch)
        ast_loss, ast_acc = ast_trainer.train_epoch(ast_loader, epoch)

        if verbose and (epoch + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Epoch {epoch+1:3d} | "
                  f"Char: acc={char_acc:.3f} | "
                  f"AST: acc={ast_acc:.3f} | "
                  f"Time: {elapsed:.1f}s")

    train_time = time.time() - start_time

    if verbose:
        print(f"\n  Training complete in {train_time:.1f}s")
        print(f"  Final Char accuracy: {char_trainer.history['train_acc'][-1]:.3f}")
        print(f"  Final AST accuracy:  {ast_trainer.history['train_acc'][-1]:.3f}")

    # THE KEY TEST: Evaluate on mutated data
    if verbose:
        print("\n[5/5] Evaluating on MUTATED test data...")
        print("  (This is where the magic happens)")

    evaluator = RobustnessEvaluator(char_tokenizer, ast_tokenizer)
    robustness_results = evaluator.evaluate_on_mutated(
        char_model, ast_model,
        test_data['code'][:min(100, len(test_data['code']))],  # Limit for speed
        device,
        n_mutations=n_mutations
    )

    # Print results
    if verbose:
        print("\n" + "=" * 70)
        print("ROBUSTNESS TEST RESULTS")
        print("=" * 70)

        summary = robustness_results['summary']

        print("\nAccuracy on ORIGINAL test data:")
        print(f"  Char-Level: {summary['char_original_mean']:.1%}")
        print(f"  AST:        {summary['ast_original_mean']:.1%}")

        print("\nAccuracy on MUTATED test data (variables renamed):")
        print(f"  Char-Level: {summary['char_mutated_mean']:.1%}")
        print(f"  AST:        {summary['ast_mutated_mean']:.1%}")

        print("\nROBUSTNESS RATIO (mutated/original):")
        print(f"  Char-Level: {summary['char_robustness']:.1%}")
        print(f"  AST:        {summary['ast_robustness']:.1%}")

        print("\n" + "=" * 70)
        print("THE VERDICT")
        print("=" * 70)

        char_collapse = summary['char_original_mean'] - summary['char_mutated_mean']
        ast_collapse = summary['ast_original_mean'] - summary['ast_mutated_mean']

        if char_collapse > 0.2 and ast_collapse < 0.1:
            print("\n  *** SUCCESS: THE MONEY PLOT IS CONFIRMED! ***")
            print(f"\n  Char model COLLAPSED: {summary['char_original_mean']:.1%} -> {summary['char_mutated_mean']:.1%} ({char_collapse:.1%} drop)")
            print(f"  AST model UNFAZED:    {summary['ast_original_mean']:.1%} -> {summary['ast_mutated_mean']:.1%} ({ast_collapse:.1%} drop)")
            print("\n  This proves structure > surface.")
            print("  Token-based models memorize. AST models understand.")
        elif char_collapse > ast_collapse:
            print("\n  ** PARTIAL SUCCESS: AST shows better robustness **")
            print(f"  Char drop: {char_collapse:.1%}, AST drop: {ast_collapse:.1%}")
            print("  The effect exists but may need larger scale to be definitive.")
        else:
            print("\n  ? UNEXPECTED: Both models show similar robustness")
            print("  This may indicate the synthetic data is too simple.")
            print("  Try with more complex code patterns.")

    # Compile all results
    results = {
        'train_time': train_time,
        'char_train_final_acc': char_trainer.history['train_acc'][-1],
        'ast_train_final_acc': ast_trainer.history['train_acc'][-1],
        'robustness': robustness_results['summary'],
        'char_history': char_trainer.history,
        'ast_history': ast_trainer.history,
        'config': {
            'n_train': n_train,
            'n_test': n_test,
            'n_epochs': n_epochs,
            'n_mutations': n_mutations
        }
    }

    return results


def plot_robustness_results(results: Dict, save_path: str = 'robustness_money_plot.png'):
    """Generate THE MONEY PLOT."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    summary = results['robustness']

    # Plot 1: The Money Plot - Bar chart of original vs mutated
    ax1 = axes[0]

    x = [0, 1]
    width = 0.35

    char_bars = [summary['char_original_mean'], summary['char_mutated_mean']]
    ast_bars = [summary['ast_original_mean'], summary['ast_mutated_mean']]

    bars1 = ax1.bar([i - width/2 for i in x], char_bars, width, label='Char-Level', color='#FF6B6B', edgecolor='darkred')
    bars2 = ax1.bar([i + width/2 for i in x], ast_bars, width, label='AST', color='#4ECDC4', edgecolor='darkgreen')

    ax1.set_ylabel('Accuracy', fontsize=14)
    ax1.set_title('THE MONEY PLOT: Structure vs Surface', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Original Code', 'Mutated Code\n(Variables Renamed)'], fontsize=12)
    ax1.legend(fontsize=12)
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    # Add value labels on bars
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1%}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add arrow showing collapse
    char_drop = summary['char_original_mean'] - summary['char_mutated_mean']
    if char_drop > 0.1:
        ax1.annotate('', xy=(0 - width/2, summary['char_mutated_mean']),
                    xytext=(0 - width/2, summary['char_original_mean']),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax1.text(-0.5, (summary['char_original_mean'] + summary['char_mutated_mean'])/2,
                f'COLLAPSE\n-{char_drop:.0%}', fontsize=10, color='red', fontweight='bold',
                ha='right', va='center')

    # Plot 2: Learning curves with robustness annotation
    ax2 = axes[1]

    ax2.plot(results['char_history']['epoch'], results['char_history']['train_acc'],
             'r-', linewidth=2, label='Char-Level (Train)')
    ax2.plot(results['ast_history']['epoch'], results['ast_history']['train_acc'],
             'g-', linewidth=2, label='AST (Train)')

    # Add horizontal lines for mutated performance
    ax2.axhline(y=summary['char_mutated_mean'], color='red', linestyle='--',
                alpha=0.7, label=f'Char on Mutated: {summary["char_mutated_mean"]:.1%}')
    ax2.axhline(y=summary['ast_mutated_mean'], color='green', linestyle='--',
                alpha=0.7, label=f'AST on Mutated: {summary["ast_mutated_mean"]:.1%}')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training Progress + Robustness Test', fontsize=14)
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # Add summary box
    robustness_text = (
        f"ROBUSTNESS RATIO:\n"
        f"Char: {summary['char_robustness']:.0%}\n"
        f"AST:  {summary['ast_robustness']:.0%}"
    )
    ax2.text(0.02, 0.98, robustness_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("AST-JEPA ROBUSTNESS TEST")
    print("The Smoking Gun Experiment")
    print("=" * 70)
    print("\nThis experiment trains on CLEAN code and tests on MUTATED code.")
    print("Variable names are obfuscated. The AST structure remains identical.")
    print("\nIf our hypothesis is correct:")
    print("  - Char-Level will COLLAPSE (memorized surface patterns)")
    print("  - AST will be UNFAZED (learned deep structure)")
    print("=" * 70 + "\n")

    # Run the experiment
    results = run_robustness_experiment(
        n_train=3000,
        n_test=300,
        n_epochs=30,
        batch_size=64,
        n_mutations=5,
        verbose=True
    )

    # Generate the money plot
    plot_path = plot_robustness_results(results, 'robustness_money_plot.png')
    print(f"\nMoney plot saved to: {plot_path}")

    # Save results
    results_json = {
        'train_time': results['train_time'],
        'char_train_final_acc': results['char_train_final_acc'],
        'ast_train_final_acc': results['ast_train_final_acc'],
        'robustness': results['robustness'],
        'config': results['config']
    }

    with open('robustness_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to: robustness_results.json")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    if results['robustness']['char_robustness'] < 0.7 and results['robustness']['ast_robustness'] > 0.9:
        print("\nTHE MONEY PLOT CONFIRMED!")
        print("Structure-based representations are fundamentally more robust")
        print("than surface-based representations.")
        print("\nNext steps:")
        print("  1. Scale up to more complex code")
        print("  2. Test on real-world code (not just synthetic)")
        print("  3. Write the paper!")
