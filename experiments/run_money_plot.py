#!/usr/bin/env python3
"""
AST-JEPA: THE MONEY PLOT
Full-scale robustness experiment with visualization
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import random
import time
import string
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from mutation_engine_v1 import VariableRenamingEngine
from ast_jepa_phase0 import (
    DataGenerator, to_compressed_ast, CharTokenizer, ASTTokenizer,
    CodeDataset, TinyTransformer, Trainer, SEED
)

def main():
    print("=" * 70)
    print("AST-JEPA ROBUSTNESS TEST: THE MONEY PLOT")
    print("=" * 70)

    device = torch.device('cpu')
    print(f"\nDevice: {device}")

    # Configuration
    N_TRAIN = 2000
    N_TEST = 200
    N_EPOCHS = 25
    BATCH_SIZE = 64

    print(f"Config: {N_TRAIN} train, {N_TEST} test, {N_EPOCHS} epochs")
    print("\nHYPOTHESIS:")
    print("  - Char model will COLLAPSE on mutated code")
    print("  - AST model will be UNFAZED (identical structure)")

    # Generate data
    print("\n[1/5] Generating training data...")
    start = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)

    gen = DataGenerator(seed=SEED)
    train_data = gen.generate_dataset(N_TRAIN, tiers=['linear', 'conditional'], depths=[1, 2])
    print(f"  Generated {len(train_data['code'])} training samples")

    print("\n[2/5] Generating test data...")
    test_gen = DataGenerator(seed=SEED + 9999)
    test_data = test_gen.generate_dataset(N_TEST, tiers=['linear', 'conditional'], depths=[1, 2])
    print(f"  Generated {len(test_data['code'])} test samples")

    # Tokenizers
    print("\n[3/5] Building tokenizers...")
    char_tok = CharTokenizer()
    char_tok.fit(train_data['code'])

    ast_tok = ASTTokenizer()
    ast_tok.fit(train_data['ast'])

    # Extend char vocab for mutations
    for c in string.ascii_lowercase + string.digits + '_':
        if c not in char_tok.char_to_idx:
            idx = len(char_tok.char_to_idx)
            char_tok.char_to_idx[c] = idx
            char_tok.idx_to_char[idx] = c
    char_tok.vocab_size = len(char_tok.char_to_idx)

    print(f"  Char vocab: {char_tok.vocab_size}, AST vocab: {ast_tok.vocab_size}")

    # Encode and create loaders
    char_seqs = [char_tok.encode(c) for c in train_data['code']]
    ast_seqs = [ast_tok.encode(a) for a in train_data['ast']]

    char_ds = CodeDataset(char_seqs, char_tok.pad_idx)
    ast_ds = CodeDataset(ast_seqs, ast_tok.pad_idx)

    char_loader = DataLoader(char_ds, batch_size=BATCH_SIZE, shuffle=True)
    ast_loader = DataLoader(ast_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Create models
    print("\n[4/5] Training models on CLEAN code...")
    char_model = TinyTransformer(
        vocab_size=char_tok.vocab_size,
        d_model=128, nhead=4, num_layers=2, dim_feedforward=256
    )
    ast_model = TinyTransformer(
        vocab_size=ast_tok.vocab_size,
        d_model=128, nhead=4, num_layers=2, dim_feedforward=256
    )

    print(f"  Char params: {char_model.count_parameters():,}")
    print(f"  AST params:  {ast_model.count_parameters():,}")

    char_trainer = Trainer(char_model, char_tok, device, learning_rate=0.001)
    ast_trainer = Trainer(ast_model, ast_tok, device, learning_rate=0.001)

    for epoch in range(N_EPOCHS):
        char_loss, char_acc = char_trainer.train_epoch(char_loader, epoch)
        ast_loss, ast_acc = ast_trainer.train_epoch(ast_loader, epoch)
        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:2d} | Char: {char_acc:.3f} | AST: {ast_acc:.3f} | Time: {elapsed:.0f}s")

    train_time = time.time() - start
    print(f"\n  Training complete in {train_time:.0f}s")
    print(f"  Final - Char: {char_trainer.history['train_acc'][-1]:.1%}, AST: {ast_trainer.history['train_acc'][-1]:.1%}")

    # THE KEY TEST
    print("\n[5/5] ROBUSTNESS TEST on MUTATED code...")
    print("  (This is where the magic happens)")

    char_model.eval()
    ast_model.eval()

    char_orig_accs = []
    char_mut_accs = []
    ast_orig_accs = []
    ast_mut_accs = []

    def eval_single(model, text, tokenizer):
        if not text:
            return 0.0
        tokens = tokenizer.encode(text)
        if len(tokens) < 3:
            return 0.0

        inp = torch.tensor(tokens[:-1]).unsqueeze(0)
        tgt = torch.tensor(tokens[1:]).unsqueeze(0)

        with torch.no_grad():
            logits = model(inp)
            preds = logits.argmax(dim=-1)
            mask = tgt != tokenizer.pad_idx
            correct = ((preds == tgt) & mask).sum().item()
            total = mask.sum().item()
        return correct / total if total > 0 else 0.0

    n_muts = 5  # mutations per sample
    test_samples = test_data['code'][:100]

    for i, code in enumerate(test_samples):
        if (i + 1) % 25 == 0:
            print(f"  Evaluated {i+1}/{len(test_samples)} samples...")

        # Original
        char_orig_accs.append(eval_single(char_model, code, char_tok))
        ast_repr = to_compressed_ast(code)
        ast_orig_accs.append(eval_single(ast_model, ast_repr, ast_tok))

        # Multiple mutations per sample
        char_muts = []
        ast_muts = []
        for m in range(n_muts):
            try:
                engine = VariableRenamingEngine(seed=i * 1000 + m)
                mutated, _ = engine.mutate(code)

                char_muts.append(eval_single(char_model, mutated, char_tok))
                mut_ast = to_compressed_ast(mutated)
                ast_muts.append(eval_single(ast_model, mut_ast, ast_tok))
            except:
                pass

        if char_muts:
            char_mut_accs.append(sum(char_muts) / len(char_muts))
            ast_mut_accs.append(sum(ast_muts) / len(ast_muts))

    # Calculate results
    char_orig = sum(char_orig_accs) / len(char_orig_accs)
    char_mut = sum(char_mut_accs) / len(char_mut_accs)
    ast_orig = sum(ast_orig_accs) / len(ast_orig_accs)
    ast_mut = sum(ast_mut_accs) / len(ast_mut_accs)

    char_drop = char_orig - char_mut
    ast_drop = ast_orig - ast_mut

    print("\n" + "=" * 70)
    print("ROBUSTNESS TEST RESULTS")
    print("=" * 70)

    print(f"\nAccuracy on ORIGINAL test code:")
    print(f"  Char-Level: {char_orig:.1%}")
    print(f"  AST:        {ast_orig:.1%}")

    print(f"\nAccuracy on MUTATED test code (variables renamed):")
    print(f"  Char-Level: {char_mut:.1%}  <-- LOOK AT THIS")
    print(f"  AST:        {ast_mut:.1%}")

    print(f"\nROBUSTNESS (accuracy retention):")
    char_retention = char_mut/char_orig if char_orig > 0 else 0
    ast_retention = ast_mut/ast_orig if ast_orig > 0 else 0
    print(f"  Char-Level: {char_retention:.0%} retained ({-char_drop:.1%} drop)")
    print(f"  AST:        {ast_retention:.0%} retained ({-ast_drop:.1%} drop)")

    # Generate THE MONEY PLOT
    print("\nGenerating THE MONEY PLOT...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: The Money Plot - Bar chart
    ax1 = axes[0]
    x = [0, 1]
    width = 0.35

    char_bars = [char_orig, char_mut]
    ast_bars = [ast_orig, ast_mut]

    bars1 = ax1.bar([i - width/2 for i in x], char_bars, width,
                    label='Char-Level (Token)', color='#FF6B6B', edgecolor='darkred', linewidth=2)
    bars2 = ax1.bar([i + width/2 for i in x], ast_bars, width,
                    label='AST (Structure)', color='#4ECDC4', edgecolor='darkgreen', linewidth=2)

    ax1.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax1.set_title('THE MONEY PLOT\nStructure vs Surface on Variable Renaming', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Original Code', 'Mutated Code\n(Variables Renamed)'], fontsize=12)
    ax1.legend(fontsize=12, loc='upper right')
    ax1.set_ylim(0, 1.15)
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.0%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=13, fontweight='bold')

    # Add collapse arrow
    if char_drop > 0.15:
        ax1.annotate('', xy=(-width/2, char_mut + 0.02),
                    xytext=(-width/2, char_orig - 0.02),
                    arrowprops=dict(arrowstyle='->', color='red', lw=3))
        ax1.text(-0.45, (char_orig + char_mut)/2, f'COLLAPSE\n-{char_drop:.0%}',
                fontsize=11, color='red', fontweight='bold', ha='right', va='center')

    # Plot 2: Learning curves with robustness
    ax2 = axes[1]

    epochs = list(range(1, N_EPOCHS + 1))
    ax2.plot(epochs, char_trainer.history['train_acc'], 'r-', linewidth=2.5,
             label='Char-Level (Train)', marker='o', markersize=4)
    ax2.plot(epochs, ast_trainer.history['train_acc'], 'g-', linewidth=2.5,
             label='AST (Train)', marker='s', markersize=4)

    # Add horizontal lines for test performance
    ax2.axhline(y=char_orig, color='red', linestyle=':', alpha=0.7, linewidth=2)
    ax2.axhline(y=char_mut, color='red', linestyle='--', alpha=0.5, linewidth=2,
                label=f'Char on Mutated: {char_mut:.0%}')
    ax2.axhline(y=ast_mut, color='green', linestyle='--', alpha=0.5, linewidth=2,
                label=f'AST on Mutated: {ast_mut:.0%}')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training + Robustness Test', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3)

    # Summary box
    robustness_ratio = ast_retention / char_retention if char_retention > 0 else float('inf')
    summary = f'ROBUSTNESS RATIO:\nChar: {char_retention:.0%} retained\nAST: {ast_retention:.0%} retained\n\nAST is {robustness_ratio:.1f}x more robust'
    ax2.text(0.02, 0.98, summary, transform=ax2.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.tight_layout()
    plt.savefig('money_plot.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print("  Saved: money_plot.png")

    # Final verdict
    print("\n" + "=" * 70)
    print("THE VERDICT")
    print("=" * 70)

    if char_drop > 0.2 and ast_drop < 0.15:
        print("\n  *** SUCCESS: THE MONEY PLOT IS CONFIRMED! ***")
        print(f"\n  Char model COLLAPSED: {char_orig:.0%} -> {char_mut:.0%} ({char_drop:.0%} drop)")
        print(f"  AST model UNFAZED:    {ast_orig:.0%} -> {ast_mut:.0%} ({ast_drop:.0%} drop)")
        print(f"\n  AST is {robustness_ratio:.1f}x more robust than char-level!")
        print("\n  This proves STRUCTURE > SURFACE.")
        print("  Token-based models memorize. AST models understand.")
    elif char_drop > ast_drop:
        print("\n  ** PARTIAL SUCCESS: AST shows better robustness **")
        print(f"  Char drop: {char_drop:.1%}, AST drop: {ast_drop:.1%}")
    else:
        print("\n  ? Results inconclusive")

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
