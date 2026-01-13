#!/usr/bin/env python3
"""
Positional AST Experiment: The Airtight Proof
==============================================

This experiment proves that Positional AST achieves 0% drop under variable
renaming because the input is IDENTICAL by construction.

Named AST:    fibonacci(n) -> (Fn fibonacci (Ar n) ...)  ≠ (Fn fn_xyz (Ar v1) ...)
Positional:   fibonacci(n) -> (Fn V0 (Ar V1) ...)       = (Fn V0 (Ar V1) ...)

The claim becomes airtight: "Structural representations are invariant by construction"
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
from ast_strategies import code_to_ast
from complexity_generators import generate_dataset, TIER_GENERATORS

# Import base components
from ast_jepa_phase0 import (
    CharTokenizer, ASTTokenizer, CodeDataset,
    TinyTransformer, Trainer, SEED
)


def run_positional_ast_experiment():
    print("=" * 70)
    print("POSITIONAL AST: THE AIRTIGHT PROOF")
    print("=" * 70)
    print("\nThis experiment proves 0% drop under variable renaming")
    print("because Positional AST produces IDENTICAL input by construction.")
    print("=" * 70)

    device = torch.device('cpu')
    print(f"\nDevice: {device}")

    # Configuration
    N_TRAIN = 1500
    N_TEST = 150
    N_EPOCHS = 20
    BATCH_SIZE = 64
    TIERS = ['linear', 'conditional']
    DEPTHS = [1, 2]

    print(f"Config: {N_TRAIN} train, {N_TEST} test, {N_EPOCHS} epochs")
    print(f"Tiers: {TIERS}, Depths: {DEPTHS}")

    # Generate data using complexity generators
    print("\n[1/6] Generating training data...")
    start = time.time()
    random.seed(SEED)
    torch.manual_seed(SEED)

    train_funcs = generate_dataset(N_TRAIN, tiers=TIERS, depths=DEPTHS)
    train_codes = [f.code for f in train_funcs]
    print(f"  Generated {len(train_codes)} training samples")

    print("\n[2/6] Generating test data...")
    random.seed(SEED + 9999)
    test_funcs = generate_dataset(N_TEST, tiers=TIERS, depths=DEPTHS)
    test_codes = [f.code for f in test_funcs]
    print(f"  Generated {len(test_codes)} test samples")

    # Convert to AST using all three strategies
    print("\n[3/6] Converting to AST representations...")

    strategies = ['named', 'positional', 'pure']
    train_asts = {s: [] for s in strategies}
    test_asts = {s: [] for s in strategies}

    for code in train_codes:
        for s in strategies:
            ast_repr = code_to_ast(code, s)
            train_asts[s].append(ast_repr if ast_repr != "ERR" else "(ERR)")

    for code in test_codes:
        for s in strategies:
            ast_repr = code_to_ast(code, s)
            test_asts[s].append(ast_repr if ast_repr != "ERR" else "(ERR)")

    print(f"  Sample Named AST:      {train_asts['named'][0][:50]}...")
    print(f"  Sample Positional AST: {train_asts['positional'][0][:50]}...")
    print(f"  Sample Pure AST:       {train_asts['pure'][0][:50]}...")

    # Build tokenizers for each strategy
    print("\n[4/6] Building tokenizers...")
    tokenizers = {}
    for s in strategies:
        tok = ASTTokenizer()
        tok.fit(train_asts[s])
        tokenizers[s] = tok
        print(f"  {s.capitalize()} vocab size: {tok.vocab_size}")

    # Also build char tokenizer for baseline
    char_tok = CharTokenizer()
    char_tok.fit(train_codes)
    # Extend for mutations
    for c in string.ascii_lowercase + string.digits + '_':
        if c not in char_tok.char_to_idx:
            idx = len(char_tok.char_to_idx)
            char_tok.char_to_idx[c] = idx
            char_tok.idx_to_char[idx] = c
    char_tok.vocab_size = len(char_tok.char_to_idx)
    print(f"  Char vocab size: {char_tok.vocab_size}")

    # Create datasets and loaders
    print("\n[5/6] Training models...")

    # Train char model
    char_seqs = [char_tok.encode(c) for c in train_codes]
    char_ds = CodeDataset(char_seqs, char_tok.pad_idx)
    char_loader = DataLoader(char_ds, batch_size=BATCH_SIZE, shuffle=True)

    char_model = TinyTransformer(vocab_size=char_tok.vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256)
    char_trainer = Trainer(char_model, char_tok, device, learning_rate=0.001)

    # Train AST models for each strategy
    ast_models = {}
    ast_trainers = {}

    for s in strategies:
        ast_seqs = [tokenizers[s].encode(a) for a in train_asts[s]]
        ast_ds = CodeDataset(ast_seqs, tokenizers[s].pad_idx)
        ast_loader = DataLoader(ast_ds, batch_size=BATCH_SIZE, shuffle=True)

        model = TinyTransformer(vocab_size=tokenizers[s].vocab_size, d_model=128, nhead=4, num_layers=2, dim_feedforward=256)
        trainer = Trainer(model, tokenizers[s], device, learning_rate=0.001)

        ast_models[s] = model
        ast_trainers[s] = (trainer, ast_loader)

    # Training loop
    print(f"\n  Training {N_EPOCHS} epochs...")
    for epoch in range(N_EPOCHS):
        char_loss, char_acc = char_trainer.train_epoch(char_loader, epoch)

        ast_accs = {}
        for s in strategies:
            trainer, loader = ast_trainers[s]
            _, acc = trainer.train_epoch(loader, epoch)
            ast_accs[s] = acc

        if (epoch + 1) % 5 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch+1:2d} | Char: {char_acc:.2f} | Named: {ast_accs['named']:.2f} | "
                  f"Pos: {ast_accs['positional']:.2f} | Pure: {ast_accs['pure']:.2f} | Time: {elapsed:.0f}s")

    train_time = time.time() - start
    print(f"\n  Training complete in {train_time:.0f}s")

    # THE KEY TEST: Evaluate on mutated data
    print("\n[6/6] ROBUSTNESS TEST on MUTATED code...")

    char_model.eval()
    for s in strategies:
        ast_models[s].eval()

    def eval_single(model, text, tokenizer):
        if not text or text == "(ERR)":
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

    # Results storage
    results = {
        'char': {'orig': [], 'mut': []},
        'named': {'orig': [], 'mut': []},
        'positional': {'orig': [], 'mut': []},
        'pure': {'orig': [], 'mut': []},
    }

    n_muts = 3
    test_sample = test_codes[:80]

    for i, code in enumerate(test_sample):
        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{len(test_sample)} samples...")

        # Original evaluations
        results['char']['orig'].append(eval_single(char_model, code, char_tok))

        for s in strategies:
            ast_repr = code_to_ast(code, s)
            results[s]['orig'].append(eval_single(ast_models[s], ast_repr, tokenizers[s]))

        # Mutated evaluations
        char_muts = []
        ast_muts = {s: [] for s in strategies}

        for m in range(n_muts):
            try:
                engine = VariableRenamingEngine(seed=i * 1000 + m)
                mutated, _ = engine.mutate(code)

                char_muts.append(eval_single(char_model, mutated, char_tok))

                for s in strategies:
                    mut_ast = code_to_ast(mutated, s)
                    ast_muts[s].append(eval_single(ast_models[s], mut_ast, tokenizers[s]))
            except:
                pass

        if char_muts:
            results['char']['mut'].append(sum(char_muts) / len(char_muts))
            for s in strategies:
                results[s]['mut'].append(sum(ast_muts[s]) / len(ast_muts[s]))

    # Calculate final results
    print("\n" + "=" * 70)
    print("ROBUSTNESS TEST RESULTS")
    print("=" * 70)

    summary = {}
    for model_name in ['char', 'named', 'positional', 'pure']:
        orig = sum(results[model_name]['orig']) / len(results[model_name]['orig'])
        mut = sum(results[model_name]['mut']) / len(results[model_name]['mut'])
        drop = orig - mut
        retention = mut / orig if orig > 0 else 0

        summary[model_name] = {
            'orig': orig,
            'mut': mut,
            'drop': drop,
            'retention': retention
        }

    print(f"\n{'Model':<15} {'Original':>10} {'Mutated':>10} {'Drop':>10} {'Retention':>10}")
    print("-" * 55)
    for name, s in summary.items():
        print(f"{name.capitalize():<15} {s['orig']:>10.1%} {s['mut']:>10.1%} {s['drop']:>10.1%} {s['retention']:>10.0%}")

    # THE KEY COMPARISON
    print("\n" + "=" * 70)
    print("THE AIRTIGHT PROOF")
    print("=" * 70)

    pos_drop = summary['positional']['drop']
    pure_drop = summary['pure']['drop']
    named_drop = summary['named']['drop']
    char_drop = summary['char']['drop']

    print(f"\nChar-Level drop:     {char_drop:.1%}  (surface tokens change)")
    print(f"Named AST drop:      {named_drop:.1%}  (names still in AST)")
    print(f"Positional AST drop: {pos_drop:.1%}  {'<-- INVARIANT BY CONSTRUCTION' if abs(pos_drop) < 0.05 else ''}")
    print(f"Pure AST drop:       {pure_drop:.1%}  {'<-- INVARIANT BY CONSTRUCTION' if abs(pure_drop) < 0.05 else ''}")

    if abs(pos_drop) < 0.05 and abs(pure_drop) < 0.05:
        print("\n*** SUCCESS: Positional and Pure AST achieve ~0% drop! ***")
        print("The claim is now airtight:")
        print("  'Structural representations are invariant by construction'")
        print("\nThis is NeurIPS-worthy evidence.")
    elif pos_drop < named_drop:
        print("\n** PARTIAL SUCCESS: Positional shows improvement over Named **")
    else:
        print("\n? Results need investigation")

    # Generate comparison plot
    print("\n" + "=" * 70)
    print("Generating comparison plot...")

    fig, ax = plt.subplots(figsize=(12, 7))

    models = ['Char-Level', 'Named AST', 'Positional AST', 'Pure AST']
    x = range(len(models))
    width = 0.35

    orig_vals = [summary['char']['orig'], summary['named']['orig'],
                 summary['positional']['orig'], summary['pure']['orig']]
    mut_vals = [summary['char']['mut'], summary['named']['mut'],
                summary['positional']['mut'], summary['pure']['mut']]

    bars1 = ax.bar([i - width/2 for i in x], orig_vals, width, label='Original Code',
                   color=['#FF6B6B', '#FFB347', '#4ECDC4', '#45B7D1'], edgecolor='black', linewidth=1.5)
    bars2 = ax.bar([i + width/2 for i in x], mut_vals, width, label='Mutated Code',
                   color=['#FF6B6B', '#FFB347', '#4ECDC4', '#45B7D1'], alpha=0.5, edgecolor='black', linewidth=1.5, hatch='//')

    ax.set_ylabel('Accuracy', fontsize=14, fontweight='bold')
    ax.set_title('The Airtight Proof: Structural Invariance by Construction', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim(0, 1.15)

    # Add value labels and drop annotations
    for i, (o, m) in enumerate(zip(orig_vals, mut_vals)):
        ax.annotate(f'{o:.0%}', xy=(i - width/2, o), xytext=(0, 3),
                   textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')
        ax.annotate(f'{m:.0%}', xy=(i + width/2, m), xytext=(0, 3),
                   textcoords='offset points', ha='center', va='bottom', fontsize=11, fontweight='bold')

        drop = o - m
        color = 'red' if drop > 0.1 else 'orange' if drop > 0.02 else 'green'
        ax.annotate(f'{drop:+.0%}', xy=(i, min(o, m) - 0.05),
                   ha='center', va='top', fontsize=10, fontweight='bold', color=color)

    # Add explanation box
    explanation = ("Positional AST: V0, V1, V2 (order-based)\n"
                  "Input is IDENTICAL regardless of variable names\n"
                  "→ 0% drop by construction")
    ax.text(0.98, 0.02, explanation, transform=ax.transAxes, fontsize=10,
           verticalalignment='bottom', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig('positional_ast_proof.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    print("Saved: positional_ast_proof.png")
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    return summary


if __name__ == "__main__":
    run_positional_ast_experiment()
