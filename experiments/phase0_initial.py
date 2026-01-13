#!/usr/bin/env python3
"""
AST-JEPA Phase 0: The Race
==========================
Testing whether compressed AST representations learn faster than character-level
representations on tiny transformer models.

Hypothesis: AST model reaches 90% accuracy in ≤50% of the samples required by char-level.

Based on insights from:
- GrammarCoder (2024): Structure helps small models
- SIP (Lindemann et al., 2023): Simulation pre-training creates beneficial inductive biases
- Your thesis: Pure compressed ASTs on tiny models is unexplored territory

Run time: ~30-60 minutes on CPU, ~10-15 minutes on GPU
"""

import ast
import random
import math
import time
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)

# ============================================================================
# PART 1: THE COMPRESSED AST LINEARIZER
# ============================================================================

class CompressedASTVisitor(ast.NodeVisitor):
    """
    Converts Python AST to compressed S-expression format.
    Uses short readable tags for balance between compression and debuggability.

    Example:
        Input:  "def f(): return a + b"
        Output: "(Fn f (Ar) ((Rt (Ad (Vr) (Vr)))))"
    """

    # Compressed tag mapping - readable but short
    OP_MAP = {
        ast.Add: 'Ad', ast.Sub: 'Sb', ast.Mult: 'Ml', ast.Div: 'Dv',
        ast.Mod: 'Md', ast.Pow: 'Pw', ast.FloorDiv: 'Fd',
    }

    CMP_MAP = {
        ast.Gt: 'Gt', ast.Lt: 'Lt', ast.GtE: 'Ge', ast.LtE: 'Le',
        ast.Eq: 'Eq', ast.NotEq: 'Ne',
    }

    def visit_Module(self, node):
        parts = [self.visit(n) for n in node.body]
        return " ".join(p for p in parts if p)

    def visit_FunctionDef(self, node):
        body_parts = [self.visit(n) for n in node.body]
        body = " ".join(p for p in body_parts if p)
        return f"(Fn {node.name} (Ar) ({body}))"

    def visit_Return(self, node):
        if node.value is None:
            return "(Rt)"
        value = self.visit(node.value)
        return f"(Rt {value})"

    def visit_BinOp(self, node):
        op = self.OP_MAP.get(type(node.op), 'Op')
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"({op} {left} {right})"

    def visit_Compare(self, node):
        # Handle single comparison (a > b)
        if len(node.ops) == 1:
            op = self.CMP_MAP.get(type(node.ops[0]), 'Cm')
            left = self.visit(node.left)
            right = self.visit(node.comparators[0])
            return f"({op} {left} {right})"
        # Handle chained comparison (a < b < c) - rare in our synthetic data
        result = self.visit(node.left)
        for op, comp in zip(node.ops, node.comparators):
            op_str = self.CMP_MAP.get(type(op), 'Cm')
            result = f"({op_str} {result} {self.visit(comp)})"
        return result

    def visit_If(self, node):
        test = self.visit(node.test)
        body_parts = [self.visit(n) for n in node.body]
        body = " ".join(p for p in body_parts if p)

        if node.orelse:
            orelse_parts = [self.visit(n) for n in node.orelse]
            orelse = " ".join(p for p in orelse_parts if p)
            return f"(If {test} ({body}) ({orelse}))"
        return f"(If {test} ({body}))"

    def visit_While(self, node):
        test = self.visit(node.test)
        body_parts = [self.visit(n) for n in node.body]
        body = " ".join(p for p in body_parts if p)

        if node.orelse:
            orelse_parts = [self.visit(n) for n in node.orelse]
            orelse = " ".join(p for p in orelse_parts if p)
            return f"(Wh {test} ({body}) ({orelse}))"
        return f"(Wh {test} ({body}))"

    def visit_Assign(self, node):
        targets = " ".join(self.visit(t) for t in node.targets)
        value = self.visit(node.value)
        return f"(As {targets} {value})"

    def visit_AugAssign(self, node):
        target = self.visit(node.target)
        op = self.OP_MAP.get(type(node.op), 'Op')
        value = self.visit(node.value)
        return f"(Au {op} {target} {value})"

    def visit_Name(self, node):
        # Abstract variable names to focus on structure
        return "(Vr)"

    def visit_Constant(self, node):
        # Abstract constants to focus on structure
        if isinstance(node.value, (int, float)):
            return "(Nm)"
        return "(Cn)"

    def visit_Expr(self, node):
        return self.visit(node.value)

    def visit_Pass(self, node):
        return "(Ps)"

    def generic_visit(self, node):
        # Fallback for unhandled nodes
        return ""


def to_compressed_ast(code: str) -> Optional[str]:
    """Convert Python code to compressed AST S-expression."""
    try:
        tree = ast.parse(code)
        visitor = CompressedASTVisitor()
        result = visitor.visit(tree)
        # Validate: check balanced parentheses
        if result.count('(') != result.count(')'):
            return None
        return result if result.strip() else None
    except SyntaxError:
        return None


def verify_round_trip(code: str) -> bool:
    """Verify that code can be parsed and linearized without errors."""
    ast_repr = to_compressed_ast(code)
    if ast_repr is None:
        return False
    # Check it's non-empty and balanced
    return len(ast_repr) > 0 and ast_repr.count('(') == ast_repr.count(')')


# ============================================================================
# PART 2: THE ORTHOGONAL COMPLEXITY DATA GENERATOR
# ============================================================================

class DataGenerator:
    """
    Generates synthetic Python code with controlled complexity.

    Complexity Grid:
    - Expression Depth: 1 (flat), 2 (nested), 3 (deep)
    - Control Flow: Linear, Conditional (if), Loop (while)
    """

    VARS = ['a', 'b', 'c', 'x', 'y', 'z']
    OPS = ['+', '-', '*']  # Avoid division to prevent div-by-zero complexity

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def _gen_expr(self, depth: int) -> str:
        """Generate arithmetic expression of given depth."""
        if depth == 1:
            # Flat: a + b
            v1, v2 = random.sample(self.VARS, 2)
            op = random.choice(self.OPS)
            return f"{v1} {op} {v2}"
        else:
            # Nested: (expr) op (expr)
            left = self._gen_expr(depth - 1)
            right = self._gen_expr(depth - 1)
            op = random.choice(self.OPS)
            return f"({left}) {op} ({right})"

    def _gen_comparison(self) -> str:
        """Generate a simple comparison."""
        v1, v2 = random.sample(self.VARS, 2)
        cmp_op = random.choice(['>', '<', '>=', '<='])
        return f"{v1} {cmp_op} {v2}"

    def generate_linear(self, depth: int) -> str:
        """Generate: def f(): return <expr>"""
        expr = self._gen_expr(depth)
        return f"def f():\n    return {expr}"

    def generate_conditional(self, depth: int) -> str:
        """Generate: def f(): if <cmp>: return <expr> else: return <expr>"""
        cmp = self._gen_comparison()
        expr1 = self._gen_expr(depth)
        expr2 = self._gen_expr(depth)
        return f"def f():\n    if {cmp}:\n        return {expr1}\n    else:\n        return {expr2}"

    def generate_loop(self, depth: int) -> str:
        """Generate: def f(): while <cmp>: a = <expr>; return a"""
        cmp = self._gen_comparison()
        expr = self._gen_expr(depth)
        return f"def f():\n    while {cmp}:\n        a = {expr}\n    return a"

    def generate_dataset(self, n_samples: int,
                         tiers: List[str] = ['linear', 'conditional', 'loop'],
                         depths: List[int] = [1, 2, 3]) -> Dict[str, List]:
        """
        Generate balanced dataset across complexity grid.

        Returns dict with 'code', 'ast', 'tier', 'depth' lists.
        """
        data = {'code': [], 'ast': [], 'tier': [], 'depth': []}

        generators = {
            'linear': self.generate_linear,
            'conditional': self.generate_conditional,
            'loop': self.generate_loop,
        }

        samples_per_cell = n_samples // (len(tiers) * len(depths))

        for tier in tiers:
            gen_func = generators[tier]
            for depth in depths:
                generated = 0
                attempts = 0
                max_attempts = samples_per_cell * 10

                while generated < samples_per_cell and attempts < max_attempts:
                    attempts += 1
                    code = gen_func(depth)
                    ast_repr = to_compressed_ast(code)

                    if ast_repr is not None and verify_round_trip(code):
                        data['code'].append(code)
                        data['ast'].append(ast_repr)
                        data['tier'].append(tier)
                        data['depth'].append(depth)
                        generated += 1

        return data


# ============================================================================
# PART 3: TOKENIZERS
# ============================================================================

class CharTokenizer:
    """Character-level tokenizer for raw code."""

    def __init__(self):
        self.char_to_idx = {}
        self.idx_to_char = {}
        self.vocab_size = 0

        # Reserve special tokens
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

    def fit(self, texts: List[str]):
        """Build vocabulary from texts."""
        chars = set()
        for text in texts:
            chars.update(text)

        # Special tokens first
        special = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        all_tokens = special + sorted(chars)

        self.char_to_idx = {c: i for i, c in enumerate(all_tokens)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.vocab_size = len(all_tokens)

    def encode(self, text: str) -> List[int]:
        """Convert text to token indices."""
        sos = self.char_to_idx[self.sos_token]
        eos = self.char_to_idx[self.eos_token]
        unk = self.char_to_idx[self.unk_token]

        tokens = [sos]
        tokens.extend(self.char_to_idx.get(c, unk) for c in text)
        tokens.append(eos)
        return tokens

    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        chars = []
        for idx in indices:
            token = self.idx_to_char.get(idx, self.unk_token)
            if token in [self.pad_token, self.sos_token, self.eos_token]:
                continue
            chars.append(token)
        return ''.join(chars)

    @property
    def pad_idx(self) -> int:
        return self.char_to_idx[self.pad_token]


class ASTTokenizer:
    """Token-level tokenizer for compressed AST S-expressions."""

    def __init__(self):
        self.token_to_idx = {}
        self.idx_to_token = {}
        self.vocab_size = 0

        # Reserve special tokens
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

    def _tokenize(self, text: str) -> List[str]:
        """Split S-expression into tokens."""
        # Tokens are: (, ), and tag names like Fn, Rt, Ad, etc.
        tokens = []
        i = 0
        while i < len(text):
            c = text[i]
            if c in '()':
                tokens.append(c)
                i += 1
            elif c.isspace():
                i += 1
            else:
                # Read until space or paren
                j = i
                while j < len(text) and text[j] not in '() \t\n':
                    j += 1
                tokens.append(text[i:j])
                i = j
        return tokens

    def fit(self, texts: List[str]):
        """Build vocabulary from S-expressions."""
        all_tokens = set()
        for text in texts:
            all_tokens.update(self._tokenize(text))

        # Special tokens first
        special = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        all_tokens_list = special + sorted(all_tokens)

        self.token_to_idx = {t: i for i, t in enumerate(all_tokens_list)}
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        self.vocab_size = len(all_tokens_list)

    def encode(self, text: str) -> List[int]:
        """Convert S-expression to token indices."""
        sos = self.token_to_idx[self.sos_token]
        eos = self.token_to_idx[self.eos_token]
        unk = self.token_to_idx[self.unk_token]

        tokens = [sos]
        for t in self._tokenize(text):
            tokens.append(self.token_to_idx.get(t, unk))
        tokens.append(eos)
        return tokens

    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to S-expression."""
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, self.unk_token)
            if token in [self.pad_token, self.sos_token, self.eos_token]:
                continue
            tokens.append(token)
        return ' '.join(tokens)

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[self.pad_token]


# ============================================================================
# PART 4: DATASET AND DATALOADER
# ============================================================================

class CodeDataset(Dataset):
    """PyTorch dataset for code/AST sequences."""

    def __init__(self, sequences: List[List[int]], pad_idx: int):
        self.sequences = sequences
        self.pad_idx = pad_idx

        # Find max length for padding
        self.max_len = max(len(s) for s in sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]

        # Input is all but last token, target is all but first
        input_seq = seq[:-1]
        target_seq = seq[1:]

        # Pad to max_len - 1 (since we removed one token)
        pad_len = self.max_len - 1 - len(input_seq)

        input_padded = input_seq + [self.pad_idx] * pad_len
        target_padded = target_seq + [self.pad_idx] * pad_len

        return (
            torch.tensor(input_padded, dtype=torch.long),
            torch.tensor(target_padded, dtype=torch.long)
        )


# ============================================================================
# PART 5: THE TINY TRANSFORMER MODEL
# ============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TinyTransformer(nn.Module):
    """
    Minimal transformer for sequence prediction.

    Architecture:
    - Embedding + Positional Encoding
    - N transformer encoder layers
    - Linear output projection

    ~1M parameters with default settings.
    """

    def __init__(self, vocab_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 256,
                 dropout: float = 0.1, max_len: int = 512):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, vocab_size)

        # Causal mask for autoregressive prediction
        self.register_buffer('causal_mask', None)

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x, padding_mask=None):
        # x: (batch, seq_len)
        seq_len = x.size(1)

        # Embedding + positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        # Causal mask
        causal_mask = self._get_causal_mask(seq_len, x.device)

        # Transformer
        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)

        # Output projection
        logits = self.output_proj(x)

        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================================
# PART 6: TRAINING AND EVALUATION
# ============================================================================

class Trainer:
    """Training loop with metrics tracking."""

    def __init__(self, model: nn.Module, tokenizer, device: torch.device,
                 learning_rate: float = 1e-3):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_idx)
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

        # Metrics tracking
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'samples_seen': [],
            'epoch': []
        }
        self.total_samples_seen = 0

    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Train for one epoch, return (loss, accuracy)."""
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_tokens = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Create padding mask
            padding_mask = (inputs == self.tokenizer.pad_idx)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(inputs, padding_mask)

            # Compute loss
            loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Compute accuracy (excluding padding)
            preds = logits.argmax(dim=-1)
            mask = targets != self.tokenizer.pad_idx
            correct = ((preds == targets) & mask).sum().item()

            total_loss += loss.item() * inputs.size(0)
            total_correct += correct
            total_tokens += mask.sum().item()

            self.total_samples_seen += inputs.size(0)

        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        # Record metrics
        self.history['train_loss'].append(avg_loss)
        self.history['train_acc'].append(accuracy)
        self.history['samples_seen'].append(self.total_samples_seen)
        self.history['epoch'].append(epoch)

        return avg_loss, accuracy

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Evaluate on a dataset, return (loss, accuracy)."""
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                padding_mask = (inputs == self.tokenizer.pad_idx)
                logits = self.model(inputs, padding_mask)

                loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

                preds = logits.argmax(dim=-1)
                mask = targets != self.tokenizer.pad_idx
                correct = ((preds == targets) & mask).sum().item()

                total_loss += loss.item() * inputs.size(0)
                total_correct += correct
                total_tokens += mask.sum().item()

        avg_loss = total_loss / len(dataloader.dataset)
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        return avg_loss, accuracy


# ============================================================================
# PART 7: THE MAIN EXPERIMENT
# ============================================================================

def run_experiment(n_samples: int = 5000,
                   n_epochs: int = 50,
                   batch_size: int = 64,
                   tiers: List[str] = ['linear'],
                   depths: List[int] = [1, 2],
                   device_str: str = 'auto',
                   verbose: bool = True) -> Dict:
    """
    Run the AST vs Char-Level race.

    Args:
        n_samples: Total training samples
        n_epochs: Number of training epochs
        batch_size: Batch size
        tiers: Which complexity tiers to include
        depths: Which expression depths to include
        device_str: 'cpu', 'cuda', or 'auto'
        verbose: Print progress

    Returns:
        Dictionary with results and histories
    """

    # Setup device
    if device_str == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_str)

    if verbose:
        print(f"=" * 60)
        print("AST-JEPA Phase 0: The Race")
        print(f"=" * 60)
        print(f"Device: {device}")
        print(f"Samples: {n_samples}, Epochs: {n_epochs}, Batch: {batch_size}")
        print(f"Tiers: {tiers}, Depths: {depths}")
        print()

    # Generate data
    if verbose:
        print("Generating dataset...")

    generator = DataGenerator(seed=SEED)
    data = generator.generate_dataset(n_samples, tiers=tiers, depths=depths)

    if verbose:
        print(f"  Generated {len(data['code'])} samples")
        print(f"  Sample code: {data['code'][0][:50]}...")
        print(f"  Sample AST:  {data['ast'][0][:50]}...")

    # Analyze sequence lengths
    code_lens = [len(c) for c in data['code']]
    ast_lens = [len(a) for a in data['ast']]

    if verbose:
        print(f"\nSequence length analysis:")
        print(f"  Code - Mean: {sum(code_lens)/len(code_lens):.1f}, Max: {max(code_lens)}")
        print(f"  AST  - Mean: {sum(ast_lens)/len(ast_lens):.1f}, Max: {max(ast_lens)}")
        print(f"  Expansion ratio: {sum(ast_lens)/sum(code_lens):.2f}x")

    # Build tokenizers
    if verbose:
        print("\nBuilding tokenizers...")

    char_tokenizer = CharTokenizer()
    char_tokenizer.fit(data['code'])

    ast_tokenizer = ASTTokenizer()
    ast_tokenizer.fit(data['ast'])

    if verbose:
        print(f"  Char vocab size: {char_tokenizer.vocab_size}")
        print(f"  AST vocab size:  {ast_tokenizer.vocab_size}")

    # Encode data
    char_sequences = [char_tokenizer.encode(c) for c in data['code']]
    ast_sequences = [ast_tokenizer.encode(a) for a in data['ast']]

    # Analyze token sequence lengths
    char_tok_lens = [len(s) for s in char_sequences]
    ast_tok_lens = [len(s) for s in ast_sequences]

    if verbose:
        print(f"\nToken sequence lengths:")
        print(f"  Char - Mean: {sum(char_tok_lens)/len(char_tok_lens):.1f}, Max: {max(char_tok_lens)}")
        print(f"  AST  - Mean: {sum(ast_tok_lens)/len(ast_tok_lens):.1f}, Max: {max(ast_tok_lens)}")

    # Create datasets
    char_dataset = CodeDataset(char_sequences, char_tokenizer.pad_idx)
    ast_dataset = CodeDataset(ast_sequences, ast_tokenizer.pad_idx)

    char_loader = DataLoader(char_dataset, batch_size=batch_size, shuffle=True)
    ast_loader = DataLoader(ast_dataset, batch_size=batch_size, shuffle=True)

    # Create models
    if verbose:
        print("\nCreating models...")

    char_model = TinyTransformer(
        vocab_size=char_tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    )

    ast_model = TinyTransformer(
        vocab_size=ast_tokenizer.vocab_size,
        d_model=128,
        nhead=4,
        num_layers=2,
        dim_feedforward=256
    )

    if verbose:
        print(f"  Char model params: {char_model.count_parameters():,}")
        print(f"  AST model params:  {ast_model.count_parameters():,}")

    # Create trainers
    char_trainer = Trainer(char_model, char_tokenizer, device)
    ast_trainer = Trainer(ast_model, ast_tokenizer, device)

    # Training loop
    if verbose:
        print("\n" + "=" * 60)
        print("Training...")
        print("=" * 60)

    start_time = time.time()

    for epoch in range(n_epochs):
        char_loss, char_acc = char_trainer.train_epoch(char_loader, epoch)
        ast_loss, ast_acc = ast_trainer.train_epoch(ast_loader, epoch)

        if verbose and (epoch + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1:3d} | "
                  f"Char: loss={char_loss:.4f}, acc={char_acc:.3f} | "
                  f"AST: loss={ast_loss:.4f}, acc={ast_acc:.3f} | "
                  f"Time: {elapsed:.1f}s")

    total_time = time.time() - start_time

    if verbose:
        print(f"\nTraining complete in {total_time:.1f}s")

    # Compute samples to 90% accuracy
    def find_samples_to_threshold(history, threshold=0.9):
        for i, acc in enumerate(history['train_acc']):
            if acc >= threshold:
                return history['samples_seen'][i]
        return None

    char_samples_90 = find_samples_to_threshold(char_trainer.history, 0.9)
    ast_samples_90 = find_samples_to_threshold(ast_trainer.history, 0.9)

    # Results summary
    results = {
        'char_final_acc': char_trainer.history['train_acc'][-1],
        'ast_final_acc': ast_trainer.history['train_acc'][-1],
        'char_samples_to_90': char_samples_90,
        'ast_samples_to_90': ast_samples_90,
        'char_history': char_trainer.history,
        'ast_history': ast_trainer.history,
        'char_vocab_size': char_tokenizer.vocab_size,
        'ast_vocab_size': ast_tokenizer.vocab_size,
        'expansion_ratio': sum(ast_lens) / sum(code_lens),
        'training_time': total_time,
        'n_samples': len(data['code']),
        'config': {
            'n_samples': n_samples,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'tiers': tiers,
            'depths': depths
        }
    }

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Final Accuracy:")
        print(f"  Char-Level: {results['char_final_acc']:.3f}")
        print(f"  AST:        {results['ast_final_acc']:.3f}")
        print()
        print(f"Samples to 90% Accuracy:")
        print(f"  Char-Level: {char_samples_90 if char_samples_90 else 'Not reached'}")
        print(f"  AST:        {ast_samples_90 if ast_samples_90 else 'Not reached'}")

        if char_samples_90 and ast_samples_90:
            speedup = char_samples_90 / ast_samples_90
            print(f"\n  ==> AST is {speedup:.2f}x faster to reach 90% accuracy")

            if speedup >= 2:
                print("\n  *** SUCCESS: AST shows significant sample efficiency advantage! ***")
            elif speedup >= 1.5:
                print("\n  ** MARGINAL: AST shows moderate advantage. Consider optimizations. **")
            else:
                print("\n  * NO ADVANTAGE: AST does not beat char-level on sample efficiency. *")

    return results


def plot_results(results: Dict, save_path: str = 'ast_jepa_results.png'):
    """Generate and save the learning curves plot."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Accuracy vs Samples
    ax1 = axes[0]
    ax1.plot(results['char_history']['samples_seen'],
             results['char_history']['train_acc'],
             'b-', linewidth=2, label='Char-Level (Baseline)')
    ax1.plot(results['ast_history']['samples_seen'],
             results['ast_history']['train_acc'],
             'r-', linewidth=2, label='AST (Ours)')

    ax1.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90% Threshold')

    # Mark samples to 90%
    if results['char_samples_to_90']:
        ax1.axvline(x=results['char_samples_to_90'], color='blue', linestyle=':', alpha=0.7)
        ax1.plot(results['char_samples_to_90'], 0.9, 'bo', markersize=10)
    if results['ast_samples_to_90']:
        ax1.axvline(x=results['ast_samples_to_90'], color='red', linestyle=':', alpha=0.7)
        ax1.plot(results['ast_samples_to_90'], 0.9, 'ro', markersize=10)

    ax1.set_xlabel('Training Samples Seen', fontsize=12)
    ax1.set_ylabel('Training Accuracy', fontsize=12)
    ax1.set_title('Sample Efficiency: AST vs Char-Level', fontsize=14)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Plot 2: Loss vs Epoch
    ax2 = axes[1]
    ax2.plot(results['char_history']['epoch'],
             results['char_history']['train_loss'],
             'b-', linewidth=2, label='Char-Level (Baseline)')
    ax2.plot(results['ast_history']['epoch'],
             results['ast_history']['train_loss'],
             'r-', linewidth=2, label='AST (Ours)')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training Loss', fontsize=12)
    ax2.set_title('Loss Convergence', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add summary text
    speedup_text = "N/A"
    if results['char_samples_to_90'] and results['ast_samples_to_90']:
        speedup = results['char_samples_to_90'] / results['ast_samples_to_90']
        speedup_text = f"{speedup:.2f}x"

    summary = (
        f"Vocab Size: Char={results['char_vocab_size']}, AST={results['ast_vocab_size']}\n"
        f"Expansion Ratio: {results['expansion_ratio']:.2f}x\n"
        f"Sample Efficiency: AST is {speedup_text} faster"
    )

    fig.text(0.5, 0.02, summary, ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


# ============================================================================
# PART 8: ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AST-JEPA PHASE 0: THE RACE")
    print("=" * 60)
    print("\nThis experiment tests whether compressed AST representations")
    print("learn faster than character-level representations.")
    print("\nBased on GrammarCoder (2024) and SIP (2023) insights.")
    print("=" * 60 + "\n")

    # Run with Tier 1 (arithmetic) only for initial test
    # This is the "proof of life" experiment
    results = run_experiment(
        n_samples=5000,
        n_epochs=50,
        batch_size=64,
        tiers=['linear'],  # Start simple: just arithmetic
        depths=[1, 2],     # Flat and nested expressions
        verbose=True
    )

    # Generate plot
    plot_path = plot_results(results, 'ast_jepa_results.png')
    print(f"\nResults plot saved to: {plot_path}")

    # Save results JSON
    # Convert non-serializable items
    results_json = {k: v for k, v in results.items()
                    if k not in ['char_history', 'ast_history']}
    results_json['char_final_loss'] = results['char_history']['train_loss'][-1]
    results_json['ast_final_loss'] = results['ast_history']['train_loss'][-1]

    with open('ast_jepa_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Results JSON saved to: ast_jepa_results.json")

    # Final verdict
    print("\n" + "=" * 60)
    print("PHASE 0 VERDICT")
    print("=" * 60)

    if results['ast_samples_to_90'] and results['char_samples_to_90']:
        speedup = results['char_samples_to_90'] / results['ast_samples_to_90']
        if speedup >= 2:
            print("SUCCESS: Proceed to Phase 1 (Full Complexity Grid)")
            print(f"  AST achieved {speedup:.1f}x sample efficiency.")
        elif speedup >= 1.2:
            print("~ MARGINAL: Consider optimizations before Phase 1")
            print(f"  AST showed {speedup:.1f}x efficiency - needs improvement.")
        else:
            print("X NEGATIVE: AST does not beat char-level")
            print("  Consider pivoting to hybrid approach (Path B).")
    elif results['ast_final_acc'] > results['char_final_acc']:
        print("? INCONCLUSIVE: Neither reached 90%, but AST has higher final accuracy.")
        print("  Try more epochs or simpler task.")
    else:
        print("X NEGATIVE: Char-level performs equal or better than AST.")
        print("  Consider pivoting to hybrid approach (Path B).")
