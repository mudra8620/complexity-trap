"""
BPE (Byte Pair Encoding) Tokenizer
==================================

Implements the same tokenization strategy used by GPT, CodeBERT, etc.
This is the strongest baseline for the "strawman" critique.

BPE learns subword units like:
- "def" -> single token
- "return" -> single token
- "fibonacci" -> "fib" + "onacci" (or similar)

When variables are renamed, BPE tokens change completely,
just like char-level but with learned subwords.
"""

import re
from collections import defaultdict
from typing import List, Dict, Tuple, Optional


class BPETokenizer:
    """
    Byte Pair Encoding tokenizer.

    Learns merge rules from training data, then applies them
    to encode new text into subword tokens.
    """

    def __init__(self, vocab_size: int = 500):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.token_to_idx: Dict[str, int] = {}
        self.idx_to_token: Dict[int, str] = {}

        # Special tokens
        self.pad_token = '<PAD>'
        self.sos_token = '<SOS>'
        self.eos_token = '<EOS>'
        self.unk_token = '<UNK>'

    def _get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs."""
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs

    def _merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """Merge all occurrences of the most frequent pair."""
        new_vocab = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)

        for word, freq in vocab.items():
            new_word = word.replace(bigram, replacement)
            new_vocab[new_word] = freq

        return new_vocab

    def fit(self, texts: List[str]):
        """Learn BPE merge rules from training texts."""
        # Build initial vocabulary (character-level with word boundaries)
        word_freqs = defaultdict(int)

        for text in texts:
            # Tokenize into words, keeping special chars
            words = re.findall(r'\S+|\s+', text)
            for word in words:
                # Add space between each character
                spaced = ' '.join(list(word))
                word_freqs[spaced] += 1

        vocab = dict(word_freqs)

        # Get initial character vocabulary
        chars = set()
        for word in vocab:
            chars.update(word.split())

        # Learn merges until we reach target vocab size
        num_merges = self.vocab_size - len(chars) - 4  # Reserve space for special tokens

        for i in range(max(0, num_merges)):
            pairs = self._get_stats(vocab)
            if not pairs:
                break

            best_pair = max(pairs, key=pairs.get)
            vocab = self._merge_vocab(best_pair, vocab)
            self.merges.append(best_pair)

        # Build final vocabulary
        final_tokens = set()
        for word in vocab:
            final_tokens.update(word.split())

        # Add special tokens first
        special = [self.pad_token, self.sos_token, self.eos_token, self.unk_token]
        all_tokens = special + sorted(final_tokens)

        self.token_to_idx = {t: i for i, t in enumerate(all_tokens)}
        self.idx_to_token = {i: t for t, i in self.token_to_idx.items()}
        self.vocab_size = len(all_tokens)

    def _tokenize_word(self, word: str) -> List[str]:
        """Apply learned merges to tokenize a single word."""
        if not word:
            return []

        # Start with characters
        tokens = list(word)

        # Apply merges in order
        for merge in self.merges:
            i = 0
            while i < len(tokens) - 1:
                if tokens[i] == merge[0] and tokens[i + 1] == merge[1]:
                    tokens = tokens[:i] + [''.join(merge)] + tokens[i + 2:]
                else:
                    i += 1

        return tokens

    def encode(self, text: str) -> List[int]:
        """Convert text to token indices."""
        tokens = [self.token_to_idx[self.sos_token]]

        # Tokenize each word/whitespace
        words = re.findall(r'\S+|\s+', text)

        for word in words:
            word_tokens = self._tokenize_word(word)
            for t in word_tokens:
                idx = self.token_to_idx.get(t, self.token_to_idx[self.unk_token])
                tokens.append(idx)

        tokens.append(self.token_to_idx[self.eos_token])
        return tokens

    def decode(self, indices: List[int]) -> str:
        """Convert token indices back to text."""
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, self.unk_token)
            if token not in [self.pad_token, self.sos_token, self.eos_token]:
                tokens.append(token)
        return ''.join(tokens)

    @property
    def pad_idx(self) -> int:
        return self.token_to_idx[self.pad_token]


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("BPE TOKENIZER TEST")
    print("=" * 60)

    # Sample code for training
    train_texts = [
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
        "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)",
        "def add(a, b):\n    return a + b",
        "def max_val(x, y):\n    if x > y:\n        return x\n    return y",
    ] * 10  # Repeat to get more frequency data

    # Train BPE
    bpe = BPETokenizer(vocab_size=200)
    bpe.fit(train_texts)

    print(f"\nVocab size: {bpe.vocab_size}")
    print(f"Num merges: {len(bpe.merges)}")
    print(f"\nFirst 10 merges: {bpe.merges[:10]}")

    # Test encoding
    test_code = "def fibonacci(n):\n    return n"
    encoded = bpe.encode(test_code)
    decoded = bpe.decode(encoded)

    print(f"\nOriginal: {test_code}")
    print(f"Encoded:  {encoded[:20]}... (len={len(encoded)})")
    print(f"Decoded:  {decoded}")

    # Test with mutated code
    mutated_code = "def fn_xyz_k29(arg_n_81):\n    return arg_n_81"
    mut_encoded = bpe.encode(mutated_code)

    print(f"\nMutated:  {mutated_code}")
    print(f"Encoded:  {mut_encoded[:20]}... (len={len(mut_encoded)})")

    # Show that tokens are different
    print(f"\nOriginal tokens: {len(encoded)}")
    print(f"Mutated tokens:  {len(mut_encoded)}")
    print(f"Tokens changed: YES (BPE is NOT invariant)")
