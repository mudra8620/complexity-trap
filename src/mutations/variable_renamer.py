"""
Variable Renaming Mutation Engine v1.0
======================================

The Smoking Gun Test for Structure vs Surface Learning

This engine proves that AST-based models learn structure, not surface patterns:
- Original: fibonacci(n) - tokens seen millions of times in training
- Mutated:  var_fibonacci_x92a(var_n_k291) - tokens never seen before
- AST:      Identical FunctionDef structure

Token-based models will collapse. Structure-based models will be unfazed.
"""

import ast
import hashlib
import random
import string
from typing import Dict, Tuple, Optional
from copy import deepcopy


class VariableRenamingEngine:
    """
    Surgically mutates variable/function names while preserving AST structure.

    The key insight: If a model truly understands code STRUCTURE,
    renaming `fibonacci(n)` to `xyz_98a(q_12)` should have ZERO impact.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)
        self.rename_map: Dict[str, str] = {}

    def _generate_obfuscated_name(self, original: str, prefix: str = "var") -> str:
        """Generate a unique obfuscated name that no model has seen before."""
        suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        return f"{prefix}_{original}_{suffix}"

    def _collect_all_names(self, tree: ast.AST) -> Dict[str, str]:
        """
        Collect all user-defined names (variables, functions, classes, params).
        Excludes builtins and imports.
        """
        names = {}
        builtins = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
        builtins.update(['print', 'len', 'range', 'int', 'str', 'list', 'dict', 'set',
                        'True', 'False', 'None', 'sum', 'min', 'max', 'abs', 'sorted',
                        'enumerate', 'zip', 'map', 'filter', 'any', 'all', 'open'])

        for node in ast.walk(tree):
            # Function definitions
            if isinstance(node, ast.FunctionDef):
                if node.name not in builtins:
                    names[node.name] = self._generate_obfuscated_name(node.name, "fn")
                # Function arguments
                for arg in node.args.args:
                    if arg.arg not in builtins:
                        names[arg.arg] = self._generate_obfuscated_name(arg.arg, "arg")

            # Class definitions
            elif isinstance(node, ast.ClassDef):
                if node.name not in builtins:
                    names[node.name] = self._generate_obfuscated_name(node.name, "cls")

            # Variable assignments
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id not in builtins:
                        if target.id not in names:
                            names[target.id] = self._generate_obfuscated_name(target.id, "var")

            # For loop variables
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name) and node.target.id not in builtins:
                    if node.target.id not in names:
                        names[node.target.id] = self._generate_obfuscated_name(node.target.id, "iter")

        return names

    def mutate(self, code: str) -> Tuple[str, Dict[str, str]]:
        """
        Apply variable renaming mutation.

        Returns:
            Tuple of (mutated_code, rename_map)
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise ValueError(f"Invalid Python syntax: {e}")

        # Collect all names to rename
        self.rename_map = self._collect_all_names(tree)

        # Apply renaming via AST transformation
        renamer = _NameRenamer(self.rename_map)
        mutated_tree = renamer.visit(tree)
        ast.fix_missing_locations(mutated_tree)

        # Convert back to source code
        mutated_code = ast.unparse(mutated_tree)

        return mutated_code, self.rename_map

    def verify_structure_preserved(self, original: str, mutated: str) -> Tuple[bool, str]:
        """
        Verify that AST structure is IDENTICAL after mutation.
        This is the key proof that we're testing structure, not surface.
        """
        try:
            original_tree = ast.parse(original)
            mutated_tree = ast.parse(mutated)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"

        # Normalize trees by removing all identifier information
        original_normalized = _normalize_ast(original_tree)
        mutated_normalized = _normalize_ast(mutated_tree)

        # Compare structure
        original_dump = ast.dump(original_normalized, annotate_fields=False)
        mutated_dump = ast.dump(mutated_normalized, annotate_fields=False)

        if original_dump == mutated_dump:
            return True, "Deep Structure is 100% Identical"
        else:
            return False, f"Structure differs!\nOriginal: {original_dump[:200]}\nMutated: {mutated_dump[:200]}"


class _NameRenamer(ast.NodeTransformer):
    """AST transformer that renames identifiers according to the rename map."""

    def __init__(self, rename_map: Dict[str, str]):
        self.rename_map = rename_map

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self.rename_map:
            node.id = self.rename_map[node.id]
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        if node.name in self.rename_map:
            node.name = self.rename_map[node.name]
        # Rename arguments
        for arg in node.args.args:
            if arg.arg in self.rename_map:
                arg.arg = self.rename_map[arg.arg]
        # Continue visiting children
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
        if node.name in self.rename_map:
            node.name = self.rename_map[node.name]
        self.generic_visit(node)
        return node

    def visit_arg(self, node: ast.arg) -> ast.arg:
        if node.arg in self.rename_map:
            node.arg = self.rename_map[node.arg]
        return node


def _normalize_ast(tree: ast.AST) -> ast.AST:
    """
    Normalize AST by replacing all identifiers with placeholders.
    This extracts pure STRUCTURE, ignoring surface tokens.
    """
    tree = deepcopy(tree)

    class Normalizer(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = "ID"
            return node

        def visit_FunctionDef(self, node):
            node.name = "FUNC"
            for arg in node.args.args:
                arg.arg = "ARG"
            self.generic_visit(node)
            return node

        def visit_ClassDef(self, node):
            node.name = "CLASS"
            self.generic_visit(node)
            return node

        def visit_arg(self, node):
            node.arg = "ARG"
            return node

        def visit_Constant(self, node):
            # Normalize constants too (optional, for stricter structure testing)
            if isinstance(node.value, str):
                node.value = "STR"
            elif isinstance(node.value, (int, float)):
                node.value = 0
            return node

    return Normalizer().visit(tree)


def generate_n_mutations(code: str, n: int = 50, seed_base: int = 42) -> list:
    """
    Generate N different mutations of the same code.

    For consistency scoring: if a model truly understands structure,
    it should give the SAME answer for all N mutations.
    """
    mutations = []
    for i in range(n):
        engine = VariableRenamingEngine(seed=seed_base + i)
        mutated, rename_map = engine.mutate(code)
        mutations.append({
            'code': mutated,
            'rename_map': rename_map,
            'seed': seed_base + i
        })
    return mutations


# =============================================================================
# DEMO / TEST
# =============================================================================

if __name__ == "__main__":
    # Classic fibonacci - the tokens every model has memorized
    original_code = '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

result = fibonacci(10)
print(result)
'''

    print("=" * 60)
    print("VARIABLE RENAMING MUTATION ENGINE v1.0")
    print("The Smoking Gun Test for Structure vs Surface")
    print("=" * 60)

    print("\n[ORIGINAL CODE]")
    print(original_code)

    # Apply mutation
    engine = VariableRenamingEngine(seed=42)
    mutated_code, rename_map = engine.mutate(original_code)

    print("\n[MUTATED CODE]")
    print(mutated_code)

    print("\n[RENAME MAP]")
    for old, new in rename_map.items():
        print(f"  {old:15} -> {new}")

    # THE KEY TEST: Verify structure is preserved
    print("\n" + "=" * 60)
    print("STRUCTURE VERIFICATION")
    print("=" * 60)

    preserved, message = engine.verify_structure_preserved(original_code, mutated_code)

    if preserved:
        print(f"\n  SUCCESS: {message}")
        print("\n  Token-based models will panic on the mutated code.")
        print("  AST-JEPA will sleep through this - it sees identical structure.")
    else:
        print(f"\n  FAILURE: {message}")

    # Generate multiple mutations for consistency testing
    print("\n" + "=" * 60)
    print("CONSISTENCY TEST SAMPLES (for batch evaluation)")
    print("=" * 60)

    mutations = generate_n_mutations(original_code, n=3)
    for i, m in enumerate(mutations):
        print(f"\n[Mutation {i+1}] Function name: {list(m['rename_map'].values())[0]}")
        # Just show first line to demonstrate variety
        first_line = m['code'].split('\n')[0]
        print(f"  {first_line}")

    print("\n" + "=" * 60)
    print("Ready for integration with training race.")
    print("=" * 60)
