"""
AST Linearization with Positional Variable IDs

This module provides three AST linearization strategies:
1. NamedAST: Keeps variable names (current - has 9% drop issue)
2. PositionalAST: Replaces names with positional IDs (V0, V1, ...) - RECOMMENDED
3. PureAST: Removes names entirely (just structural tags)

The PositionalAST variant should achieve ~0% drop under variable renaming
because the same structural positions map to the same IDs regardless of
original variable names.
"""

import ast
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# =============================================================================
# TAG DEFINITIONS
# =============================================================================

TAGS = {
    # Statements
    'FunctionDef': 'Fn',
    'Return': 'Rt',
    'Assign': 'As',
    'If': 'If',
    'While': 'Wh',
    'For': 'Fr',
    'Pass': 'Ps',
    
    # Expressions
    'BinOp': 'Bo',
    'UnaryOp': 'Uo',
    'Compare': 'Cp',
    'Call': 'Cl',
    'IfExp': 'Ie',  # Ternary
    
    # Operators
    'Add': 'Ad',
    'Sub': 'Sb',
    'Mult': 'Ml',
    'Div': 'Dv',
    'Mod': 'Md',
    'Pow': 'Pw',
    'FloorDiv': 'Fd',
    
    # Comparison operators
    'Eq': 'Eq',
    'NotEq': 'Ne',
    'Lt': 'Lt',
    'LtE': 'Le',
    'Gt': 'Gt',
    'GtE': 'Ge',
    
    # Unary operators
    'USub': 'Us',
    'Not': 'Nt',
    
    # Boolean operators
    'And': 'An',
    'Or': 'Or',
    
    # Atoms
    'Name': 'Nm',
    'Constant': 'Cn',
    'Num': 'Cn',  # Python 3.7 compatibility
    'Str': 'St',
    
    # Structure
    'arguments': 'Ar',
    'arg': 'Ag',
}


# =============================================================================
# STRATEGY 1: NAMED AST (Current - Has 9% Drop Issue)
# =============================================================================

class NamedASTVisitor(ast.NodeVisitor):
    """
    Converts AST to S-expression, KEEPING variable names.
    
    Example:
        def add(a, b): return a + b
        ->
        (Fn add (Ar a b) (Rt (Bo Ad (Nm a) (Nm b))))
    
    Problem: 'a' and 'b' appear in output, so renaming to 'x', 'y' changes it:
        (Fn add (Ar x y) (Rt (Bo Ad (Nm x) (Nm y))))
    
    This is NOT structurally invariant. The model can still learn name patterns.
    """
    
    def visit_Module(self, node):
        return " ".join(self.visit(child) for child in node.body)
    
    def visit_FunctionDef(self, node):
        name = node.name
        args = self.visit(node.args)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        return f"(Fn {name} {args} {body})"
    
    def visit_arguments(self, node):
        args = " ".join(arg.arg for arg in node.args)
        return f"(Ar {args})"
    
    def visit_Return(self, node):
        value = self.visit(node.value) if node.value else ""
        return f"(Rt {value})"
    
    def visit_BinOp(self, node):
        op = TAGS.get(type(node.op).__name__, '?')
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"(Bo {op} {left} {right})"
    
    def visit_Compare(self, node):
        left = self.visit(node.left)
        ops = " ".join(TAGS.get(type(op).__name__, '?') for op in node.ops)
        comparators = " ".join(self.visit(c) for c in node.comparators)
        return f"(Cp {left} {ops} {comparators})"
    
    def visit_Name(self, node):
        return f"(Nm {node.id})"  # <-- KEEPS THE NAME
    
    def visit_Constant(self, node):
        return f"(Cn {node.value})"
    
    def visit_Num(self, node):  # Python 3.7
        return f"(Cn {node.n})"
    
    def visit_If(self, node):
        test = self.visit(node.test)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        orelse = " ".join(self.visit(stmt) for stmt in node.orelse) if node.orelse else ""
        if orelse:
            return f"(If {test} {body} {orelse})"
        return f"(If {test} {body})"
    
    def visit_IfExp(self, node):
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"(Ie {test} {body} {orelse})"
    
    def visit_While(self, node):
        test = self.visit(node.test)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        return f"(Wh {test} {body})"
    
    def visit_For(self, node):
        target = self.visit(node.target)
        iter_expr = self.visit(node.iter)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        return f"(Fr {target} {iter_expr} {body})"
    
    def visit_Call(self, node):
        func = self.visit(node.func)
        args = " ".join(self.visit(arg) for arg in node.args)
        return f"(Cl {func} {args})"
    
    def visit_Assign(self, node):
        targets = " ".join(self.visit(t) for t in node.targets)
        value = self.visit(node.value)
        return f"(As {targets} {value})"
    
    def visit_UnaryOp(self, node):
        op = TAGS.get(type(node.op).__name__, '?')
        operand = self.visit(node.operand)
        return f"(Uo {op} {operand})"
    
    def visit_BoolOp(self, node):
        op = TAGS.get(type(node.op).__name__, '?')
        values = " ".join(self.visit(v) for v in node.values)
        return f"(Bl {op} {values})"
    
    def generic_visit(self, node):
        return f"(?{type(node).__name__})"


# =============================================================================
# STRATEGY 2: POSITIONAL AST (Recommended - Should Have ~0% Drop)
# =============================================================================

class PositionalASTVisitor(ast.NodeVisitor):
    """
    Converts AST to S-expression, replacing variable names with POSITIONAL IDs.
    
    The first unique name encountered becomes V0, the second V1, etc.
    This ensures that:
    - def add(a, b): return a + b
    - def fn_x(var1, var2): return var1 + var2
    
    Both become:
        (Fn V0 (Ar V1 V2) (Rt (Bo Ad (Nm V1) (Nm V2))))
    
    The function name is always V0, first param is V1, etc.
    This is STRUCTURALLY INVARIANT to renaming.
    """
    
    def __init__(self):
        self.name_map: Dict[str, str] = {}
        self.name_counter = 0
    
    def _get_positional_id(self, name: str) -> str:
        """Get or create positional ID for a variable name."""
        if name not in self.name_map:
            self.name_map[name] = f"V{self.name_counter}"
            self.name_counter += 1
        return self.name_map[name]
    
    def reset(self):
        """Reset for a new function."""
        self.name_map = {}
        self.name_counter = 0
    
    def visit_Module(self, node):
        results = []
        for child in node.body:
            self.reset()  # Reset per function
            results.append(self.visit(child))
        return " ".join(results)
    
    def visit_FunctionDef(self, node):
        # Function name is always first (V0)
        func_id = self._get_positional_id(node.name)
        args = self.visit(node.args)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        return f"(Fn {func_id} {args} {body})"
    
    def visit_arguments(self, node):
        # Parameters get sequential IDs (V1, V2, ...)
        arg_ids = " ".join(self._get_positional_id(arg.arg) for arg in node.args)
        return f"(Ar {arg_ids})"
    
    def visit_Return(self, node):
        value = self.visit(node.value) if node.value else ""
        return f"(Rt {value})"
    
    def visit_BinOp(self, node):
        op = TAGS.get(type(node.op).__name__, '?')
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"(Bo {op} {left} {right})"
    
    def visit_Compare(self, node):
        left = self.visit(node.left)
        ops = " ".join(TAGS.get(type(op).__name__, '?') for op in node.ops)
        comparators = " ".join(self.visit(c) for c in node.comparators)
        return f"(Cp {left} {ops} {comparators})"
    
    def visit_Name(self, node):
        pos_id = self._get_positional_id(node.id)
        return f"(Nm {pos_id})"  # <-- POSITIONAL ID, NOT ORIGINAL NAME
    
    def visit_Constant(self, node):
        return f"(Cn {node.value})"
    
    def visit_Num(self, node):
        return f"(Cn {node.n})"
    
    def visit_If(self, node):
        test = self.visit(node.test)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        orelse = " ".join(self.visit(stmt) for stmt in node.orelse) if node.orelse else ""
        if orelse:
            return f"(If {test} {body} {orelse})"
        return f"(If {test} {body})"
    
    def visit_IfExp(self, node):
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"(Ie {test} {body} {orelse})"
    
    def visit_While(self, node):
        test = self.visit(node.test)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        return f"(Wh {test} {body})"
    
    def visit_For(self, node):
        target = self.visit(node.target)
        iter_expr = self.visit(node.iter)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        return f"(Fr {target} {iter_expr} {body})"
    
    def visit_Call(self, node):
        func = self.visit(node.func)
        args = " ".join(self.visit(arg) for arg in node.args)
        return f"(Cl {func} {args})"
    
    def visit_Assign(self, node):
        targets = " ".join(self.visit(t) for t in node.targets)
        value = self.visit(node.value)
        return f"(As {targets} {value})"
    
    def visit_UnaryOp(self, node):
        op = TAGS.get(type(node.op).__name__, '?')
        operand = self.visit(node.operand)
        return f"(Uo {op} {operand})"
    
    def visit_BoolOp(self, node):
        op = TAGS.get(type(node.op).__name__, '?')
        values = " ".join(self.visit(v) for v in node.values)
        return f"(Bl {op} {values})"
    
    def generic_visit(self, node):
        return f"(?{type(node).__name__})"


# =============================================================================
# STRATEGY 3: PURE STRUCTURAL AST (No Names At All)
# =============================================================================

class PureStructuralASTVisitor(ast.NodeVisitor):
    """
    Converts AST to S-expression with NO variable names at all.
    
    Example:
        def add(a, b): return a + b
        ->
        (Fn (Ar) (Ar) (Rt (Bo Ad (Nm) (Nm))))
    
    This is MAXIMALLY invariant but loses information about which
    variable is used where (a+a vs a+b would look the same in body).
    
    Use this to establish the "ceiling" of structural invariance.
    """
    
    def visit_Module(self, node):
        return " ".join(self.visit(child) for child in node.body)
    
    def visit_FunctionDef(self, node):
        n_args = len(node.args.args)
        args = " ".join("(Ag)" for _ in range(n_args))
        body = " ".join(self.visit(stmt) for stmt in node.body)
        return f"(Fn ({args}) {body})"
    
    def visit_arguments(self, node):
        return f"(Ar)"  # Just tag, no content
    
    def visit_Return(self, node):
        value = self.visit(node.value) if node.value else ""
        return f"(Rt {value})"
    
    def visit_BinOp(self, node):
        op = TAGS.get(type(node.op).__name__, '?')
        left = self.visit(node.left)
        right = self.visit(node.right)
        return f"(Bo {op} {left} {right})"
    
    def visit_Compare(self, node):
        left = self.visit(node.left)
        ops = " ".join(TAGS.get(type(op).__name__, '?') for op in node.ops)
        comparators = " ".join(self.visit(c) for c in node.comparators)
        return f"(Cp {left} {ops} {comparators})"
    
    def visit_Name(self, node):
        return "(Nm)"  # <-- NO NAME, JUST THE TAG
    
    def visit_Constant(self, node):
        # Keep constants (they're not variable names)
        return f"(Cn {node.value})"
    
    def visit_Num(self, node):
        return f"(Cn {node.n})"
    
    def visit_If(self, node):
        test = self.visit(node.test)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        orelse = " ".join(self.visit(stmt) for stmt in node.orelse) if node.orelse else ""
        if orelse:
            return f"(If {test} {body} {orelse})"
        return f"(If {test} {body})"
    
    def visit_IfExp(self, node):
        test = self.visit(node.test)
        body = self.visit(node.body)
        orelse = self.visit(node.orelse)
        return f"(Ie {test} {body} {orelse})"
    
    def visit_While(self, node):
        test = self.visit(node.test)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        return f"(Wh {test} {body})"
    
    def visit_For(self, node):
        target = self.visit(node.target)
        iter_expr = self.visit(node.iter)
        body = " ".join(self.visit(stmt) for stmt in node.body)
        return f"(Fr {target} {iter_expr} {body})"
    
    def visit_Call(self, node):
        func = self.visit(node.func)
        args = " ".join(self.visit(arg) for arg in node.args)
        return f"(Cl {func} {args})"
    
    def visit_Assign(self, node):
        targets = " ".join(self.visit(t) for t in node.targets)
        value = self.visit(node.value)
        return f"(As {targets} {value})"
    
    def visit_UnaryOp(self, node):
        op = TAGS.get(type(node.op).__name__, '?')
        operand = self.visit(node.operand)
        return f"(Uo {op} {operand})"
    
    def visit_BoolOp(self, node):
        op = TAGS.get(type(node.op).__name__, '?')
        values = " ".join(self.visit(v) for v in node.values)
        return f"(Bl {op} {values})"
    
    def generic_visit(self, node):
        return f"(?{type(node).__name__})"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def code_to_ast(code: str, strategy: str = 'positional') -> str:
    """
    Convert Python code to AST S-expression.
    
    Args:
        code: Python source code string
        strategy: 'named', 'positional', or 'pure'
    
    Returns:
        S-expression string
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "ERR"
    
    visitors = {
        'named': NamedASTVisitor,
        'positional': PositionalASTVisitor,
        'pure': PureStructuralASTVisitor,
    }
    
    visitor_class = visitors.get(strategy, PositionalASTVisitor)
    visitor = visitor_class()
    
    try:
        return visitor.visit(tree)
    except Exception:
        return "ERR"


def demonstrate_invariance():
    """Demonstrate that positional AST is invariant to renaming."""
    
    original = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)"""
    
    renamed = """def fn_xyz_k29(arg_q_81):
    if arg_q_81 <= 1:
        return arg_q_81
    return fn_xyz_k29(arg_q_81 - 1) + fn_xyz_k29(arg_q_81 - 2)"""
    
    print("=" * 70)
    print("DEMONSTRATING STRUCTURAL INVARIANCE")
    print("=" * 70)
    
    for strategy in ['named', 'positional', 'pure']:
        print(f"\n{'=' * 70}")
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 70)
        
        ast_orig = code_to_ast(original, strategy)
        ast_renamed = code_to_ast(renamed, strategy)
        
        print(f"\nOriginal code:\n{original}\n")
        print(f"AST: {ast_orig[:100]}...")
        
        print(f"\nRenamed code:\n{renamed}\n")
        print(f"AST: {ast_renamed[:100]}...")
        
        is_identical = ast_orig == ast_renamed
        print(f"\n{'✓ IDENTICAL' if is_identical else '✗ DIFFERENT'} - ", end="")
        print(f"Invariant: {is_identical}")


if __name__ == "__main__":
    demonstrate_invariance()
    
    # Quick test
    print("\n" + "=" * 70)
    print("QUICK TEST: Simple Addition")
    print("=" * 70)
    
    code1 = "def add(a, b): return a + b"
    code2 = "def fn_x(var1, var2): return var1 + var2"
    
    for strategy in ['named', 'positional', 'pure']:
        ast1 = code_to_ast(code1, strategy)
        ast2 = code_to_ast(code2, strategy)
        print(f"\n{strategy.upper()}:")
        print(f"  Code 1: {ast1}")
        print(f"  Code 2: {ast2}")
        print(f"  Match: {'✓' if ast1 == ast2 else '✗'}")
