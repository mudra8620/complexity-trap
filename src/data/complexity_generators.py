"""
Complexity Tier Data Generators

Generates synthetic Python functions across four tiers of complexity:
- Tier 0: Linear (arithmetic expressions only)
- Tier 1: Conditional (if/else statements)
- Tier 2: Loop (for/while loops)
- Tier 3: Recursion (self-referential functions)

Each tier can be generated at different expression depths.
"""

import random
import ast
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# =============================================================================
# CONFIGURATION
# =============================================================================

VARIABLE_NAMES = ['a', 'b', 'c', 'd', 'x', 'y', 'z', 'n', 'm', 'i', 'j', 'k']
BINARY_OPS = ['+', '-', '*']  # Exclude '/' to avoid division by zero
COMPARISON_OPS = ['>', '<', '>=', '<=', '==', '!=']
CONSTANTS = list(range(1, 11))  # 1-10

# Semantic categories for classification
CATEGORIES = {
    'linear': [
        'addition', 'subtraction', 'multiplication', 
        'mixed_arithmetic', 'polynomial'
    ],
    'conditional': [
        'max_min', 'absolute_value', 'sign_check',
        'range_check', 'equality_check'
    ],
    'loop': [
        'summation', 'product', 'counting',
        'accumulation', 'iteration'
    ],
    'recursion': [
        'factorial_like', 'fibonacci_like', 'sum_recursive',
        'countdown', 'power_recursive'
    ]
}


@dataclass
class GeneratedFunction:
    """A generated function with its metadata."""
    code: str
    tier: str
    category: str
    depth: int
    n_variables: int


# =============================================================================
# TIER 0: LINEAR (Arithmetic Only)
# =============================================================================

def generate_linear_expression(variables: List[str], depth: int) -> str:
    """Generate a linear arithmetic expression."""
    if depth <= 1 or random.random() < 0.3:
        # Base case: variable or constant
        if random.random() < 0.7:
            return random.choice(variables)
        else:
            return str(random.choice(CONSTANTS))
    
    # Recursive case: binary operation
    op = random.choice(BINARY_OPS)
    left = generate_linear_expression(variables, depth - 1)
    right = generate_linear_expression(variables, depth - 1)
    
    return f"({left} {op} {right})"


def generate_linear_function(n_vars: int = 3, depth: int = 2) -> GeneratedFunction:
    """Generate a Tier 0 (linear) function."""
    variables = random.sample(VARIABLE_NAMES, n_vars)
    params = ", ".join(variables)
    
    expr = generate_linear_expression(variables, depth)
    
    # Determine category based on expression
    if '+' in expr and '-' not in expr and '*' not in expr:
        category = 'addition'
    elif '-' in expr and '+' not in expr and '*' not in expr:
        category = 'subtraction'
    elif '*' in expr and '+' not in expr and '-' not in expr:
        category = 'multiplication'
    elif '*' in expr:
        category = 'polynomial'
    else:
        category = 'mixed_arithmetic'
    
    code = f"def func({params}):\n    return {expr}"
    
    return GeneratedFunction(
        code=code,
        tier='linear',
        category=category,
        depth=depth,
        n_variables=n_vars
    )


# =============================================================================
# TIER 1: CONDITIONAL (If/Else)
# =============================================================================

def generate_conditional_function(n_vars: int = 2, depth: int = 2) -> GeneratedFunction:
    """Generate a Tier 1 (conditional) function."""
    variables = random.sample(VARIABLE_NAMES, n_vars)
    params = ", ".join(variables)
    
    templates = [
        # Max/Min pattern
        {
            'template': '''def func({params}):
    if {v1} > {v2}:
        return {v1}
    else:
        return {v2}''',
            'category': 'max_min'
        },
        # Absolute value pattern
        {
            'template': '''def func({params}):
    if {v1} >= 0:
        return {v1}
    else:
        return -{v1}''',
            'category': 'absolute_value'
        },
        # Sign check pattern
        {
            'template': '''def func({params}):
    if {v1} > 0:
        return 1
    elif {v1} < 0:
        return -1
    else:
        return 0''',
            'category': 'sign_check'
        },
        # Range check pattern
        {
            'template': '''def func({params}):
    if {v1} > {const1} and {v1} < {const2}:
        return {v1}
    else:
        return 0''',
            'category': 'range_check'
        },
        # Equality check pattern
        {
            'template': '''def func({params}):
    if {v1} == {v2}:
        return {expr1}
    else:
        return {expr2}''',
            'category': 'equality_check'
        },
        # Ternary pattern
        {
            'template': '''def func({params}):
    return {v1} if {v1} > {v2} else {v2}''',
            'category': 'max_min'
        },
    ]
    
    template_info = random.choice(templates)
    template = template_info['template']
    category = template_info['category']
    
    v1, v2 = variables[0], variables[1] if len(variables) > 1 else variables[0]
    const1, const2 = sorted(random.sample(CONSTANTS, 2))
    expr1 = generate_linear_expression(variables, depth - 1)
    expr2 = generate_linear_expression(variables, depth - 1)
    
    code = template.format(
        params=params,
        v1=v1, v2=v2,
        const1=const1, const2=const2,
        expr1=expr1, expr2=expr2
    )
    
    return GeneratedFunction(
        code=code,
        tier='conditional',
        category=category,
        depth=depth,
        n_variables=n_vars
    )


# =============================================================================
# TIER 2: LOOP (For/While)
# =============================================================================

def generate_loop_function(n_vars: int = 2, depth: int = 2) -> GeneratedFunction:
    """Generate a Tier 2 (loop) function."""
    variables = random.sample(VARIABLE_NAMES, n_vars)
    params = ", ".join(variables)
    
    # Use 'n' as loop bound if available, else first variable
    bound_var = 'n' if 'n' in variables else variables[0]
    
    templates = [
        # Summation pattern (for loop)
        {
            'template': '''def func({params}):
    result = 0
    for i in range({bound}):
        result = result + i
    return result''',
            'category': 'summation'
        },
        # Summation with variable (for loop)
        {
            'template': '''def func({params}):
    result = 0
    for i in range({bound}):
        result = result + {v1}
    return result''',
            'category': 'accumulation'
        },
        # Product pattern (for loop)
        {
            'template': '''def func({params}):
    result = 1
    for i in range(1, {bound} + 1):
        result = result * i
    return result''',
            'category': 'product'
        },
        # Counting pattern (for loop)
        {
            'template': '''def func({params}):
    count = 0
    for i in range({bound}):
        if i > {const}:
            count = count + 1
    return count''',
            'category': 'counting'
        },
        # While loop summation
        {
            'template': '''def func({params}):
    result = 0
    i = 0
    while i < {bound}:
        result = result + i
        i = i + 1
    return result''',
            'category': 'summation'
        },
        # While loop with condition
        {
            'template': '''def func({params}):
    result = {v1}
    while result > {const}:
        result = result - 1
    return result''',
            'category': 'iteration'
        },
    ]
    
    template_info = random.choice(templates)
    template = template_info['template']
    category = template_info['category']
    
    v1 = variables[0]
    const = random.choice(CONSTANTS)
    
    code = template.format(
        params=params,
        bound=bound_var,
        v1=v1,
        const=const
    )
    
    return GeneratedFunction(
        code=code,
        tier='loop',
        category=category,
        depth=depth,
        n_variables=n_vars
    )


# =============================================================================
# TIER 3: RECURSION
# =============================================================================

def generate_recursive_function(n_vars: int = 1, depth: int = 2) -> GeneratedFunction:
    """Generate a Tier 3 (recursive) function."""
    # Recursive functions typically have fewer parameters
    n_vars = min(n_vars, 2)
    variables = random.sample(['n', 'x', 'a', 'b'], n_vars)
    params = ", ".join(variables)
    
    main_var = variables[0]
    
    templates = [
        # Factorial-like pattern
        {
            'template': '''def func({params}):
    if {v1} <= 1:
        return 1
    return {v1} * func({v1} - 1)''',
            'category': 'factorial_like'
        },
        # Fibonacci-like pattern
        {
            'template': '''def func({params}):
    if {v1} <= 1:
        return {v1}
    return func({v1} - 1) + func({v1} - 2)''',
            'category': 'fibonacci_like'
        },
        # Sum recursive pattern
        {
            'template': '''def func({params}):
    if {v1} <= 0:
        return 0
    return {v1} + func({v1} - 1)''',
            'category': 'sum_recursive'
        },
        # Countdown pattern
        {
            'template': '''def func({params}):
    if {v1} <= 0:
        return 0
    return 1 + func({v1} - 1)''',
            'category': 'countdown'
        },
        # Power recursive pattern (if 2 vars)
        {
            'template': '''def func({params}):
    if {v1} <= 0:
        return 1
    return {const} * func({v1} - 1)''',
            'category': 'power_recursive'
        },
    ]
    
    template_info = random.choice(templates)
    template = template_info['template']
    category = template_info['category']
    
    const = random.choice([2, 3, 5])
    
    code = template.format(
        params=params,
        v1=main_var,
        const=const
    )
    
    return GeneratedFunction(
        code=code,
        tier='recursion',
        category=category,
        depth=depth,
        n_variables=n_vars
    )


# =============================================================================
# MAIN GENERATOR
# =============================================================================

TIER_GENERATORS = {
    'linear': generate_linear_function,
    'conditional': generate_conditional_function,
    'loop': generate_loop_function,
    'recursion': generate_recursive_function,
}


def generate_dataset(
    n_samples: int,
    tiers: List[str] = None,
    depths: List[int] = None,
    n_vars_range: Tuple[int, int] = (2, 4)
) -> List[GeneratedFunction]:
    """
    Generate a dataset of synthetic functions.
    
    Args:
        n_samples: Total number of samples to generate
        tiers: List of tiers to include (default: all)
        depths: List of depths to use (default: [1, 2, 3])
        n_vars_range: (min, max) number of variables
    
    Returns:
        List of GeneratedFunction objects
    """
    if tiers is None:
        tiers = list(TIER_GENERATORS.keys())
    if depths is None:
        depths = [1, 2, 3]
    
    samples_per_tier = n_samples // len(tiers)
    dataset = []
    
    for tier in tiers:
        generator = TIER_GENERATORS[tier]
        
        for _ in range(samples_per_tier):
            depth = random.choice(depths)
            n_vars = random.randint(*n_vars_range)
            
            try:
                func = generator(n_vars=n_vars, depth=depth)
                
                # Validate the generated code parses
                ast.parse(func.code)
                dataset.append(func)
            except Exception as e:
                # If generation fails, try again with simpler params
                try:
                    func = generator(n_vars=2, depth=1)
                    ast.parse(func.code)
                    dataset.append(func)
                except:
                    pass  # Skip this sample
    
    random.shuffle(dataset)
    return dataset


def validate_dataset(dataset: List[GeneratedFunction]) -> Dict:
    """Validate dataset and return statistics."""
    stats = {
        'total': len(dataset),
        'by_tier': {},
        'by_category': {},
        'by_depth': {},
        'parse_errors': 0
    }
    
    for func in dataset:
        # Count by tier
        stats['by_tier'][func.tier] = stats['by_tier'].get(func.tier, 0) + 1
        
        # Count by category
        stats['by_category'][func.category] = stats['by_category'].get(func.category, 0) + 1
        
        # Count by depth
        stats['by_depth'][func.depth] = stats['by_depth'].get(func.depth, 0) + 1
        
        # Validate parsing
        try:
            ast.parse(func.code)
        except:
            stats['parse_errors'] += 1
    
    return stats


# =============================================================================
# DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMPLEXITY TIER DATA GENERATOR")
    print("=" * 70)
    
    # Generate examples from each tier
    for tier in TIER_GENERATORS:
        print(f"\n{'=' * 70}")
        print(f"TIER: {tier.upper()}")
        print("=" * 70)
        
        generator = TIER_GENERATORS[tier]
        
        for i in range(3):
            func = generator(n_vars=2, depth=2)
            print(f"\nExample {i+1} (category: {func.category}):")
            print(func.code)
    
    # Generate a test dataset
    print("\n" + "=" * 70)
    print("GENERATING TEST DATASET")
    print("=" * 70)
    
    dataset = generate_dataset(
        n_samples=100,
        tiers=['linear', 'conditional', 'loop', 'recursion'],
        depths=[1, 2, 3]
    )
    
    stats = validate_dataset(dataset)
    
    print(f"\nTotal samples: {stats['total']}")
    print(f"Parse errors: {stats['parse_errors']}")
    print(f"\nBy tier:")
    for tier, count in sorted(stats['by_tier'].items()):
        print(f"  {tier}: {count}")
    print(f"\nBy category:")
    for cat, count in sorted(stats['by_category'].items()):
        print(f"  {cat}: {count}")
