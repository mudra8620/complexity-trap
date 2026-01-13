"""
Data generation utilities for the Complexity Trap experiments.

Provides synthetic code generators for different complexity tiers:
- Linear: Simple sequential operations
- Conditional: If/else branching logic
- Loop: Iterative constructs (for, while)
- Recursion: Self-referential function calls
"""

from .complexity_generators import generate_tier
