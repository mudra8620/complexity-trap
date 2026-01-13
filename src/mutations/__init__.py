"""
Code mutation utilities for robustness testing.

Provides semantics-preserving transformations:
- Variable renaming
- Function renaming (future)
- Code reformatting (future)
"""

from .variable_renamer import mutate_code
