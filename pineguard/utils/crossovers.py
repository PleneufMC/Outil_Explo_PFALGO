"""
Crossover/Crossunder helpers — Exact Pine equivalence.

Pine:
  ta.crossover(a, b)  = a > b AND a[1] <= b[1]    // Note: <= not <
  ta.crossunder(a, b) = a < b AND a[1] >= b[1]     // Note: >= not >

PIÈGE: The [1] comparison uses <= and >=, not strict < and >.
Missing this causes different crossover counts.
"""

import pandas as pd

Series = pd.Series


def crossover(a: Series, b: Series) -> Series:
    """
    Pine: ta.crossover(a, b) — strict crossover.

    True when:
      a[i] > b[i] AND a[i-1] <= b[i-1]

    Note the <= (not <) for the previous bar.
    """
    return (a > b) & (a.shift(1) <= b.shift(1))


def crossunder(a: Series, b: Series) -> Series:
    """
    Pine: ta.crossunder(a, b) — strict crossunder.

    True when:
      a[i] < b[i] AND a[i-1] >= b[i-1]

    Note the >= (not >) for the previous bar.
    """
    return (a < b) & (a.shift(1) >= b.shift(1))
