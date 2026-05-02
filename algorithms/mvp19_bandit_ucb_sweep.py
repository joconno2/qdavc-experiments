"""
MVP 19: UCB Exploration Constant Sweep

Same as MVP 7 (Bandit Experts) but with configurable ucb_c.
This file provides the class; the run script instantiates with
different c values (0.25, 0.5, 2.0, 4.0) to test whether
exploration pressure matters.
"""

# This is just MVP 7 re-exported with a different name for clarity.
# The actual sweep happens in the run script via ALGORITHM_KWARGS.
from mvp7_bandit_experts import BanditExpertElites

ALGORITHM_CLASS = BanditExpertElites
ALGORITHM_NAME = "Bandit-c0.5"  # overridden by run script
ALGORITHM_KWARGS = {"ucb_c": 0.5}
