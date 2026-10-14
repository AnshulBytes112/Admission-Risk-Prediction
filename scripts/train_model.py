#!/usr/bin/env python3
"""Train/evaluate models on the synthetic processed feature dataset.

Maintained for backward compatibility; forwards to scripts.train_risk_model.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train_risk_model import main

if __name__ == "__main__":
    sys.exit(main())
