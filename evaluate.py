#!/usr/bin/env python3
"""Convenience root entrypoint delegating to scripts/evaluate.py."""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from scripts.evaluate import main

if __name__ == "__main__":
    main()
