#!/usr/bin/env python3
"""Test os.getcwd() call."""

import os
import sys

print("1. About to call os.getcwd()...", flush=True)
try:
    cwd = os.getcwd()
    print(f"2. Got cwd: {cwd}", flush=True)
except Exception as e:
    print(f"2. FAILED: {e}", flush=True)
    
print("3. Done!", flush=True)