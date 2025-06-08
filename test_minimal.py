#!/usr/bin/env python3
"""Minimal test to identify hanging issue."""

import sys
import os

print("1. Basic print works", flush=True)

# Test logging
print("2. Testing logging setup...", flush=True)
import logging

print("3. Creating log file handler...", flush=True)
try:
    handler = logging.FileHandler("test.log")
    print("4. Handler created successfully", flush=True)
except Exception as e:
    print(f"4. FAILED: {e}", flush=True)

print("5. Creating stream handler...", flush=True)
try:
    stream = logging.StreamHandler()
    print("6. Stream handler created", flush=True)
except Exception as e:
    print(f"6. FAILED: {e}", flush=True)

print("7. Setting up basicConfig...", flush=True)
try:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[handler, stream]
    )
    print("8. BasicConfig done", flush=True)
except Exception as e:
    print(f"8. FAILED: {e}", flush=True)

print("9. Testing log output...", flush=True)
logging.info("Test message")
print("10. Done!", flush=True)