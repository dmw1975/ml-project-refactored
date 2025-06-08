#!/usr/bin/env python3
"""Test logging of parsed arguments."""

import logging
import argparse
import sys

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("test_args.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

print("1. Creating parser...", flush=True)
parser = argparse.ArgumentParser()
parser.add_argument('--visualize', action='store_true')
parser.add_argument('--non-interactive', action='store_true')

print("2. Parsing args...", flush=True)
args = parser.parse_args(['--visualize', '--non-interactive'])

print("3. About to log parsed arguments...", flush=True)
logging.info("Parsed arguments:")

print("4. Starting to iterate over args...", flush=True)
for arg, value in vars(args).items():
    print(f"   Processing arg: {arg} = {value}", flush=True)
    logging.info(f"  {arg}: {value}")
    
print("5. Done logging args!", flush=True)