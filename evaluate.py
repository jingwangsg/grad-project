import argparse
"""
- restore model
- estimate on test dataset (Not high snr)
- output the correctness changing along with snr
"""

parser = argparse.ArgumentParser()
parser.add_argument("--restore_file")

snr = 