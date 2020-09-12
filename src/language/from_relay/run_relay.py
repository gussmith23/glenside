"""Run Relay code given inputs as filepath; write output to file"""
import sys
import argparse
import tvm
from tvm import relay
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('--relay_filepath',
                    nargs=1,
                    type=argparse.FileType('rb'),
                    default=sys.stdin)
parser.add_argument('npy_out_filepath',
                    type=argparse.FileType('wb'))
parser.add_argument('npy_arg_filepath',
                    nargs='*',
                    type=argparse.FileType('rb'))
parsed = parser.parse_args(sys.argv[1:])

relay_in = parsed.relay_filepath.read()
expr = tvm.parser.fromtext(relay_in)

inputs = [np.load(arg_file) for arg_file in parsed.npy_arg_filepath]

output = relay.create_executor(mod=expr).evaluate()(*inputs).asnumpy()

np.save(parsed.npy_out_filepath, output)
