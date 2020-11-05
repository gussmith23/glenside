"""Run Relay code given inputs as filepath; write output to file"""
import sys
import argparse
import tvm
from tvm import relay
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '--dense',
    action='store_true',
    help='Temporary band-aid b/c nn.dense Relay parser is broken')
parser.add_argument('--relay_filepath',
                    nargs=1,
                    type=argparse.FileType('rb'),
                    default=sys.stdin)
parser.add_argument('npy_out_filepath', type=argparse.FileType('wb'))
parser.add_argument('npy_arg_filepath',
                    nargs='*',
                    type=argparse.FileType('rb'))

parsed = parser.parse_args()

if parsed.dense:
    data = relay.var('data', shape=(16, 32), dtype='float32')
    weights = relay.var('weights', shape=(64, 32), dtype='float32')
    expr = tvm.IRModule.from_expr(
        relay.Function([data, weights], relay.nn.dense(data, weights)))
else:
    relay_in = parsed.relay_filepath.read()
    expr = tvm.parser.fromtext(relay_in)

inputs = [np.load(arg_file) for arg_file in parsed.npy_arg_filepath]
for i in inputs:
    print(i.shape)
output = relay.create_executor(mod=expr).evaluate()(*inputs).asnumpy()

np.save(parsed.npy_out_filepath, output)
