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
                    type=argparse.FileType('rb'),
                    default=sys.stdin)
parser.add_argument('--npy_out_filepath', 
                    nargs='*')
parser.add_argument('--npy_arg_filepath',
                    nargs='*')

parsed = parser.parse_args()

if parsed.dense:
    data = relay.var('data', shape=(16, 32), dtype='float32')
    weights = relay.var('weights', shape=(64, 32), dtype='float32')
    expr = tvm.IRModule.from_expr(
        relay.Function([data, weights], relay.nn.dense(data, weights)))
else:
    relay_in = parsed.relay_filepath.read()
    expr = tvm.parser.fromtext(relay_in)

inputs = []
for filepath in parsed.npy_arg_filepath:
    with open(filepath, 'rb') as arg_file:
        inputs.append(np.load(arg_file))

# need graph runtime or crashes for yolo
output = relay.create_executor(mod=expr, kind="graph").evaluate()(*inputs)

for i in range(len(parsed.npy_out_filepath)):
    filepath = parsed.npy_out_filepath[i]
    with open(filepath, "wb"):
        np.save(filepath, output[i].asnumpy().astype('float32'))
