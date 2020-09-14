"""Tools (and script) to convert Relay to Glenside"""
import sys
import tvm
from tvm import relay
import json


def glenside_from_ir_module(module):
    """Convert TVM IRModule to Glenside text format

    Parameters
    ----------
    module : tvm.IRModule
        The IRModule to convert.

    Returns
    -------
    (glenside_str, shapes) : (String, Map[String, List[int]])
        glenside_str is the glenside text-format implementing the module.
        shapes is a map from tensor variable name to tensor shape."""

    shapes = [(var.name_hint,
               tuple(int(val) for val in var.checked_type.shape))
              for var in module['main'].params]

    return (_recursive_helper(module['main'].body), shapes)


def _recursive_helper(expr):
    """

    Parameters
    ----------
    expr : tvm.ir.RelayExpr
        The expression to generate text for.

    Returns
    -------
    glenside_str : String
        The glenside text-format implementing the expr."""
    assert issubclass(type(expr), tvm.ir.RelayExpr)

    if isinstance(expr, tvm.relay.Var):
        assert expr.name_hint
        return "(access-tensor {})".format(expr.name_hint)
    if isinstance(expr, tvm.relay.Call):
        if expr.op == tvm.ir.Op.get("nn.softmax"):
            assert len(expr.args) == 1
            assert 'axis' in expr.attrs.keys()
            # Axis argument other than -1 not implemented.
            assert expr.attrs.axis == -1
            return "(compute softmax {})".format(
                _access(_recursive_helper(expr.args[0]),
                        len(expr.args[0].checked_type.shape) - 1))
        if expr.op == tvm.ir.Op.get("nn.bias_add"):
            assert len(expr.args) == 2

            # We need to broadcast the bias up to the correct size.
            # In Relay, this is done automatically.
            # In Glenside, this is done explicitly, in two steps:
            # 1. Insert the missing axes into the bias tensor, so that the bias
            #    tensor has the same number of dimensions as the data tensor
            # 2. Broadcast the bias tensor so that it has the same shape as the
            #    data tensor.
            assert 'axis' in expr.attrs.keys()
            assert len(expr.args[1].checked_type.shape) == 1, \
                "Only supporting vector biases at the moment"

            data = _recursive_helper(expr.args[0])
            bias = _recursive_helper(expr.args[1])

            # Insert axes before
            for _ in range(expr.attrs.axis):
                bias = "(access-insert-axis {} 0)".format(bias)

            # Insert axes after
            for x in range(expr.attrs.axis + 1,
                           len(expr.args[0].checked_type.shape)):
                bias = "(access-insert-axis {} {})".format(bias, x)

            # Broadcast
            bias = "(access-broadcast {} (get-access-shape {}))".format(
                bias, data)

            return _elementwise_add(data, bias)

    # If we make it here, we haven't yet implemented parsing for the expression.
    sys.stderr.write("Cannot parse expression of type {}\n".format(type(expr)))
    if isinstance(expr, tvm.relay.Call):
        sys.stderr.write("Call to operator: {}\n".format(expr.op))
    exit(1)


def _elementwise_add(a, b):
    """Generate elementwise addition

    Parameters
    ----------
    a, b : String
        The inputs to add."""
    return "(compute elementwise-add (access-pair {} {}))" \
        .format(_access(a, 0), _access(b, 0))


def _access(expr, axis):
    """Access expression at an axis

    Parameters
    ----------
    expr : String
        The inputs to add.

    axis : int
        The axis to access at.

    Returns
    -------
    out_code : String
        (access <code generated for expr> <axis)
    """
    return "(access {} {})".format(expr, axis)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('infile',
                        nargs='?',
                        type=argparse.FileType('r'),
                        default=sys.stdin)
    parser.add_argument('outfile',
                        nargs='?',
                        type=argparse.FileType('w'),
                        default=sys.stdout)
    parsed = parser.parse_args(sys.argv[1:])

    relay_in = parsed.infile.read()
    out, shapes = glenside_from_ir_module(tvm.parser.fromtext(relay_in))
    parsed.outfile.write(json.dumps({'program': out, 'shapes': shapes}))
