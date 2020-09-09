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

    shapes = {
        var.name_hint: tuple(int(val) for val in var.checked_type.shape)
        for var in module['main'].params
    }

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

    if False:
        pass
    else:
        sys.stderr.write("Cannot parse expression of type {}\n".format(
            type(expr)))
        if isinstance(expr, tvm.relay.Call):
            sys.stderr.write("Call to operator: {}\n".format(expr.op))
        exit(1)


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
