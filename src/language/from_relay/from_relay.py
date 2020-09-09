"""Tools (and script) to convert Relay to Glenside"""
import tvm
from tvm import relay


def glenside_from_ir_module(module):
    """Convert TVM IRModule to Glenside text format

    Parameters
    ----------
    module : tvm.IRModule
        The IRModule to convert.

    Returns
    -------
    glenside_str : String
        The glenside text-format implementing the module."""
    return "unimplemented"


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
    assert issubclass(expr, tvm.ir.RelayExpr)


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
    out = glenside_from_ir_module(tvm.parser.fromtext(relay_in))
    parsed.outfile.write(out)
