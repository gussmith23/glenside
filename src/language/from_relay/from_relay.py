"""Tools (and script) to convert Relay to Glenside"""
import sys
import tvm
from tvm import relay
import json


def _ndim(expr):
    '''Return number of dimensions of a Relay expression'''
    return len(expr.checked_type.shape)


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
    elif isinstance(expr, tvm.relay.Call):
        if expr.op == tvm.ir.Op.get("nn.softmax"):
            assert len(expr.args) == 1
            assert 'axis' in expr.attrs.keys()
            # Axis argument other than -1 not implemented.
            assert expr.attrs.axis == -1
            return "(compute softmax {})".format(
                _access(_recursive_helper(expr.args[0]),
                        len(expr.args[0].checked_type.shape) - 1))
        elif expr.op == tvm.ir.Op.get("nn.bias_add"):
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

        elif expr.op == tvm.ir.Op.get("nn.dense"):
            assert len(expr.args) == 2
            assert expr.attrs.out_dtype == '', \
                'out_dtype not yet supported'
            assert _ndim(expr.args[0]) == 2, \
                'First arg to dense should have 2 dimensions for now'
            assert _ndim(expr.args[1]) == 2, \
                'Second arg to dense should have 2 dimensions for now'

            return '(compute dot-product (access-cartesian-product {} {}))' \
                .format(_access(_recursive_helper(expr.args[0]), 1),
                        _access(_recursive_helper(expr.args[1]), 1))

        elif expr.op == tvm.ir.Op.get("nn.batch_flatten"):
            assert len(expr.args) == 1
            assert _ndim(expr.args[0]) >= 1
            return '(access-flatten {})' \
                .format(_access(_recursive_helper(expr.args[0]), 1))

        elif expr.op == tvm.ir.Op.get("nn.global_avg_pool2d"):
            assert len(expr.args) == 1
            assert _ndim(expr.args[0]) == 4
            assert expr.attrs.layout == 'NCHW', \
                'Other layouts not yet supported'
            if expr.attrs.layout == 'NCHW':
                # Compute the mean
                out = '(compute reduce-mean {})' \
                    .format(_access(_recursive_helper(expr.args[0]), 2))
                # Insert the last two dimensions back
                out = '(access-insert-axis (access-insert-axis {} 2) 3)' \
                    .format(out)
                # Re-access at correct location
                out = _access(out, 2)
                return out
            else:
                assert False, 'unreachable'

        elif expr.op == tvm.ir.Op.get("nn.relu"):
            assert len(expr.args) == 1
            return '(compute relu {})' \
                .format(_recursive_helper(expr.args[0]))

        elif expr.op == tvm.ir.Op.get('add') \
           or expr.op == tvm.ir.Op.get('multiply') \
           or expr.op == tvm.ir.Op.get('divide'):
            assert len(expr.args) == 2
            a = _recursive_helper(expr.args[0])
            a_shape = [int(v) for v in expr.args[0].checked_type.shape]
            b = _recursive_helper(expr.args[1])
            b_shape = [int(v) for v in expr.args[1].checked_type.shape]

            # Make the number of dimensions match
            while len(a_shape) < len(b_shape):
                a = '(access-insert-axis {} 0)'.format(a)
                a_shape = [1] + a_shape
            while len(b_shape) < len(a_shape):
                b = '(access-insert-axis {} 0)'.format(b)
                b_shape = [1] + b_shape

            # Cannot handle complex broadcasts.
            # That is, cannot handle broadcasts where dim an in shape a must be
            # broadcast up to dimension bn, while simultaneously dimension bm
            # in shape b must be broadcast up to dimension am.
            # This is a limitation of access-broadcast which will take some
            # rethinking to fix.
            assert all(map(lambda t: t[0] <= t[1], zip(a_shape, b_shape))) \
                or \
                all(map(lambda t: t[0] >= t[1], zip(a_shape, b_shape)))

            if any(map(lambda t: t[0] < t[1], zip(a_shape, b_shape))):
                a = "(access-broadcast {} (get-access-shape {}))" \
                    .format(a, b)
            elif any(map(lambda t: t[0] > t[1], zip(a_shape, b_shape))):
                b = "(access-broadcast {} (get-access-shape {}))" \
                    .format(b, a)

            if expr.op == tvm.ir.Op.get('add'):
                return _elementwise_add(a, b)
            elif expr.op == tvm.ir.Op.get('multiply'):
                return _elementwise_mul(a, b)
            elif expr.op == tvm.ir.Op.get('divide'):
                return _elementwise_div(a, b)
            else:
                assert False, 'unreachable'

        elif expr.op == tvm.ir.Op.get('nn.conv2d'):
            assert len(expr.args) == 2
            assert _ndim(expr.args[0]) == 4
            assert _ndim(expr.args[1]) == 4
            assert len(expr.attrs.padding) == 4
            assert [int(v) for v in expr.attrs.dilation] == [1, 1]
            assert expr.attrs.groups == 1
            assert expr.attrs.data_layout == 'NCHW'
            assert expr.attrs.kernel_layout == 'OIHW'
            assert expr.attrs.out_layout == ''
            assert expr.attrs.out_dtype == ''

            data = _recursive_helper(expr.args[0])
            weights = _recursive_helper(expr.args[1])

            stride = [int(v) for v in expr.attrs.strides]
            pad = [int(v) for v in expr.attrs.padding]
            data_layout = expr.attrs.data_layout
            kernel_layout = expr.attrs.kernel_layout

            # TODO(@gussmith23) Layout assumption
            assert kernel_layout == 'OIHW'
            # it's not actually the "weight shape" exactly. it's the shape of
            # ONE weight, with batch dim = 1.
            weights_shape = "(shape 1 {})" \
                .format(" ".join(str(v) for v in expr.args[1].checked_type.shape[1:]))

            # TODO(@gussmith23) Layout assumption
            assert data_layout == 'NCHW'
            data = "(access-pad {} zero-padding 2 {} {})" \
                .format(data, pad[0], pad[2])

            # TODO(@gussmith23) Layout assumption
            assert data_layout == 'NCHW'
            data = "(access-pad {} zero-padding 3 {} {})" \
                .format(data, pad[1], pad[3])

            # Access windows expects access to access last dimension
            data = _access(data, 4)

            # TODO(@gussmith23) Layout assumption
            assert data_layout == 'NCHW'
            assert kernel_layout == 'OIHW'
            stride_list = "(shape 1 1 {} {})" \
                .format(stride[0], stride[1])

            # TODO(@gussmith23) Layout assumption
            assert data_layout == 'NCHW'
            assert kernel_layout == 'OIHW'
            data = "(access-windows {} {} {})" \
                .format(data, weights_shape, stride_list)

            # TODO(@gussmith23) Layout assumption
            # Squeeze input channels (axis 1) and kernel batch (axis 4)
            assert data_layout == 'NCHW'
            data = "(access-squeeze (access-squeeze {} 4) 1)" \
                .format(data)

            data = _access(data, 3)

            weights = _access(weights, 1)

            data = "(compute dot-product (access-cartesian-product {} {}))" \
                .format(weights, data)

            # TODO(@gussmith23) Layout assumption
            assert data_layout == 'NCHW'
            # Transpose to NCHW
            data = "(access-transpose {} (list 1 0 2 3))".format(data)

            return data

        elif expr.op == tvm.ir.Op.get('nn.max_pool2d'):
            assert len(expr.args) == 1
            assert _ndim(expr.args[0]) == 4
            assert len(expr.attrs.pool_size) == 2
            assert len(expr.attrs.padding) == 4
            assert len(expr.attrs.strides) == 2
            assert expr.attrs.layout == 'NCHW'
            assert expr.attrs.ceil_mode == False

            data_layout = expr.attrs.layout
            data = _recursive_helper(expr.args[0])
            stride = [int(v) for v in expr.attrs.strides]
            pad = [int(v) for v in expr.attrs.padding]
            pool_size = [int(v) for v in expr.attrs.pool_size]

            # TODO(@gussmith23) Layout assumption
            assert data_layout == 'NCHW'
            data = "(access-pad {} min-padding 2 {} {})" \
                .format(data, pad[0], pad[2])

            # TODO(@gussmith23) Layout assumption
            assert data_layout == 'NCHW'
            data = "(access-pad {} min-padding 3 {} {})" \
                .format(data, pad[1], pad[3])

            # Access windows expects access to access last dimension
            data = _access(data, 4)

            # TODO(@gussmith23) Layout assumption
            assert data_layout == 'NCHW'
            stride_list = "(shape 1 1 {} {})" \
                .format(stride[0], stride[1])

            # TODO(@gussmith23) Layout assumption
            assert data_layout == 'NCHW'
            pool_window_shape = '(shape 1 1 {} {})' \
                .format(pool_size[0], pool_size[1])

            # TODO(@gussmith23) Layout assumption
            assert data_layout == 'NCHW'
            data = "(access-windows {} {} {})" \
                .format(data, pool_window_shape, stride_list)

            # Compute over window dimensions
            data = _access(data, 4)

            data = '(compute reduce-max {})'.format(data)

            return data

        elif expr.op == tvm.ir.Op.get('expand_dims'):
            assert len(expr.args) == 1

            data = _recursive_helper(expr.args[0])

            for _ in range(int(expr.attrs.num_newaxis)):
                data = '(access-insert-axis {} {})' \
                    .format(data, int(expr.attrs.axis))

            return data

    # If we make it here, we haven't yet implemented parsing for the expression.
    sys.stderr.write("Cannot parse expression of type {}\n".format(type(expr)))
    if isinstance(expr, tvm.relay.Call):
        sys.stderr.write("Call to operator: {}\n".format(expr.op))
    exit(1)


def _elementwise_div(a, b):
    """Generate elementwise division

    Parameters
    ----------
    a, b : String
        The inputs to divide."""
    return "(compute elementwise-div (access-pair {} {}))" \
        .format(_access(a, 0), _access(b, 0))


def _elementwise_mul(a, b):
    """Generate elementwise multiplication

    Parameters
    ----------
    a, b : String
        The inputs to multiply."""
    return "(compute elementwise-mul (access-pair {} {}))" \
        .format(_access(a, 0), _access(b, 0))


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
    parser.add_argument(
        '--dense',
        action='store_true',
        help='Temporary band-aid b/c nn.dense Relay parser is broken')
    parsed = parser.parse_args(sys.argv[1:])

    if parsed.dense:
        data = relay.var('data', shape=(16, 32), dtype='float32')
        weights = relay.var('weights', shape=(64, 32), dtype='float32')
        module = tvm.IRModule.from_expr(
            relay.Function([data, weights], relay.nn.dense(data, weights)))
        out, shapes = glenside_from_ir_module(module)
    else:
        relay_in = parsed.infile.read()
        out, shapes = glenside_from_ir_module(tvm.parser.fromtext(relay_in))

    parsed.outfile.write(json.dumps({'program': out, 'shapes': shapes}))
