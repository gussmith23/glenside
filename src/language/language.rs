use crate::language::RelayOperator::*;
use egg::{define_language, DidMerge, EGraph, Id, Language as LanguageTrait};
use itertools::{any, multizip};
use log::debug;
use ndarray::Ix0;
use ndarray::{s, Dimension, Ix, IxDyn};
use ordered_float::NotNan;

use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::convert::TryInto;
use std::fmt::Display;
use std::iter::FromIterator;
use std::str::FromStr;

define_language! {
    pub enum Language {
        // (systolic-array <rows (usize)> <cols (usize)> <access-0> <access-1>)
        // Represents a systolic array of size rows X cols, fed with two
        // accesses.
        // This is Scott's weight-stationary systolic array design. It reads in
        // two matrices: first matrix is in the layout [M, N] and second is in
        // the layout [N, O]. The systolic array computes the matrix
        // multiplication, leading to a matrix with layout [M, O].
        // The systolic array expects exactly one shape for the second argument:
        // [N, O]. These correspond directly to the rows/cols parameters of the
        // systolic array. The first argument is partially constrained: its
        // second dimension must be N, but its first dimension may be any
        // length.
        // In terms of Glenside accesses, we expect <access-0> to have shape [M]
        // [N], and <access-1> to have shape [] [N, O].
        // TODO(@gussmith23) How to make the M argument "programmable"?
        // TODO(@gussmith23) do we need to specify rows and cols? You can infer these
        // from the size of the input, but it's also useful for searching.
        "systolic-array" = SystolicArray([Id; 4]),

        // Same as the systolic array above, but relies on Scott's blocking code
        // instead of relying on Glenside to discover the blocking. By
        // "blocking", we mean splitting up a matrix multiply to run on a
        // smaller systolic array.
        "systolic-array-with-blocking" = SystolicArrayWithBlocking([Id; 4]),

        // (systolic-array-conv2d-nchw-oihw-with-blocking
        //  <rows: Num> <cols: Num>
        //  <weights: Access> <data: Access>
        //  <kh: Num> <kw: Num>
        //  <stride-h: Num> <stride-w: Num>)
        // A systolic array operating in conv2d mode, with data in layout NCHW
        // and weights in layout OIHW. We don't actually have an atom for this,
        // but it's currently used as an intermediate to help discover
        // systolic-array-conv2d-nhwc-hwio-with-blocking.
        "systolic-array-conv2d-nchw-oihw-with-blocking" = SystolicArrayConv2dNchwOihwWithBlocking([Id; 8]),

        // (systolic-array-conv2d-nhwc-hwio-with-blocking
        //  <rows: Num> <cols: Num>
        //  <weights: Access> <data: Access>
        //  <kh: Num> <kw: Num>
        //  <stride-h: Num> <stride-w: Num>)
        // A systolic array operating in conv2d mode, with data in layout NHWC
        // and weights in layout HWIO. Scott should have an atom for this at
        // some point.
        "systolic-array-conv2d-nhwc-hwio-with-blocking" = SystolicArrayConv2dNhwcHwioWithBlocking([Id; 8]),

        // (systolic-array-conv2d-im2col-nchw-oihw-with-blocking <rows: Num>
        // <cols: Num> <weights: Access> <data: Access> <kh: Num> <kw:
        // Num> <stride-h: Num> <stride-w: Num>)
        // A systolic array operating in conv2d-im2col mode, with data in layout
        // NCHW and weights in layout OIHW. We don't actually have an atom for
        // this, but it's currently used as an intermediate to help discover
        // systolic-array-conv2d-im2col-nhwc-hwio-with-blocking. Conv2d im2col
        // mode indicates that the systolic array is reading in the image data
        // and transforming it to im2col on the fly.
        "systolic-array-conv2d-im2col-nchw-oihw-with-blocking" = SystolicArrayConv2dIm2colNchwOihwWithBlocking([Id; 8]),
        "systolic-array-conv2d-im2col-nhwc-hwio-with-blocking" = SystolicArrayConv2dIm2colNhwcHwioWithBlocking([Id; 8]),

        // (access-windows <access> <filters-shape: Shape> <stride-shape: Shape>)
        // Form the windows which will be convolved over.
        // TODO(@gussmith23) AccessWindows shouldn't be specific to filters.
        // AccessWindows is used in other contexts too, i.e. pooling.
        "access-windows" = AccessWindows([Id; 3]),

        // (shape-of <tensor>)
        // Returns the shape of the tensor.
        // TODO(@gussmith) Choose between ([Id; 1]) and (Id) and be consistent
        // When describing the arguments of a construct that takes a single Id
        // argument (like shape-of), we can use (Id) or ([Id; 1]). I'm not sure
        // which is better, but I should choose one and be consistent.
        "shape-of" = ShapeOf([Id; 1]),

        // (slice-shape <shape> <dim>)
        // Slices a shape by taking dimensions >= <dim>.
        "slice-shape" = SliceShape([Id; 2]),

        // (shape-insert-axis <shape: Shape> <axis: usize>)
        // Inserts an axis with value 1.
        "shape-insert-axis" = ShapeInsertAxis([Id; 2]),

        // (shape-remove-axis <shape: Shape> <axis: usize>)
        // Removes axis from shape.
        "shape-remove-axis" = ShapeRemoveAxis([Id; 2]),

        // (access <tensor> <dim>)
        // The most basic access pattern.
        // Let <tensor> have dims d0, .., dn.
        // Interprets <tensor> as a shaped list of shape d0, .., d(<dim>-1)
        // whose elements are of shape d<dim>, .., dn.
        "access" = Access([Id; 2]),

        // (access-transpose <a: access> <new-order: list>)
        // Uses numpy.transpose() semantics. Reorders axes in an access.
        // Does not change the access dimension.
        "access-transpose" = AccessTranspose([Id; 2]),

        // (access-cartesian-product <access1> <access2>)
        // Cartesian product access pattern.
        // Assume <access1> has shape
        // [a1, ..., an]
        // and <access2> has shape
        // [b1, ..., bm].
        // Both must have the same item shape,
        // [c1, ..., co]
        // Outputs a tensor of shape
        // [a1, ..., an, b1, ..., bm, 2, c1, ..., co]
        // which represents the cartesian product of the items in both accesses.
        "access-cartesian-product" = AccessCartesianProduct([Id; 2]),

        // (compute <compute-type> <access>)
        // Compute over the items in <access>.
        //
        // Compute types:
        //
        // dot-product
        // Expects an item shape of
        // [n, a0, ..., am]
        // Where n specifies the tuple multiplicity and [a0, ..., am] is the
        // shape of the tensors to be dot-producted with one another.
        "compute" = Compute([Id; 2]),

        // (get-access-shape <access>)
        // Returns the shape of the access.
        "get-access-shape" = GetAccessShape([Id;1]),
        // The below comment is wrong:
        //
        // This shouldn't actually be needed at the moment. We are handling all
        // statically-sized networks, and so anywhere where we would have used
        // this, we should be able to just plug in a literal access-shape. If
        // and when we start supporting dynamic networks, this will become
        // needed.
        //
        // Turns out we need this for flatten-unflatten. We can't just use
        // static shapes because the underlying shape might change as the shapes
        // "settle". It would be better to not have this "settling" at all, but
        // for now, we do.

        // (access-reshape <access> <shape>)
        // Reshapes the access to have the given
        "access-reshape" = AccessReshape([Id; 2]),

        // (access-flatten <access>)
        // Flattens the access's shape and item shape.
        "access-flatten" = AccessFlatten(Id),

        // (shape <usize>...)
        // Shape literal.
        "shape" = Shape(Box<[Id]>),

        // (list <usize>...)
        // List literal
        "list" = List(Box<[Id]>),

        // (construct-tuple <access> <access> ...)
        // Tuple Construction
        "construct-tuple" = ConstructTuple(Box<[Id]>),

        // (tuple-get-item <tuple> <i>)
        // Get the item at the ith index of tuple
        "tuple-get-item" = TupleGetItem([Id;2]),

        // (access-shape <shape: shape> <item-shape: shape>)
        // Access shape literal.
        "access-shape" = AccessShape([Id;2]),

        // (access-slice <access> <axis (usize)> <low (usize)> <high (usize)>)
        // Slices into <access> at axis <axis>, slicing the half-open range
        // [<low>, <high>).
        // TODO(@gussmith23) Implement access-slice-item
        // If axis >= access.shape.ndim(), it slices into access.item_shape.
        // This is me being lazy and not wanting to implement separate
        // access-slice-shape and access-slice-item operators for right now.
        "access-slice" = AccessSlice([Id; 4]),

        // (access-concatenate <a0> <a1> <axis (usize)>)
        // Concatenate accesses <a0> and <a1> along <axis>.
        // TODO(@gussmith23) Implement access-concatenate-item
        // If axis >= access.shape.ndim(), it concatenates along dimensions in
        // access.item_shape.
        "access-concatenate" = AccessConcatenate([Id; 3]),

        // (access-pair <a0> <a1>)
        // Simply pair every item of a0 with every item of a1.
        "access-pair" = AccessPair([Id; 2]),

        // (access-shift-right <a0>)
        // Shifts a dimension from shape to item shape.
        "access-shift-right" = AccessShiftRight(Id),

        // (access-tensor <t>)
        // Access a tensor literal.
        "access-tensor" = AccessTensor(Id),

        // (access-pad <a>
        //             <pad-type (PadType)>
        //             <axis (usize)> <pad-before (usize)> <pad-after (usize)>)
        // Pads a tensor at the given axis.
        "access-pad" = AccessPad([Id; 5]),

        // (access-squeeze <a> <axis (usize)>)
        "access-squeeze" = AccessSqueeze([Id; 2]),

        // (access-insert-axis <a> <axis (usize)>)
        "access-insert-axis" = AccessInsertAxis([Id; 2]),

        // (access-broadcast <a> <shape: shape>)
        // Simple broadcasting. <a> and <shape> must have the same total number
        // of dimensions. All dimensions in <a> must either match the
        // corresponding dimension in <shape> or be 1.
        "access-broadcast" = AccessBroadcast([Id; 2]),

        // (access-literal <literal: Literal>)
        // Access a literal. This may be able to be folded in to some other
        // access pattern, later on. It fits in with access-tensor as a "access
        // pattern constructor"; it takes something that isn't an access pattern
        // and converts it to an access pattern.
        "access-literal" = AccessLiteral(Id),

        // (literal <val: Float64>)
        // A literal value. Can only represent 0-dimensional values for now, but
        // in the future, we can and should support array constants.
        "literal" = Literal(Id),

        // (relay-operator-call <relay-operator: RelayOperator> <args>...)
        "relay-operator-call" = RelayOperatorCall(Box<[Id]>),

        "accelerator-call" = AcceleratorCall(Box<[Id]>),

        // (constant-tensor <value> <shape>)
        "constant-tensor" = ConstantTensor([Id; 2]),

        Num(i64),

        DataType(DataType),

        // Important that this go after usize, so that usizes are parsed as
        // usizes, not as floats.
        NotNanFloat64(NotNan<f64>),

        RelayOperator(RelayOperator),
        RelayActivationLayout(RelayActivationLayout),
        RelayKernelLayout(RelayKernelLayout),

        // pad-type: zero-padding
        // (No other options right now)
        PadType(PadType),

        ComputeType(ComputeType),

        AcceleratorFunc(AcceleratorFunc),

        Symbol(String),
    }
}

// TODO(@gussmith23) We need to just make a full-fledged Relay dialect.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelayOperator {
    /// (relay-operator relay-batch-norm-inference <data: access>
    ///  <gamma: access> <beta: access>
    ///  <moving_mean: access> <moving_var: access>
    ///  <axis: usize> <epsilon: float>)
    /// The batch-norm-at-inference-time operator. We don't currently support
    /// normal batch norm, which is a training-time operator, but we do support
    /// its inference-time simplified version.
    /// TODO(@gussmith23) How to handle batch norms?
    RelayBatchNormInference,

    /// (relay-operator relay-softmax <data: access> <axis: int>)
    RelaySoftmax,

    /// (relay-operator relay-relu <data: access>)
    RelayReLU,

    /// (relay-operator relay-leaky-relu <data: access> <alpha: Float64>)
    RelayLeakyReLU,

    /// (relay-operator relay-max-pool2d <data: access>
    ///  <pool size: shape> <strides: shape> <padding: shape>
    ///  <layout: RelayActivationLayout>)
    RelayMaxPool2D,

    /// (relay-operator relay-global-avg-pool2d <data: access>
    ///  <layout: RelayActivationLayout>)
    RelayGlobalAvgPool2D,

    /// (relay-operator relay-avg-pool2d <data: access>
    /// <pool_size: shape> <strides: shape> <padding: shape> <layout: RelayActivationLayout>
    /// )
    RelayAvgPool2D,

    /// (relay-operator relay-upsampling <data: access> <scale_h: Float64> <scale_w: Float64>
    /// <layout: RelayActivationLayout>)
    RelayUpSampling,

    /// (relay-operator relay-batch-flatten <data: access>)
    RelayBatchFlatten,

    /// (relay-operator relay-bias-add <data: access> <bias: access>
    ///  <axis: usize>)
    RelayBiasAdd,

    /// (relay-operator relay-dense <data: access> <weight: access>)
    RelayDense,

    /// (relay-operator relay-reshape <data: access> <shape: shape>)
    RelayReshape,

    /// (relay-operator-call relay-conv1d <data: access> <kernel: access>
    ///  <strides: shape> <padding: shape>)
    RelayConv1D,

    /// (relay-operator relay-erf <data: access>)
    RelayErf,

    /// (relay-operator relay-mean <data: access> <axis: usize>)
    RelayMean,

    /// (relay-operator relay-add <a: access> <b: access>)
    RelayAdd,

    /// (relay-operator relay-multiply <a: access> <b: access>)
    RelayMultiply,

    /// (relay-operator relay-divide <a: access> <b: access>)
    RelayDivide,

    /// (relay-operator relay-sigmoid <data: access>)
    RelaySigmoid,

    /// (relay-operator relay-maximum <a: access> <b: access>)
    RelayMaximum,

    /// (relay-operator relay-minimum <a:access> <b: access>)
    RelayMinimum,

    /// (relay-operator relay-conv2d <data: access> <kernel: access>
    ///  <strides: shape> <padding: shape> <group: usize>
    ///  <channels: usize> <kernel_size: shape> <activation_layout>
    ///  <kernel_layout>)
    RelayConv2D,

    /// (relay-operator relay-split <data: access> <indices_or_sections: usize> <axis: usize>)
    RelaySplit,

    /// (relay-operator relay-cast <data: access> <dtype: DataType>)
    RelayCast,

    /// (relay-operator relay-clip <data: access> <a_min: NonNanf64> <a_max: NonNanf64>)
    RelayClip,

    /// (relay-operator relay-left-shift <data: access> <nbits: i32>)
    RelayLeftShift,

    /// (relay-operator relay-right-shift <data: access> <nbits: i32>)
    RelayRightShift,

    /// (relay-operator round <data: access>)
    RelayRound,

    /// (relay-operator relay-take <data: access> <indices: access> <axis: usize>)
    RelayTake,

    /// (relay-operator relay-dropout <data: access> <rate: f64>)
    RelayDropout,

    /// (relay-operator relay-tanh <data: access>)
    RelayTanh,

    /// (relay-operator relay-stack <data: access> ... <axis: i32>)
    RelayStack,

    /// (relay-operator relay-log-softmax <data: access> <axis: usize>)
    RelayLogSoftmax,

    /// (relay-operator relay-strided-slice <data: access> <begin: shape> <end: shape> <strides: shape>)
    RelayStridedSlice,

    RelayLayerNorm,

    RelayBatchMatmul,

    RelayZeros,

    RelaySqrt,

    RelayNegative,

    /// (relay-operator-call relay-expand-dims <data: access> <axis: int> <num_newaxis: int>)
    RelayExpandDims,

    /// (relay-operator-call relay-pad <data: access>
    ///                                <pad_width: shape>)
    /// Note that the pad_width is a flattened version of the list of tuples
    /// which is used in Relay.
    /// Padding value is assumed to be zero.
    RelayPad,

    /// (relay-operator-call relay-concatenate <to-concat: Tuple of access> <axis: num>)
    RelayConcatenate,

    /// (relay-operator-call relay-transpose <data: access> <axes: list of num>)
    RelayTranspose,

    /// (relay-operator-call relay-squeeze <data: access> <axes: list of num>)
    RelaySqueeze,

    RelayConv3D,

    RelayConv3DTranspose,
}

/// All variants of [`RelayOperator`].
/// TODO(@gussmith23) We shouldn't always need this. Currently just used for
/// from_relay.
pub static RELAY_OPS: &[RelayOperator] = &[
    RelayBatchNormInference,
    RelaySoftmax,
    RelayReLU,
    RelayLeakyReLU,
    RelayMaxPool2D,
    RelayGlobalAvgPool2D,
    RelayAvgPool2D,
    RelayUpSampling,
    RelayBatchFlatten,
    RelayBiasAdd,
    RelayDense,
    RelayReshape,
    RelayConv1D,
    RelayConv3D,
    RelayErf,
    RelayMean,
    RelayAdd,
    RelayMultiply,
    RelayDivide,
    RelaySigmoid,
    RelayMaximum,
    RelayMinimum,
    RelayConv2D,
    RelaySplit,
    RelayCast,
    RelayClip,
    RelayLeftShift,
    RelayRightShift,
    RelayRound,
    RelayTake,
    RelayDropout,
    RelayTanh,
    RelayStack,
    RelayLogSoftmax,
    RelayStridedSlice,
    RelayLayerNorm,
    RelayBatchMatmul,
    RelayZeros,
    RelaySqrt,
    RelayNegative,
    RelayExpandDims,
    RelayPad,
    RelayConcatenate,
    RelayTranspose,
    RelaySqueeze,
    RelayConv3DTranspose,
];

impl FromStr for RelayOperator {
    type Err = ();
    fn from_str(input: &str) -> Result<RelayOperator, Self::Err> {
        match input {
            "relay-squeeze" => Ok(RelayOperator::RelaySqueeze),
            "relay-transpose" => Ok(RelayOperator::RelayTranspose),
            "relay-concatenate" => Ok(RelayOperator::RelayConcatenate),
            "relay-batch-norm-inference" => Ok(RelayOperator::RelayBatchNormInference),
            "relay-softmax" => Ok(RelayOperator::RelaySoftmax),
            "relay-relu" => Ok(RelayOperator::RelayReLU),
            "relay-max-pool2d" => Ok(RelayOperator::RelayMaxPool2D),
            "relay-global-avg-pool2d" => Ok(RelayOperator::RelayGlobalAvgPool2D),
            "relay-batch-flatten" => Ok(RelayOperator::RelayBatchFlatten),
            "relay-bias-add" => Ok(RelayOperator::RelayBiasAdd),
            "relay-add" => Ok(RelayOperator::RelayAdd),
            "relay-sigmoid" => Ok(RelayOperator::RelaySigmoid),
            "relay-avg-pool2d" => Ok(RelayOperator::RelayAvgPool2D),
            "relay-upsampling" => Ok(RelayOperator::RelayUpSampling),
            "relay-maximum" => Ok(RelayOperator::RelayMaximum),
            "relay-minimum" => Ok(RelayOperator::RelayMinimum),
            "relay-leaky-relu" => Ok(RelayOperator::RelayLeakyReLU),
            "relay-dense" => Ok(RelayOperator::RelayDense),
            "relay-reshape" => Ok(RelayOperator::RelayReshape),
            "relay-conv1d" => Ok(RelayOperator::RelayConv1D),
            "relay-erf" => Ok(RelayOperator::RelayErf),
            "relay-mean" => Ok(RelayOperator::RelayMean),
            "relay-multiply" => Ok(RelayOperator::RelayMultiply),
            "relay-conv2d" => Ok(RelayOperator::RelayConv2D),
            "relay-conv3d" => Ok(RelayOperator::RelayConv3D),
            "relay-split" => Ok(RelayOperator::RelaySplit),
            "relay-cast" => Ok(RelayOperator::RelayCast),
            "relay-clip" => Ok(RelayOperator::RelayClip),
            "relay-left-shift" => Ok(RelayOperator::RelayLeftShift),
            "relay-right-shift" => Ok(RelayOperator::RelayRightShift),
            "relay-round" => Ok(RelayOperator::RelayRound),
            "relay-take" => Ok(RelayOperator::RelayTake),
            "relay-dropout" => Ok(RelayOperator::RelayDropout),
            "relay-stack" => Ok(RelayOperator::RelayStack),
            "relay-log-softmax" => Ok(RelayOperator::RelayLogSoftmax),
            "relay-strided-slice" => Ok(RelayOperator::RelayStridedSlice),
            "relay-layer-norm" => Ok(RelayOperator::RelayLayerNorm),
            "relay-batch-matmul" => Ok(RelayOperator::RelayBatchMatmul),
            "relay-zeros" => Ok(RelayOperator::RelayZeros),
            "relay-negative" => Ok(RelayOperator::RelayNegative),
            "relay-sqrt" => Ok(RelayOperator::RelaySqrt),
            "relay-expand-dims" => Ok(RelayOperator::RelayExpandDims),
            "relay-pad" => Ok(RelayOperator::RelayPad),
            "relay-divide" => Ok(RelayOperator::RelayDivide),
            "relay-conv3d-transpose" => Ok(RelayOperator::RelayConv3DTranspose),
            _ => Err(()),
        }
    }
}
impl Display for RelayOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RelayOperator::RelaySqueeze => "relay-squeeze",
                RelayOperator::RelayTranspose => "relay-transpose",
                RelayOperator::RelayConcatenate => "relay-concatenate",
                RelayOperator::RelayStridedSlice => "relay-strided-slice",
                RelayOperator::RelayBatchNormInference => "relay-batch-norm-inference",
                RelayOperator::RelaySoftmax => "relay-softmax",
                RelayOperator::RelayReLU => "relay-relu",
                RelayOperator::RelayLeakyReLU => "relay-leaky-relu",
                RelayOperator::RelayMaxPool2D => "relay-max-pool2d",
                RelayOperator::RelayGlobalAvgPool2D => "relay-global-avg-pool2d",
                RelayOperator::RelayBatchFlatten => "relay-batch-flatten",
                RelayOperator::RelayBiasAdd => "relay-bias-add",
                RelayOperator::RelayAdd => "relay-add",
                RelayOperator::RelaySigmoid => "relay-sigmoid",
                RelayOperator::RelayAvgPool2D => "relay-avg-pool2d",
                RelayOperator::RelayUpSampling => "relay-upsampling",
                RelayOperator::RelayMaximum => "relay-maximum",
                RelayOperator::RelayMinimum => "relay-minimum",
                RelayOperator::RelayDense => "relay-dense",
                RelayOperator::RelayReshape => "relay-reshape",
                RelayOperator::RelayConv1D => "relay-conv1d",
                RelayOperator::RelayErf => "relay-erf",
                RelayOperator::RelayMean => "relay-mean",
                RelayOperator::RelayMultiply => "relay-mul",
                RelayOperator::RelayConv2D => "relay-conv2d",
                RelayOperator::RelaySplit => "relay-split",
                RelayOperator::RelayCast => "relay-cast",
                RelayOperator::RelayClip => "relay-clip",
                RelayOperator::RelayLeftShift => "relay-left-shift",
                RelayOperator::RelayRightShift => "relay-right-shift",
                RelayOperator::RelayRound => "relay-round",
                RelayOperator::RelayTake => "relay-take",
                RelayOperator::RelayDropout => "relay-dropout",
                RelayOperator::RelayTanh => "relay-tanh",
                RelayOperator::RelayStack => "relay-stack",
                RelayOperator::RelayLogSoftmax => "relay-log-softmax",
                RelayOperator::RelayLayerNorm => "relay-layer-norm",
                RelayOperator::RelayBatchMatmul => "relay-batch-matmul",
                RelayOperator::RelayZeros => "relay-zeros",
                RelayOperator::RelaySqrt => "relay-sqrt",
                RelayOperator::RelayNegative => "relay-negative",
                RelayOperator::RelayExpandDims => "relay-expand-dims",
                RelayOperator::RelayPad => "relay-pad",
                RelayOperator::RelayDivide => "relay-divide",
                RelayOperator::RelayConv3D => "relay-conv3d",
                RelayOperator::RelayConv3DTranspose => "relay-conv3d-transpose",
            }
        )
    }
}

/// Only for use in [`RelayOperatorCall`]s.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelayActivationLayout {
    NCHW,
    NHWC,
    NCDHW,
}
impl FromStr for RelayActivationLayout {
    type Err = ();
    fn from_str(input: &str) -> Result<RelayActivationLayout, Self::Err> {
        match input {
            "relay-activation-layout-nhwc" => Ok(RelayActivationLayout::NHWC),
            "relay-activation-layout-nchw" => Ok(RelayActivationLayout::NCHW),
            _ => Err(()),
        }
    }
}
impl Display for RelayActivationLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RelayActivationLayout::NHWC => "relay-activation-layout-nhwc",
                RelayActivationLayout::NCHW => "relay-activation-layout-nchw",
                RelayActivationLayout::NCDHW => "relay-activation-layout-ncdhw",
            }
        )
    }
}

/// Only for use in [`RelayOperatorCall`]s.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RelayKernelLayout {
    OIHW,
    HWIO,
    OIDHW,
}
impl FromStr for RelayKernelLayout {
    type Err = ();
    fn from_str(input: &str) -> Result<RelayKernelLayout, Self::Err> {
        match input {
            "relay-kernel-layout-hwio" => Ok(RelayKernelLayout::HWIO),
            "relay-kernel-layout-oihw" => Ok(RelayKernelLayout::OIHW),
            _ => Err(()),
        }
    }
}
impl Display for RelayKernelLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                RelayKernelLayout::HWIO => "relay-kernel-layout-hwio",
                RelayKernelLayout::OIHW => "relay-kernel-layout-oihw",
                RelayKernelLayout::OIDHW => "relay-kernel-layout-oidhw",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ComputeType {
    DotProduct,
    ReduceSum,
    ReLU,
    Sqrt,
    Negative,
    /// Expects item shape of `a x b1 x .. x bn`. Performs an elementwise
    /// addition of the `a` tensors of size `b1 x .. x bn`.
    /// TODO(@gussmith) Multiple-arg compute feels clunky and ad-hoc.
    /// Should figure out an explicit way to define access multiple-stream
    /// access patterns.
    ElementwiseAdd,
    /// Expects item shape of `a x b1 x .. x bn`. Performs an elementwise
    /// multiplication of the `a` tensors of size `b1 x .. x bn`.
    ElementwiseMul,
    ElementwiseDiv,
    /// Takes the max across all elements in each item. Reduces any item shape
    /// to a scalar.
    ReduceMax,
    /// Computes softmax. Currently expects access axis to be 0. Unsure how to
    /// define softmax for other access patterns.
    Softmax,
    /// For an item shape of `a1 x a2 x ...`, returns an item shape of `1` where
    /// the returned scalar is the mean of the `a1 x a2 x ...`-shaped tensor.
    ReduceMean,
}
impl FromStr for ComputeType {
    type Err = ();
    fn from_str(input: &str) -> Result<ComputeType, Self::Err> {
        match input {
            "dot-product" => Ok(ComputeType::DotProduct),
            "reduce-sum" => Ok(ComputeType::ReduceSum),
            "reduce-max" => Ok(ComputeType::ReduceMax),
            "relu" => Ok(ComputeType::ReLU),
            "sqrt" => Ok(ComputeType::Sqrt),
            "negative" => Ok(ComputeType::Negative),
            "elementwise-add" => Ok(ComputeType::ElementwiseAdd),
            "elementwise-mul" => Ok(ComputeType::ElementwiseMul),
            "elementwise-div" => Ok(ComputeType::ElementwiseDiv),
            "softmax" => Ok(ComputeType::Softmax),
            "reduce-mean" => Ok(ComputeType::ReduceMean),
            _ => Err(()),
        }
    }
}
impl Display for ComputeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                ComputeType::DotProduct => "dot-product",
                ComputeType::ReduceSum => "reduce-sum",
                ComputeType::ReduceMax => "reduce-max",
                ComputeType::ReLU => "relu",
                ComputeType::Sqrt => "sqrt",
                ComputeType::Negative => "negative",
                ComputeType::ElementwiseAdd => "elementwise-add",
                ComputeType::ElementwiseMul => "elementwise-mul",
                ComputeType::ElementwiseDiv => "elementwise-div",
                ComputeType::Softmax => "softmax",
                ComputeType::ReduceMean => "reduce-mean",
            }
        )
    }
}

/// Specifies how to pick the values we pad with.
#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Ord, Copy)]
pub enum PadType {
    /// Pad with zeroes.
    ZeroPadding,
    /// Pad with minimum representable number in the number system.
    MinPadding,
}
impl FromStr for PadType {
    type Err = ();
    fn from_str(input: &str) -> Result<PadType, Self::Err> {
        match input {
            "zero-padding" => Ok(PadType::ZeroPadding),
            "min-padding" => Ok(PadType::MinPadding),
            _ => Err(()),
        }
    }
}
impl Display for PadType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                PadType::ZeroPadding => "zero-padding",
                PadType::MinPadding => "min-padding",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AcceleratorFunc {
    FlexLinear,
    FlexLSTM,
    VTADense,
    VTAConv1D,
    HlsCNNConv2D,
    // (accelerator-call flex-maxpool <access>)
    //
    // Compute's FlexASR's maxpool operator. The input access should be of
    // shape ((),(t, h)) where t is the number of timesteps and h is the
    // number of hidden states. t should be divisible by 2; h should be
    // divisible by 16. The result is an access pattern with shape ((),(t/2,
    // h)).
    //
    // TODO(@gussmith23) Add tests for flexasr-maxpool.
    FlexASRMaxPool,
}

impl FromStr for AcceleratorFunc {
    type Err = ();

    fn from_str(s: &str) -> Result<AcceleratorFunc, Self::Err> {
        match s {
            "flex-linear" => Ok(AcceleratorFunc::FlexLinear),
            "flex-lstm" => Ok(AcceleratorFunc::FlexLSTM),
            "vta-dense" => Ok(AcceleratorFunc::VTADense),
            "vta-conv1d" => Ok(AcceleratorFunc::VTAConv1D),
            "hlscnn-conv2d" => Ok(AcceleratorFunc::HlsCNNConv2D),
            "flex-maxpool" => Ok(AcceleratorFunc::FlexASRMaxPool),
            _ => Err(()),
        }
    }
}

impl Display for AcceleratorFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                AcceleratorFunc::FlexLinear => "flex-linear",
                AcceleratorFunc::FlexLSTM => "flex-lstm",
                AcceleratorFunc::VTADense => "vta-dense",
                AcceleratorFunc::VTAConv1D => "vta-conv1d",
                AcceleratorFunc::HlsCNNConv2D => "hlscnn-conv2d",
                AcceleratorFunc::FlexASRMaxPool => "flex-maxpool",
            }
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AcceleratorFuncData {
    pattern: AcceleratorFunc,
    accelerator: String,
}

// TODO(@gussmith23) Pick a better analysis name.
#[derive(Debug, Clone, PartialEq)]
pub enum MyAnalysisData {
    Literal(ndarray::ArrayD<f64>),
    Num(i64),
    DataType(DataType),
    AccessPattern(AccessPatternData),
    Shape(ShapeData),
    Tuple(Vec<MyAnalysisData>),
    // TODO(@gussmith23) Needed?
    //Tensor(TensorData),
    ComputeType(ComputeType),
    PadType(PadType),
    List(Vec<usize>),
    RelayOperator(RelayOperator),
    RelayActivationLayout(RelayActivationLayout),
    RelayKernelLayout(RelayKernelLayout),
    AcceleratorFunc(AcceleratorFuncData),
}

#[derive(Debug, PartialEq, Eq, Hash, Clone, PartialOrd, Ord, Copy)]
pub enum DataType {
    Bool,
    Int(usize),
    Float(usize),
    Uint(usize),
}

impl Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                DataType::Bool => "bool".into(),
                DataType::Int(x) => format!("int{}", x),
                DataType::Float(x) => format!("float{}", x),
                DataType::Uint(x) => format!("uint{}", x),
            }
        )
    }
}

impl FromStr for DataType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (dtype, bits) = match s.find(char::is_numeric) {
            Some(idx) => s.split_at(idx),
            None => (s, "32"),
        };
        if dtype == "bool" {
            return Ok(DataType::Bool);
        }
        if let Ok(bits) = bits.parse::<usize>() {
            match dtype {
                "int" => Ok(DataType::Int(bits)),
                "float" => Ok(DataType::Float(bits)),
                "uint" => Ok(DataType::Uint(bits)),
                _ => Err(format!("Not supported: {}", dtype)),
            }
        } else {
            Err(format!("cannot parse bits"))
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ShapeData {
    pub shape: IxDyn,
    pub dtype: DataType,
}

/// New version of rangeset.
pub trait RangeSet2 {
    type Index;

    /// Inserts elements, shifting existing ranges as needed.
    fn insert_elements(&mut self, index: Self::Index, num_elements_inserted: usize);

    /// Updates ranges as if `num_elements_removed` elements were removed at
    /// `index`.
    fn remove_elements(&mut self, index: Self::Index, num_elements_removed: usize);

    /// Checks whether `range` is covered by the ranges in this set.
    fn covered(&self, range: (Self::Index, Self::Index)) -> bool;

    /// Adds range. Ranges are half-open.
    fn add_range(&mut self, range: (Self::Index, Self::Index));
}

type BoolVecRangeSet = Vec<bool>;
impl RangeSet2 for BoolVecRangeSet {
    type Index = usize;

    fn insert_elements(&mut self, index: Self::Index, num_elements_inserted: usize) {
        // Make index-1 the last valid index.
        if index >= self.len() {
            self.resize(index, false);
        }
        *self = self[..index]
            .iter()
            .chain(std::iter::repeat(&false).take(num_elements_inserted))
            .chain(self[index..].iter())
            .cloned()
            .collect();
    }

    fn remove_elements(&mut self, index: Self::Index, num_elements_removed: usize) {
        *self = self[..index]
            .iter()
            .chain(self[index + num_elements_removed..].iter())
            .cloned()
            .collect()
    }

    fn covered(&self, range: (Self::Index, Self::Index)) -> bool {
        // If the top end of the range is not actually represented, then those
        // values are implicitly false and so the range is not covered.
        // Otherwise, check that the values are all true.
        range.1 <= self.len() && self[range.0..range.1].iter().all(|v| *v)
    }

    fn add_range(&mut self, range: (Self::Index, Self::Index)) {
        // Make range.1-1 the last valid index.
        if range.1 > self.len() {
            self.resize(range.1, false);
        }
        for i in range.0..range.1 {
            self[i] = true;
        }
    }
}

#[cfg(test)]
mod bool_vec_range_set_tests {
    use super::*;

    #[test]
    fn insert_elements_0() {
        let mut range_set = BoolVecRangeSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((4, 8));
        range_set.add_range((7, 10));
        range_set.insert_elements(5, 5);
        assert!(range_set.covered((0, 3)));
        assert!(range_set.covered((2, 5)));
        assert!(range_set.covered((10, 11)));
        assert!(range_set.covered((4, 5)));
        assert!(range_set.covered((10, 13)));
        assert!(range_set.covered((12, 15)));
    }

    #[test]
    fn insert_elements_1() {
        let mut range_set = BoolVecRangeSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((4, 8));
        range_set.add_range((7, 10));
        range_set.insert_elements(5, 5);
        range_set.add_range((5, 10));
        assert!(range_set.covered((0, 3)));
        assert!(range_set.covered((2, 11)));
        assert!(range_set.covered((4, 13)));
        assert!(range_set.covered((12, 15)));
    }

    #[test]
    fn remove_elements() {
        let mut range_set = BoolVecRangeSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((5, 8));
        range_set.add_range((9, 12));
        range_set.add_range((10, 14));
        range_set.remove_elements(5, 5);
        assert!(range_set.covered((0, 3)));
        assert!(range_set.covered((2, 5)));
        assert!(range_set.covered((5, 7)));
        assert!(range_set.covered((5, 9)));
    }

    #[test]
    fn covered() {
        let mut range_set = BoolVecRangeSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((5, 6));
        range_set.add_range((6, 8));
        range_set.add_range((10, 12));
        range_set.add_range((11, 14));
        assert!(range_set.covered((0, 2)));
        assert!(!range_set.covered((0, 4)));
        assert!(!range_set.covered((2, 5)));
        assert!(!range_set.covered((3, 5)));
        assert!(range_set.covered((5, 7)));
        assert!(range_set.covered((5, 8)));
        assert!(!range_set.covered((5, 9)));
        assert!(range_set.covered((10, 14)));
        assert!(!range_set.covered((10, 16)));
        assert!(!range_set.covered((22, 23)));
    }

    #[test]
    fn test() {
        let mut range_set = BoolVecRangeSet::default();
        range_set.insert_elements(0, 1);
        range_set.add_range((0, 1));
        range_set.insert_elements(33, 2);
        range_set.add_range((33, 35));
        assert!(range_set.covered((0, 1)));
        assert!(!range_set.covered((1, 33)));
        assert!(range_set.covered((33, 35)));
    }
}

/// Used to represent ranges over a set from 0..n, for some n. Ranges are
/// half-open.
type RangeHashSet = HashSet<(usize, usize)>;
pub enum RangeInsertStrategy {
    /// If elements are inserted in the middle of a range, the range gets split
    /// in two.
    BreakRanges,
    /// If elements are inserted in the middle of a range, they get folded into
    /// the range.
    PreserveRanges,
}
pub trait RangeSet {
    type Index;

    /// Updates ranges as if `num_elements_inserted` elements were inserted at
    /// `index`, according to `strategy`.
    fn insert_elements(
        &mut self,
        strategy: RangeInsertStrategy,
        index: Self::Index,
        num_elements_inserted: usize,
    );

    /// Updates ranges as if `num_elements_removed` elements were removed at
    /// `index`.
    fn remove_elements(&mut self, index: Self::Index, num_elements_removed: usize);

    /// Checks whether `range` is covered by the ranges in this set.
    fn covered(&self, range: (Self::Index, Self::Index)) -> bool;

    /// Adds range. Ranges are half-open.
    fn add_range(&mut self, range: (Self::Index, Self::Index));
}
impl RangeSet for RangeHashSet {
    type Index = usize;

    fn insert_elements(
        &mut self,
        strategy: RangeInsertStrategy,
        index: usize,
        num_elements_inserted: usize,
    ) {
        let mut new_ranges = Vec::default();
        for (low, high) in self.drain() {
            assert!(low <= high);
            match strategy {
                RangeInsertStrategy::PreserveRanges => {
                    let (new_low, new_high) = if index < low {
                        (low + num_elements_inserted, high + num_elements_inserted)
                    } else if index >= low && index <= high {
                        (low, high + num_elements_inserted)
                    } else if index > high {
                        (low, high)
                    } else {
                        unreachable!()
                    };
                    new_ranges.push((new_low, new_high));
                }
                RangeInsertStrategy::BreakRanges => {
                    match {
                        if index <= low {
                            (
                                Some((low + num_elements_inserted, high + num_elements_inserted)),
                                None,
                            )
                        } else if index > low && index < high {
                            (
                                Some((low, index)),
                                Some((index + num_elements_inserted, high + num_elements_inserted)),
                            )
                        } else if index >= high {
                            (Some((low, high)), None)
                        } else {
                            unreachable!()
                        }
                    } {
                        (Some(range1), Some(range2)) => {
                            new_ranges.push(range1);
                            new_ranges.push(range2);
                        }
                        (Some(range1), None) => {
                            new_ranges.push(range1);
                        }
                        _ => panic!(),
                    };
                }
            }
        }

        for range in new_ranges.iter() {
            self.insert(*range);
        }
    }

    fn remove_elements(&mut self, index: usize, num_elements_removed: usize) {
        let new_ranges = self
            .drain()
            .filter_map(|(low, high): (usize, usize)| {
                let new_low = if low <= index {
                    low
                } else if low > index {
                    low - std::cmp::min(num_elements_removed, low - index)
                } else {
                    unreachable!()
                };
                let new_high = if index >= high {
                    high
                } else if index < high {
                    high - std::cmp::min(num_elements_removed, high - index)
                } else {
                    unreachable!()
                };

                // If the range is valid and nonempty
                if new_low < new_high {
                    Some((new_low, new_high))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        for new_range in new_ranges {
            self.insert(new_range);
        }
    }

    /// I'm hoping this implementation will be fast.
    fn covered(&self, range: (Self::Index, Self::Index)) -> bool {
        let mut to_be_covered =
            HashSet::<_, std::collections::hash_map::RandomState>::from_iter(range.0..range.1);

        for (low, high) in self.iter() {
            to_be_covered = to_be_covered
                .difference(&HashSet::from_iter(*low..*high))
                .cloned()
                .collect();
        }

        to_be_covered.is_empty()
    }

    /// Adds range. Ranges are half-open.
    fn add_range(&mut self, range: (Self::Index, Self::Index)) {
        self.insert(range);
    }
}

#[cfg(test)]
mod range_hash_set_tests {
    use super::*;

    #[test]
    fn insert_elements_break_ranges() {
        let mut range_set = RangeHashSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((4, 8));
        range_set.add_range((7, 10));
        range_set.insert_elements(RangeInsertStrategy::BreakRanges, 5, 5);
        assert_eq!(range_set.len(), 6);
        assert!(range_set.contains(&(0, 3)));
        assert!(range_set.contains(&(2, 5)));
        assert!(range_set.contains(&(10, 11)));
        assert!(range_set.contains(&(4, 5)));
        assert!(range_set.contains(&(10, 13)));
        assert!(range_set.contains(&(12, 15)));
    }

    #[test]
    fn insert_elements_preserve_ranges() {
        let mut range_set = RangeHashSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((4, 8));
        range_set.add_range((7, 10));
        range_set.insert_elements(RangeInsertStrategy::PreserveRanges, 5, 5);
        assert_eq!(range_set.len(), 4);
        assert!(range_set.contains(&(0, 3)));
        assert!(range_set.contains(&(2, 11)));
        assert!(range_set.contains(&(4, 13)));
        assert!(range_set.contains(&(12, 15)));
    }

    #[test]
    fn remove_elements() {
        let mut range_set = RangeHashSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((2, 6));
        range_set.add_range((5, 8));
        range_set.add_range((9, 12));
        range_set.add_range((10, 14));
        range_set.remove_elements(5, 5);
        assert_eq!(range_set.len(), 4);
        assert!(range_set.contains(&(0, 3)));
        assert!(range_set.contains(&(2, 5)));
        assert!(range_set.contains(&(5, 7)));
        assert!(range_set.contains(&(5, 9)));
    }

    #[test]
    fn covered() {
        let mut range_set = RangeHashSet::default();
        range_set.add_range((0, 3));
        range_set.add_range((5, 6));
        range_set.add_range((6, 8));
        range_set.add_range((10, 12));
        range_set.add_range((11, 14));
        assert!(range_set.covered((0, 2)));
        assert!(!range_set.covered((0, 4)));
        assert!(!range_set.covered((2, 5)));
        assert!(!range_set.covered((3, 5)));
        assert!(range_set.covered((5, 7)));
        assert!(range_set.covered((5, 8)));
        assert!(!range_set.covered((5, 9)));
        assert!(range_set.covered((10, 14)));
        assert!(!range_set.covered((10, 16)));
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AccessPatternData {
    pub shape: IxDyn,
    pub item_shape: IxDyn,
    /// Regions proven to be zero-valued. The outermost map maps a axis index to
    /// a set of usize tuples, indicating the half-open indices [low, high)
    /// which are known to be zero in that axis.
    /// TODO(@gussmith23) Might want to replace this with full partial eval
    /// I realized while implementing this that I could just implement partial
    /// eval, and it would do the same for me. I don't think it would be more
    /// time efficient, though, and I'm even more certain that it wouldn't be
    /// space efficient.
    pub zero_regions: HashMap<Ix, BoolVecRangeSet>,
    /// Indicates whether the access pattern shape for this eclass has
    /// "settled". An access pattern shape for an eclass containing only Relay
    /// nodes is likely to change, because Relay nodes treat access pattern
    /// shapes e.g. ((a, b), (c, d)) as tensor shapes e.g. (a, b, c, d), and
    /// often the Relay nodes choose either the shape or the item shape to store
    /// the tensor shape. The access pattern is thus likely to change when the
    /// class gets merged, so we call it "unsettled". An enode is considered to
    /// have a settled shape if and only if:
    ///
    /// - The node is a Glenside node, and
    /// - All of the node's children of access pattern type are also settled.
    ///
    /// (This specifies how we implement make().)
    ///
    /// An eclass's shape is settled if at least one of its nodes is settled.
    /// (This specifies how we implement merge().)
    ///
    pub access_pattern_shape_settled: bool,
    pub contains_accelerator_calls: bool,
}

impl AccessPatternData {
    /// Convenience method for getting the access pattern dimensions as a
    /// vector.
    /// ```
    /// assert_eq!(
    ///     glenside::language::AccessPatternData {
    ///         shape: ndarray::IxDyn(&[1, 2, 3]),
    ///         item_shape: ndarray::IxDyn(&[4, 5]),
    ///         zero_regions: std::collections::HashMap::default(),
    ///         access_pattern_shape_settled: false,
    ///         contains_accelerator_calls: false,
    ///     }
    ///     .as_vec(),
    ///     vec![1, 2, 3, 4, 5]
    /// );
    /// ```
    pub fn as_vec(&self) -> Vec<usize> {
        self.shape
            .slice()
            .iter()
            .chain(self.item_shape.slice().iter())
            .cloned()
            .collect::<Vec<_>>()
    }
}

impl std::ops::Index<usize> for AccessPatternData {
    type Output = ndarray::Ix;
    fn index(&self, index: usize) -> &Self::Output {
        if index < self.shape.ndim() {
            &self.shape[index]
        } else {
            &self.item_shape[index - self.shape.ndim()]
        }
    }
}

impl std::ops::IndexMut<usize> for AccessPatternData {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if index < self.shape.ndim() {
            &mut self.shape[index]
        } else {
            &mut self.item_shape[index - self.shape.ndim()]
        }
    }
}

pub fn access_windows_resulting_shape(
    access_shape: &IxDyn,
    filters_shape: &IxDyn,
    stride_shape: &IxDyn,
) -> Vec<usize> {
    assert_eq!(access_shape.ndim(), stride_shape.ndim());
    assert_eq!(filters_shape.ndim(), stride_shape.ndim());

    multizip((
        access_shape.slice().iter(),
        filters_shape.slice().iter(),
        stride_shape.slice().iter(),
    ))
    .map(
        |(&dim_len, &kernel_dim_len, &stride): (&usize, &usize, &usize)| {
            let total_dim_len = dim_len;
            assert!(
                total_dim_len >= kernel_dim_len,
                "{} !>= {}",
                total_dim_len,
                kernel_dim_len
            );
            let num_spots = total_dim_len - (kernel_dim_len - 1);
            (num_spots + stride - 1) / stride
        },
    )
    .collect()
}

// #[derive(Debug, Clone, PartialEq)]
// pub struct TensorData {
//     shape: IxDyn,
// }

// TODO(@gussmith23) Pick a better analysis name.
#[derive(Debug, Clone, PartialEq)]
pub struct MyAnalysisDataLegacyData {
    pub(crate) shape: Option<IxDyn>,
    pub(crate) usize_value: Option<usize>,
}
#[derive(Default)]
pub struct MyAnalysis {
    pub name_to_shape: HashMap<String, Vec<usize>>,
    pub name_to_dtype: HashMap<String, DataType>,
}
impl MyAnalysis {
    /// Legacy function: gets Num value as a usize. Before Num, we instead had a
    /// Num construct.
    pub fn get_usize(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> usize {
        match &egraph[id].data {
            &MyAnalysisData::Num(s) => s.try_into().unwrap(),
            _ => panic!(),
        }
    }
    pub(crate) fn get_shape(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> &IxDyn {
        match &egraph[id].data {
            MyAnalysisData::Shape(s) => &s.shape,
            _ => panic!(),
        }
    }
    pub(crate) fn get_dtype(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> &DataType {
        match &egraph[id].data {
            MyAnalysisData::Shape(s) => &s.dtype,
            _ => panic!(),
        }
    }
    pub(crate) fn get_shape_of_value(id: Id, egraph: &EGraph<Language, MyAnalysis>) -> &IxDyn {
        match &egraph[id].data {
            MyAnalysisData::Shape(s) => &s.shape,
            _ => panic!(),
        }
    }
}

impl egg::Analysis<Language> for MyAnalysis {
    type Data = MyAnalysisData;

    fn merge(&mut self, to: &mut Self::Data, from: Self::Data) -> DidMerge {
        match (to, &from) {
            (MyAnalysisData::AccessPattern(a), MyAnalysisData::AccessPattern(b)) => {
                // Merge zero regions.
                // TODO(@gussmith23) Make sure merge returns `true` infrequently
                // Returning `true` more often forces more rebuilds, which kills
                // performance!
                // let mut changed = false;
                // for (axis_index, b.range_set) in b.zero_regions.iter() {
                //     // Skip if `from` doesn't contain any interesting data.
                //     if !b.range_set.iter().any(|v| *v) {
                //         continue;
                //     }

                //     if let Some(a.range_set) = a.zero_regions.get_mut(&axis_index) {
                //         // We first check whether `b.zero_regions` contains
                //         // any information not already known in
                //         // `a.zero_regions`. This is done by checking them
                //         // element-by-element. If it is ever true that
                //         // `b.zero_regions` contains a `true` where
                //         // `a.zero_regions` contains a `false` or does not have
                //         // data (because they may be different lengths), then
                //         // they're different and must be merged.

                //         // TODO(@gussmith23) Delete these
                //         //println!("to: {:?}", a.range_set.len());
                //         //println!("from: {:?}", b.range_set.len());

                //         // Check.
                //         let needs_merge = a.range_set
                //             .iter()
                //             .zip_longest(b.range_set.iter())
                //             .map(|v| {
                //                 match v {
                //                     // `*from` being true implies `*to` must be true.
                //                     Both(to, from) => {
                //                         if *from {
                //                             *from != *to
                //                         } else {
                //                             false
                //                         }
                //                     }
                //                     // If `to` has a value and `from` doesn't, then
                //                     // no merging needed.
                //                     Left(_) => false,
                //                     // If `from` has a value, then we need to merge
                //                     // if that value is true.
                //                     Right(from) => *from,
                //                 }
                //             })
                //             .any(|v| v);

                //         if needs_merge {
                //             *a.range_set = a.range_set
                //                 .iter()
                //                 .zip_longest(b.range_set.iter())
                //                 .map(|v| match v {
                //                     Both(to, from) => *to || *from,
                //                     Left(to) => *to,
                //                     Right(from) => *from,
                //                 })
                //                 .collect();
                //             // changed = true;
                //         }
                //     } else {
                //         // If no info exists for this axis in `a.zero_regions`,
                //         // then we insert the information from
                //         // `b.zero_regions`, but only if there's actual
                //         // useful information there (i.e. at least one `true`
                //         // value).
                //         if b.range_set.iter().any(|v| *v) {
                //             a.zero_regions.insert(*axis_index, b.range_set.clone());
                //             // changed = true;
                //         }
                //     }
                // }

                let (mut a_merged, mut b_merged) = (false, false);

                let num_els_a: usize = a.as_vec().iter().product();
                let num_els_b: usize = b.as_vec().iter().product();
                // Previously, I was comparing the underlying tensor shapes, e.g.:
                //
                // assert_eq!(
                //     // Underlying tensor shape of a/to. Sorry this is
                //     // ugly, can't use as_vec b/c can't capture a_ap
                //     // mutably and also capture its fields mutably.
                //     vec![a.shape.slice(), a.item_shape.slice()].concat(),
                //     // Underlying tensor shape of b/from.
                //     b_ap.as_vec()
                // );
                //
                // But I'm not sure that's actually right; different Glenside
                // operators can change the number of dimensions, and change the
                // number of dimensions differently based on where the access
                // pattern is accessed (e.g. flatten). So we had to make the
                // check weaker. Maybe there's a stronger check than this,
                // though? Again, this all goes away if we get rid of the
                // problem of "unsettled" shapes.
                assert_eq!(
                    num_els_a, num_els_b,
                    "The two underlying tensors should have the same number of elements."
                );

                match (
                    a.access_pattern_shape_settled,
                    b.access_pattern_shape_settled,
                ) {
                    (false, false) => {
                        // Do nothing. Neither one is more correct.
                    }
                    (false, true) => {
                        // Take the shape of b/from and put it into a/to. b/from is settled, and thus correct.
                        let (a_shape_old, a_item_shape_old) =
                            (a.shape.clone(), a.item_shape.clone());
                        a.shape = b.shape.clone();
                        a.item_shape = b.item_shape.clone();
                        a_merged |= (a_shape_old != a.shape) | (a_item_shape_old != a.item_shape);
                    }
                    (true, false) => {
                        // Take the shape of a/to and put it into b/from. Though
                        // we don't actually edit b/from, because it's being
                        // merged into a. But we do calculate whether b changed,
                        // so that we can trigger updates on the enodes that
                        // point to b.
                        b_merged |= (b.shape != a.shape) | (b.item_shape != a.item_shape);
                    }
                    (true, true) => {
                        // If both are settled, then they must match.
                        assert_eq!(a.shape, b.shape);
                        assert_eq!(a.item_shape, b.item_shape);
                    }
                }

                a_merged |= !a.contains_accelerator_calls && b.contains_accelerator_calls;
                b_merged |= a.contains_accelerator_calls && !b.contains_accelerator_calls;
                a.contains_accelerator_calls |= b.contains_accelerator_calls;

                let a_access_pattern_shape_settled_old = a.access_pattern_shape_settled;
                a.access_pattern_shape_settled |= b.access_pattern_shape_settled;
                a_merged |= a_access_pattern_shape_settled_old != a.access_pattern_shape_settled;
                b_merged |= b.access_pattern_shape_settled != a.access_pattern_shape_settled;

                DidMerge(a_merged, b_merged)
            }
            (MyAnalysisData::Tuple(t0), MyAnalysisData::Tuple(t1)) => {
                assert_eq!(t0.len(), t1.len());
                let (mut a_merged, mut b_merged) = (false, false);

                for (v0, v1) in t0.iter_mut().zip(t1.iter().cloned()) {
                    let did_merge = self.merge(v0, v1);
                    a_merged |= did_merge.0;
                    b_merged |= did_merge.1;
                }

                DidMerge(a_merged, b_merged)
            }
            (to @ _, _) => {
                assert_eq!(*to, from);
                return DidMerge(false, false);
            }
        }
    }

    fn make(egraph: &EGraph<Language, Self>, enode: &Language) -> Self::Data {
        fn all_children_are_settled(
            egraph: &EGraph<Language, MyAnalysis>,
            enode: &Language,
        ) -> bool {
            enode
                .children()
                .iter()
                .filter_map(|id| match &egraph[*id].data {
                    MyAnalysisData::AccessPattern(AccessPatternData {
                        access_pattern_shape_settled,
                        ..
                    }) => Some(*access_pattern_shape_settled),
                    _ => None,
                })
                .all(std::convert::identity)
        }

        use Language::*;
        match enode {
            &GetAccessShape([id]) => match egraph[id].data.clone() {
                MyAnalysisData::AccessPattern(mut a) => {
                    a.zero_regions = HashMap::default();
                    a.contains_accelerator_calls = false;
                    // settled should remain the same!
                    MyAnalysisData::AccessPattern(a)
                }
                _ => panic!(),
            },
            &SystolicArrayConv2dIm2colNhwcHwioWithBlocking(
                [rows_id, cols_id, weights_id, data_id, kh_id, kw_id, stride_h_id, stride_w_id],
            ) => {
                let (_rows, _cols, weights, data, kh, kw, stride_h, stride_w) = match (
                    &egraph[rows_id].data,
                    &egraph[cols_id].data,
                    &egraph[weights_id].data,
                    &egraph[data_id].data,
                    &egraph[kh_id].data,
                    &egraph[kw_id].data,
                    &egraph[stride_h_id].data,
                    &egraph[stride_w_id].data,
                ) {
                    (
                        MyAnalysisData::Num(rows),
                        MyAnalysisData::Num(cols),
                        MyAnalysisData::AccessPattern(weights),
                        MyAnalysisData::AccessPattern(data),
                        MyAnalysisData::Num(kh),
                        MyAnalysisData::Num(kw),
                        MyAnalysisData::Num(stride_h),
                        MyAnalysisData::Num(stride_w),
                    ) => (*rows, *cols, weights, data, *kh, *kw, *stride_h, *stride_w),
                    _ => panic!("Does not type check"),
                };
                assert_eq!(weights.shape.ndim() + weights.item_shape.ndim(), 4);
                assert_eq!(data.shape.ndim() + data.item_shape.ndim(), 4);

                let (n, h, w, c) = (data[0], data[1], data[2], data[3]);
                let (_kh, _kw, _c, o) = (weights[0], weights[1], weights[2], weights[3]);
                assert_eq!(c, _c);
                assert_eq!(usize::try_from(kh).unwrap(), _kh);
                assert_eq!(usize::try_from(kw).unwrap(), _kw);

                // These aren't actually requirements at the moment.
                //assert_eq!(o % cols, 0);
                //assert_eq!(c % rows, 0);

                let new_h = (usize::try_from(h).unwrap() - usize::try_from(kh - 1).unwrap()
                    + usize::try_from(stride_h).unwrap()
                    - 1)
                    / usize::try_from(stride_h).unwrap();
                let new_w =
                    (w - usize::try_from(kw - 1).unwrap() + usize::try_from(stride_w).unwrap() - 1)
                        / usize::try_from(stride_w).unwrap();

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(&[n, new_h, new_w, o]),
                    item_shape: IxDyn(&[]),
                    zero_regions: {
                        debug!("Zero regions unimplemented");
                        HashMap::default()
                    },
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: data.contains_accelerator_calls
                        || weights.contains_accelerator_calls,
                })
            }
            &SystolicArrayConv2dNhwcHwioWithBlocking(
                [rows_id, cols_id, weights_id, data_id, kh_id, kw_id, stride_h_id, stride_w_id],
            ) => {
                let (rows, cols, weights, data, kh, kw, stride_h, stride_w) = match (
                    &egraph[rows_id].data,
                    &egraph[cols_id].data,
                    &egraph[weights_id].data,
                    &egraph[data_id].data,
                    &egraph[kh_id].data,
                    &egraph[kw_id].data,
                    &egraph[stride_h_id].data,
                    &egraph[stride_w_id].data,
                ) {
                    (
                        MyAnalysisData::Num(rows),
                        MyAnalysisData::Num(cols),
                        MyAnalysisData::AccessPattern(weights),
                        MyAnalysisData::AccessPattern(data),
                        MyAnalysisData::Num(kh),
                        MyAnalysisData::Num(kw),
                        MyAnalysisData::Num(stride_h),
                        MyAnalysisData::Num(stride_w),
                    ) => (*rows, *cols, weights, data, *kh, *kw, *stride_h, *stride_w),
                    _ => panic!("Does not type check"),
                };
                assert_eq!(weights.shape.ndim() + weights.item_shape.ndim(), 4);
                assert_eq!(data.shape.ndim() + data.item_shape.ndim(), 4);

                let (n, h, w, c) = (data[0], data[1], data[2], data[3]);
                let (_kh, _kw, _c, o) = (weights[0], weights[1], weights[2], weights[3]);
                assert_eq!(c, _c);
                assert_eq!(usize::try_from(kh).unwrap(), _kh);
                assert_eq!(usize::try_from(kw).unwrap(), _kw);

                assert_eq!(o % usize::try_from(cols).unwrap(), 0);
                assert_eq!(c % usize::try_from(rows).unwrap(), 0);

                let new_h =
                    (h - usize::try_from(kh - 1).unwrap() + usize::try_from(stride_h).unwrap() - 1)
                        / usize::try_from(stride_h).unwrap();
                let new_w =
                    (w - usize::try_from(kw - 1).unwrap() + usize::try_from(stride_w).unwrap() - 1)
                        / usize::try_from(stride_w).unwrap();

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(&[n, new_h, new_w, o]),
                    item_shape: IxDyn(&[]),
                    zero_regions: {
                        debug!("Zero regions unimplemented");
                        HashMap::default()
                    },
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: data.contains_accelerator_calls
                        || weights.contains_accelerator_calls,
                })
            }
            &SystolicArrayConv2dIm2colNchwOihwWithBlocking(
                [rows_id, cols_id, weights_id, data_id, kh_id, kw_id, stride_h_id, stride_w_id],
            ) => {
                let (_rows, _cols, weights, data, kh, kw, stride_h, stride_w) = match (
                    &egraph[rows_id].data,
                    &egraph[cols_id].data,
                    &egraph[weights_id].data,
                    &egraph[data_id].data,
                    &egraph[kh_id].data,
                    &egraph[kw_id].data,
                    &egraph[stride_h_id].data,
                    &egraph[stride_w_id].data,
                ) {
                    (
                        MyAnalysisData::Num(rows),
                        MyAnalysisData::Num(cols),
                        MyAnalysisData::AccessPattern(weights),
                        MyAnalysisData::AccessPattern(data),
                        MyAnalysisData::Num(kh),
                        MyAnalysisData::Num(kw),
                        MyAnalysisData::Num(stride_h),
                        MyAnalysisData::Num(stride_w),
                    ) => (*rows, *cols, weights, data, *kh, *kw, *stride_h, *stride_w),
                    _ => panic!("Does not type check"),
                };
                assert_eq!(weights.shape.ndim() + weights.item_shape.ndim(), 4);
                assert_eq!(data.shape.ndim() + data.item_shape.ndim(), 4);

                let (n, c, h, w) = (data[0], data[1], data[2], data[3]);
                let (o, _c, _kh, _kw) = (weights[0], weights[1], weights[2], weights[3]);
                assert_eq!(c, _c);
                assert_eq!(usize::try_from(kh).unwrap(), _kh);
                assert_eq!(usize::try_from(kw).unwrap(), _kw);

                // These aren't actually requirements for the moment.
                //assert_eq!(o % cols, 0);
                //assert_eq!(c % rows, 0);

                let new_h =
                    (h - usize::try_from(kh - 1).unwrap() + usize::try_from(stride_h).unwrap() - 1)
                        / usize::try_from(stride_h).unwrap();
                let new_w =
                    (w - usize::try_from(kw - 1).unwrap() + usize::try_from(stride_w).unwrap() - 1)
                        / usize::try_from(stride_w).unwrap();

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(&[n, o, new_h, new_w]),
                    item_shape: IxDyn(&[]),
                    zero_regions: {
                        debug!("Zero regions unimplemented");
                        HashMap::default()
                    },
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: data.contains_accelerator_calls
                        || weights.contains_accelerator_calls,
                })
            }
            &SystolicArrayConv2dNchwOihwWithBlocking(
                [rows_id, cols_id, weights_id, data_id, kh_id, kw_id, stride_h_id, stride_w_id],
            ) => {
                let (rows, cols, weights, data, kh, kw, stride_h, stride_w) = match (
                    &egraph[rows_id].data,
                    &egraph[cols_id].data,
                    &egraph[weights_id].data,
                    &egraph[data_id].data,
                    &egraph[kh_id].data,
                    &egraph[kw_id].data,
                    &egraph[stride_h_id].data,
                    &egraph[stride_w_id].data,
                ) {
                    (
                        MyAnalysisData::Num(rows),
                        MyAnalysisData::Num(cols),
                        MyAnalysisData::AccessPattern(weights),
                        MyAnalysisData::AccessPattern(data),
                        MyAnalysisData::Num(kh),
                        MyAnalysisData::Num(kw),
                        MyAnalysisData::Num(stride_h),
                        MyAnalysisData::Num(stride_w),
                    ) => (*rows, *cols, weights, data, *kh, *kw, *stride_h, *stride_w),
                    _ => panic!("Does not type check"),
                };
                assert_eq!(weights.shape.ndim() + weights.item_shape.ndim(), 4);
                assert_eq!(data.shape.ndim() + data.item_shape.ndim(), 4);

                let (n, c, h, w) = (data[0], data[1], data[2], data[3]);
                let (o, _c, _kh, _kw) = (weights[0], weights[1], weights[2], weights[3]);
                assert_eq!(c, _c);
                assert_eq!(usize::try_from(kh).unwrap(), _kh);
                assert_eq!(usize::try_from(kw).unwrap(), _kw);

                assert_eq!(o % usize::try_from(cols).unwrap(), 0);
                assert_eq!(c % usize::try_from(rows).unwrap(), 0);

                let new_h =
                    (h - usize::try_from(kh - 1).unwrap() + usize::try_from(stride_h).unwrap() - 1)
                        / usize::try_from(stride_h).unwrap();
                let new_w =
                    (w - usize::try_from(kw - 1).unwrap() + usize::try_from(stride_w).unwrap() - 1)
                        / usize::try_from(stride_w).unwrap();

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(&[n, o, new_h, new_w]),
                    item_shape: IxDyn(&[]),
                    zero_regions: {
                        debug!("Zero regions unimplemented");
                        HashMap::default()
                    },
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: data.contains_accelerator_calls
                        || weights.contains_accelerator_calls,
                })
            }
            AcceleratorCall(ids) => {
                let accelerator_call = &egraph[ids[0]].data;
                let accelerator_func_data = match accelerator_call {
                    MyAnalysisData::AcceleratorFunc(data) => data,
                    _ => panic!(
                        "Invalid data for accelerator function: {:?}",
                        accelerator_call
                    ),
                };
                match accelerator_func_data.pattern {
                    crate::language::AcceleratorFunc::FlexLSTM => {
                        let out_shape = match &egraph[ids[ids.len() - 1]].data {
                            MyAnalysisData::Shape(shape) => shape.shape.slice().to_vec(),
                            _ => panic!("no shape data appended for FlexLSTM"),
                        };

                        MyAnalysisData::AccessPattern(AccessPatternData {
                            zero_regions: HashMap::default(),
                            shape: IxDyn(&out_shape[..]),
                            item_shape: IxDyn(&[]),
                            // Setting this to false for now b/c their shapes are all messed up.
                            // access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: true,
                        })
                    }
                    crate::language::AcceleratorFunc::FlexLinear
                    | crate::language::AcceleratorFunc::VTADense => {
                        match (&egraph[ids[1]].data, &egraph[ids[2]].data) {
                            (
                                MyAnalysisData::AccessPattern(activations),
                                MyAnalysisData::AccessPattern(weights),
                            ) => {
                                assert_eq!(activations.as_vec().len(), 2);
                                assert_eq!(weights.as_vec().len(), 2);
                                MyAnalysisData::AccessPattern(AccessPatternData {
                                    zero_regions: HashMap::default(),
                                    shape: IxDyn(&[activations.as_vec()[0], weights.as_vec()[0]]),
                                    item_shape: IxDyn(&[]),
                                    // Setting this to false for now b/c their shapes are all messed up.
                                    // access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                                    access_pattern_shape_settled: false,
                                    contains_accelerator_calls: true,
                                })
                            }
                            _ => panic!(),
                        }
                    }
                    crate::language::AcceleratorFunc::VTAConv1D => {
                        // TODO: add shape here
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: IxDyn(&[]),
                            item_shape: IxDyn(&[]),
                            zero_regions: HashMap::default(),
                            // Setting this to false for now b/c their shapes are all messed up.
                            // access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: true,
                        })
                    }
                    crate::language::AcceleratorFunc::FlexASRMaxPool => {
                        let mut access = match &egraph[ids[1]].data {
                            MyAnalysisData::AccessPattern(a) => a.clone(),
                            _ => panic!(),
                        };

                        assert_eq!(access.item_shape.ndim(), 2);
                        assert_eq!(access.shape.ndim(), 0);
                        let t = access.item_shape[0];
                        let h = access.item_shape[1];
                        assert_eq!(t % 2, 0);
                        assert_eq!(h % 16, 0);
                        access.item_shape[0] = access.item_shape[0] / 2;
                        access.contains_accelerator_calls = true;
                        // Setting this to false for now b/c their shapes are all messed up.
                        // access.access_pattern_shape_settled =
                        // all_children_are_settled(egraph, enode);
                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::AcceleratorFunc::HlsCNNConv2D => {
                        let access = match ids[1..ids.len() - 1]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(data), MyAnalysisData::AccessPattern(_weight), MyAnalysisData::Shape(strides), MyAnalysisData::Shape(padding), MyAnalysisData::Num(_group), MyAnalysisData::Num(channels), MyAnalysisData::Shape(kernel_size), MyAnalysisData::RelayActivationLayout(_act_layout), MyAnalysisData::RelayKernelLayout(_ker_layout)] =>
                            {
                                let mut data_shape = data
                                    .shape
                                    .slice()
                                    .iter()
                                    .chain(data.item_shape.slice().iter())
                                    .cloned()
                                    .collect::<Vec<_>>();
                                data_shape[2] += padding.shape[0] + padding.shape[2];
                                data_shape[3] += padding.shape[1] + padding.shape[3];
                                let n = data_shape[0].clone();
                                let c = channels.clone();
                                let access_window_shape = access_windows_resulting_shape(
                                    &IxDyn(&data_shape[1..]),
                                    &kernel_size.shape,
                                    &strides.shape,
                                );
                                let h = access_window_shape[1];
                                let w = access_window_shape[2];
                                AccessPatternData {
                                    shape: IxDyn(&[n, c.try_into().unwrap(), h, w]),
                                    item_shape: IxDyn(&[]),
                                    // Setting this to false for now b/c their shapes are all messed up.
                                    // access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                                    access_pattern_shape_settled: false,
                                    zero_regions: HashMap::default(),
                                    contains_accelerator_calls: true,
                                }
                            }
                            _ => panic!("Cannot parse arguments for Conv2D"),
                        };
                        MyAnalysisData::AccessPattern(access)
                    }
                }
            }
            AcceleratorFunc(name) => {
                let accelerator = match &name {
                    crate::language::AcceleratorFunc::FlexLinear
                    | crate::language::AcceleratorFunc::FlexASRMaxPool
                    | crate::language::AcceleratorFunc::FlexLSTM => "flexnlp",
                    crate::language::AcceleratorFunc::VTAConv1D
                    | crate::language::AcceleratorFunc::VTADense => "vta",
                    crate::language::AcceleratorFunc::HlsCNNConv2D => "hlscnn",
                };
                MyAnalysisData::AcceleratorFunc(AcceleratorFuncData {
                    pattern: name.clone(),
                    accelerator: String::from(accelerator),
                })
            }
            RelayActivationLayout(l) => MyAnalysisData::RelayActivationLayout(l.clone()),
            RelayKernelLayout(l) => MyAnalysisData::RelayKernelLayout(l.clone()),
            ConstructTuple(ids) => {
                let tuple_shape = ids
                    .iter()
                    .map(|id| (&egraph[*id].data).clone())
                    .collect::<Vec<_>>();
                MyAnalysisData::Tuple(tuple_shape)
            }
            TupleGetItem(ids) => {
                let index = MyAnalysis::get_usize(ids[1], egraph);
                let data = match &egraph[ids[0]].data {
                    MyAnalysisData::Tuple(x) => x,
                    _ => panic!("Expected {:?} to be a Tuple.", &egraph[ids[0]]),
                };
                data[index].clone()
            }
            RelayOperator(op) => MyAnalysisData::RelayOperator(op.clone()),
            RelayOperatorCall(params) => {
                assert!(params.len() > 0);

                let op_type = match &egraph[params[0]].data {
                    MyAnalysisData::RelayOperator(op_type) => op_type,
                    _ => panic!(),
                };

                match op_type {
                    crate::language::RelayOperator::RelaySqueeze => {
                        assert_eq!(params.len(), 3);
                        let a = match &egraph[params[1]].data {
                            MyAnalysisData::AccessPattern(a) => a,
                            _ => panic!(),
                        };
                        let axes = match &egraph[params[2]].data {
                            MyAnalysisData::List(v) => v,
                            _ => panic!(),
                        };

                        let new_shape = a
                            .as_vec()
                            .iter()
                            .enumerate()
                            .filter_map(|(i, v)| {
                                if axes.contains(&i) {
                                    assert_eq!(
                                        *v, 1,
                                        "Cannot squeeze an axis unless its value is 1."
                                    );
                                    None
                                } else {
                                    Some(*v)
                                }
                            })
                            .collect::<Vec<_>>();

                        if any(&[a], |a| !a.zero_regions.is_empty()) {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }

                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: IxDyn(&new_shape),
                            item_shape: IxDyn(&[]),
                            zero_regions: HashMap::default(),
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: any(&[a], |a| a.contains_accelerator_calls),
                        })
                    }
                    crate::language::RelayOperator::RelayTranspose => {
                        assert_eq!(params.len(), 3);
                        let a = match &egraph[params[1]].data {
                            MyAnalysisData::AccessPattern(a) => a,
                            _ => panic!(),
                        };
                        let axes = match &egraph[params[2]].data {
                            MyAnalysisData::List(v) => v,
                            _ => panic!(),
                        };
                        assert_eq!(a.as_vec().len(), axes.len());

                        let new_shape: Vec<_> = axes.iter().map(|&i| a.as_vec()[i]).collect();

                        if any(&[a], |a| !a.zero_regions.is_empty()) {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }

                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: IxDyn(&new_shape),
                            item_shape: IxDyn(&[]),
                            zero_regions: HashMap::default(),
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: any(&[a], |a| a.contains_accelerator_calls),
                        })
                    }
                    crate::language::RelayOperator::RelayConcatenate => {
                        assert_eq!(params.len(), 3);

                        let axis = match &egraph[params[1]].data {
                            MyAnalysisData::Num(v) => *v,
                            _ => panic!(),
                        };

                        // In the future we need to handle negative axis, but
                        // for now I think they'll always be positive.
                        assert!(axis >= 0);
                        let axis = axis as usize;

                        let access_pattern_iter = match &egraph[params[2]].data {
                            MyAnalysisData::Tuple(t) => t,
                            _ => panic!(),
                        }
                        .iter()
                        .map(|a| match a {
                            MyAnalysisData::AccessPattern(a) => a,
                            _ => panic!(),
                        });

                        let mut shapes = access_pattern_iter
                            .clone()
                            .map(AccessPatternData::as_vec)
                            .collect::<Vec<_>>();

                        assert!(shapes.len() > 0);

                        let new_shape = shapes
                            .drain(..)
                            .reduce(|acc, s| {
                                acc.iter()
                                    .zip(s.iter())
                                    .enumerate()
                                    .map(|(i, (acc_val, this_val))| {
                                        if i == axis {
                                            acc_val + this_val
                                        } else {
                                            assert_eq!(acc_val, this_val);
                                            *acc_val
                                        }
                                    })
                                    .collect()
                            })
                            .unwrap();

                        if any(access_pattern_iter.clone(), |a| !a.zero_regions.is_empty()) {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }

                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: IxDyn(&new_shape),
                            item_shape: IxDyn(&[]),
                            zero_regions: HashMap::default(),
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: any(access_pattern_iter, |a| {
                                a.contains_accelerator_calls
                            }),
                        })
                    }
                    crate::language::RelayOperator::RelayPad => {
                        assert_eq!(params.len(), 3);

                        let a = match &egraph[params[1]].data {
                            MyAnalysisData::AccessPattern(a) => a.clone(),
                            _ => panic!(),
                        };
                        let pad_width = match &egraph[params[2]].data {
                            MyAnalysisData::Shape(ShapeData { shape, .. }) => shape,
                            _ => panic!(),
                        };
                        assert_eq!(
                            pad_width.ndim(),
                            2 * a.as_vec().len(),
                            "There should be two padding values per tensor dimension."
                        );

                        let newshape = a
                            .as_vec()
                            .iter()
                            .enumerate()
                            .map(|(i, v)| pad_width[2 * i] + *v + pad_width[2 * i + 1])
                            .collect::<Vec<_>>();

                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: IxDyn(&newshape[..a.shape.ndim()]),
                            item_shape: IxDyn(&newshape[a.shape.ndim()..]),
                            zero_regions: HashMap::default(),
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: a.contains_accelerator_calls,
                        })
                    }
                    crate::language::RelayOperator::RelayExpandDims => {
                        assert_eq!(params.len(), 4);

                        let mut a = match &egraph[params[1]].data {
                            MyAnalysisData::AccessPattern(a) => a.clone(),
                            _ => panic!(),
                        };
                        // TODO(@gussmith23) This pattern appears a lot, and it's annoying.
                        let axis: i64 = match &egraph[params[2]].data {
                            MyAnalysisData::Num(v) => *v,
                            _ => panic!(),
                        };
                        let num_axis: i64 = match &egraph[params[3]].data {
                            MyAnalysisData::Num(v) => *v,
                            _ => panic!(),
                        };

                        // From the TVM docs.
                        assert!(-(a.as_vec().len() as i64) - 1 <= axis);
                        assert!(axis <= a.as_vec().len() as i64);

                        // Convert negative axis.
                        let axis: usize = if axis < 0 {
                            (axis + (a.as_vec().len() as i64 + 1)) as usize
                        } else {
                            axis as usize
                        };

                        if axis < a.shape.ndim() {
                            a.shape = IxDyn(
                                &a.shape
                                    .slice()
                                    .iter()
                                    .take(axis)
                                    .chain(std::iter::repeat(&1).take(num_axis as usize))
                                    .chain(a.shape.slice().iter().skip(axis))
                                    .cloned()
                                    .collect::<Vec<_>>(),
                            );
                        } else {
                            let axis = axis - a.shape.ndim();
                            a.item_shape = IxDyn(
                                &a.item_shape
                                    .slice()
                                    .iter()
                                    .take(axis)
                                    .chain(std::iter::repeat(&1).take(num_axis as usize))
                                    .chain(a.item_shape.slice().iter().skip(axis))
                                    .cloned()
                                    .collect::<Vec<_>>(),
                            );
                        }

                        a.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(a)
                    }
                    crate::language::RelayOperator::RelayZeros => {
                        let s = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::Shape(s)] => s.shape.clone(),

                            _ => panic!(),
                        };
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: s.clone(),
                            item_shape: IxDyn(&[]),
                            zero_regions: HashMap::default(),
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: false,
                        })
                    }
                    crate::language::RelayOperator::RelayBatchMatmul => {
                        let (a0, a1) = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)] =>
                            {
                                assert_eq!(a0.as_vec().len(), 3);
                                assert_eq!(a1.as_vec().len(), 3);
                                (a0, a1)
                            }
                            _ => panic!(),
                        };

                        let (s0, s1) = (a0.as_vec(), a1.as_vec());
                        assert_eq!(s0[0], s1[0]);
                        assert_eq!(s0[2], s1[2]);
                        let out_shape = vec![s0[0], s0[1], s1[1]];

                        if any(&[a0, a1], |a| !a.zero_regions.is_empty()) {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }

                        let zero_regions = HashMap::default();
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: IxDyn(&out_shape),
                            item_shape: IxDyn(&[]),
                            zero_regions,
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: any([a0, a1], |a| {
                                a.contains_accelerator_calls
                            }),
                        })
                    }
                    crate::language::RelayOperator::RelayLayerNorm => {
                        let mut out = match &egraph[params[1]].data {
                            MyAnalysisData::AccessPattern(a) => a.clone(),
                            _ => panic!(),
                        };
                        out.access_pattern_shape_settled = false;
                        MyAnalysisData::AccessPattern(out)
                    }
                    crate::language::RelayOperator::RelayRound => match &egraph[params[1]].data {
                        x @ MyAnalysisData::AccessPattern(_) => x.clone(),
                        MyAnalysisData::Shape(shape) => {
                            MyAnalysisData::AccessPattern(AccessPatternData {
                                shape: IxDyn(&[]),
                                item_shape: IxDyn(&shape.shape.slice()),
                                access_pattern_shape_settled: false,
                                zero_regions: HashMap::default(),
                                contains_accelerator_calls: false,
                            })
                        }
                        _ => panic!("Invalid rounding"),
                    },
                    crate::language::RelayOperator::RelayLeftShift
                    | crate::language::RelayOperator::RelayRightShift => {
                        match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(access), _] => {
                                MyAnalysisData::AccessPattern(access.clone())
                            }
                            [MyAnalysisData::Shape(shape), _] => {
                                MyAnalysisData::AccessPattern(AccessPatternData {
                                    shape: IxDyn(&[]),
                                    item_shape: IxDyn(&shape.shape.slice()),
                                    access_pattern_shape_settled: false,
                                    zero_regions: HashMap::default(),
                                    contains_accelerator_calls: false,
                                })
                            }
                            _ => panic!("Invalid bit-shifting"),
                        }
                    }
                    crate::language::RelayOperator::RelayStack => {
                        let accesses = params[1..params.len() - 1]
                            .iter()
                            .map(|id| match &egraph[*id].data {
                                MyAnalysisData::AccessPattern(a) => a.clone(),
                                _ => panic!(),
                            })
                            .collect::<Vec<_>>();

                        assert!(accesses.len() > 0);
                        let shape = accesses[0].as_vec();
                        for access in &accesses {
                            if access.as_vec() != shape {
                                todo!("Stack inputs of different shapes not yet supported");
                            }
                        }

                        let axis = match egraph[params[params.len() - 1]].data {
                            MyAnalysisData::Num(v) => i32::try_from(v).unwrap(),
                            _ => panic!(),
                        };
                        // This comes right from the Relay impl.
                        assert!(
                            axis >= -(i32::try_from(shape.len()).unwrap() + 1)
                                && axis < i32::try_from(shape.len()).unwrap() + 1
                        );

                        let shape_len_i32: i32 = shape.len().try_into().unwrap();
                        let axis = if axis < 0 {
                            axis + shape_len_i32 + 1
                        } else {
                            axis
                        };

                        let out_shape: Vec<_> = shape[..axis.try_into().unwrap()]
                            .iter()
                            .chain(std::iter::once(&accesses.len()))
                            .chain(shape[usize::try_from(axis).unwrap()..].iter())
                            .cloned()
                            .collect();

                        if any(&accesses, |a| !a.zero_regions.is_empty()) {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        let zero_regions = HashMap::default();

                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: IxDyn(&out_shape),
                            item_shape: IxDyn(&[]),
                            zero_regions,
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: any(accesses, |a| {
                                a.contains_accelerator_calls
                            }),
                        })
                    }
                    crate::language::RelayOperator::RelayDropout => {
                        let (mut access, _) = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::Literal(f)] => (
                                a.clone(),
                                f.clone()
                                    .into_dimensionality::<Ix0>()
                                    .expect("Rate argument must be a scalar")
                                    .into_scalar(),
                            ),
                            _ => panic!("Parameters do not type check"),
                        };

                        access.access_pattern_shape_settled = false;

                        // See the documentation (well, the code comments...) for dropout.
                        MyAnalysisData::Tuple(vec![
                            MyAnalysisData::AccessPattern(access.clone()),
                            MyAnalysisData::AccessPattern(access),
                        ])
                    }
                    crate::language::RelayOperator::RelayTake => {
                        let (data, indices, axis) = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(data), MyAnalysisData::AccessPattern(indices), MyAnalysisData::Num(axis)] => {
                                (data.clone(), indices.clone(), axis.clone())
                            }
                            _ => panic!(),
                        };

                        let data_shape = data.as_vec();
                        let indices_shape = indices.as_vec();
                        assert!(usize::try_from(axis).unwrap() < data_shape.len());

                        let out_shape: Vec<_> = data_shape[..axis.try_into().unwrap()]
                            .iter()
                            .chain(indices_shape.iter())
                            .chain(data_shape[usize::try_from(axis).unwrap() + 1..].iter())
                            .cloned()
                            .collect();

                        if !data.zero_regions.is_empty() || !indices.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        let zero_regions = HashMap::default();

                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: IxDyn(&out_shape),
                            item_shape: IxDyn(&[]),
                            zero_regions,
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: data.contains_accelerator_calls
                                || indices.contains_accelerator_calls,
                        })
                    }
                    crate::language::RelayOperator::RelayStridedSlice => {
                        let (data, begin, end, strides) = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::Shape(begin), MyAnalysisData::Shape(end), MyAnalysisData::Shape(strides)] => {
                                (
                                    a,
                                    begin.shape.slice(),
                                    end.shape.slice(),
                                    strides.shape.slice(),
                                )
                            }
                            _ => panic!("Parameters do not type check",),
                        };

                        assert!(strides.iter().all(|i| *i == 1));
                        assert_eq!(begin.len(), end.len());
                        assert_eq!(begin.len(), strides.len());
                        assert_eq!(begin.len(), data.as_vec().len());

                        let new_shape: Vec<_> = begin
                            .iter()
                            .zip(end.iter())
                            .map(|(begin, end)| end - begin)
                            .collect();

                        if !data.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        let zero_regions = HashMap::default();

                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: IxDyn(new_shape.as_slice()),
                            item_shape: IxDyn(&[]),
                            zero_regions: zero_regions,
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: data.contains_accelerator_calls,
                        })
                    }
                    crate::language::RelayOperator::RelayAdd
                    | crate::language::RelayOperator::RelayDivide
                    | crate::language::RelayOperator::RelayMultiply
                    | crate::language::RelayOperator::RelayMaximum
                    | crate::language::RelayOperator::RelayMinimum => {
                        let (a, b) = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::AccessPattern(b)] => {
                                (a.clone(), b.clone())
                            }
                            _ => panic!(
                                "Parameters do not type check: {:?} {:?}",
                                egraph[params[1]].data, egraph[params[2]].data
                            ),
                        };

                        if !a.zero_regions.is_empty() || !b.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        let zero_regions = HashMap::default();

                        let a_ndim = a.shape.ndim() + a.item_shape.ndim();
                        let b_ndim = b.shape.ndim() + b.item_shape.ndim();

                        let new_shape = std::iter::repeat(&1usize)
                            .take(if b_ndim > a_ndim { b_ndim - a_ndim } else { 0 })
                            .chain(a.shape.slice().iter())
                            .chain(a.item_shape.slice().iter())
                            .zip(
                                std::iter::repeat(&1usize)
                                    .take(if a_ndim > b_ndim { a_ndim - b_ndim } else { 0 })
                                    .chain(b.shape.slice().iter())
                                    .chain(b.item_shape.slice().iter()),
                            )
                            .map(|(a, b): (&usize, &usize)| {
                                assert!(
                                    a == b || (*a == 1 || *b == 1),
                                    "Shapes can't be broadcast"
                                );
                                std::cmp::max(a, b)
                            })
                            .cloned()
                            .collect::<Vec<_>>();

                        MyAnalysisData::AccessPattern(AccessPatternData {
                            shape: IxDyn(new_shape.as_slice()),
                            item_shape: IxDyn(&[]),
                            zero_regions,
                            access_pattern_shape_settled: false,
                            contains_accelerator_calls: a.contains_accelerator_calls
                                || b.contains_accelerator_calls,
                        })
                    }
                    crate::language::RelayOperator::RelayErf => {
                        let access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a)] => AccessPatternData {
                                shape: a.shape.clone(),
                                item_shape: a.item_shape.clone(),
                                zero_regions: HashMap::default(),
                                access_pattern_shape_settled: false,
                                contains_accelerator_calls: a.contains_accelerator_calls,
                            },
                            _ => panic!("Erf only supports accepting 1 input tensor"),
                        };
                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelaySplit => {
                        match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(data), MyAnalysisData::Num(sections), MyAnalysisData::Num(axis)] =>
                            {
                                let relay_shape = IxDyn(&data.as_vec());
                                let axis = if *axis < 0 {
                                    usize::try_from(
                                        *axis + i64::try_from(relay_shape.ndim()).unwrap(),
                                    )
                                    .unwrap()
                                } else {
                                    *axis as usize
                                };
                                let mut access_vec = Vec::default();
                                for _ in 0..*sections {
                                    let mut oshape: Vec<_> =
                                        relay_shape.slice().iter().cloned().collect();
                                    oshape[axis] =
                                        oshape[axis] / usize::try_from(*sections).unwrap();
                                    access_vec.push(MyAnalysisData::AccessPattern(
                                        AccessPatternData {
                                            shape: IxDyn(&[]),
                                            item_shape: IxDyn(&oshape[..]),
                                            access_pattern_shape_settled: false,
                                            zero_regions: HashMap::default(),
                                            contains_accelerator_calls: data
                                                .contains_accelerator_calls,
                                        },
                                    ));
                                }
                                MyAnalysisData::Tuple(access_vec)
                            }
                            [MyAnalysisData::AccessPattern(data), MyAnalysisData::List(sections), MyAnalysisData::Num(axis)] =>
                            {
                                let relay_shape = IxDyn(&data.as_vec());
                                let axis = if *axis < 0 {
                                    usize::try_from(
                                        *axis + i64::try_from(relay_shape.ndim()).unwrap(),
                                    )
                                    .unwrap()
                                } else {
                                    *axis as usize
                                };
                                let mut begin = 0;
                                let mut access_vec = Vec::default();
                                for index in sections.iter() {
                                    assert!(
                                        *index > begin,
                                        "`index` of the sections must be greater than `begin`"
                                    );
                                    let mut oshape: Vec<_> =
                                        relay_shape.slice().iter().cloned().collect();
                                    oshape[axis] = *index - begin;
                                    begin = *index;
                                    access_vec.push(MyAnalysisData::AccessPattern(
                                        AccessPatternData {
                                            shape: IxDyn(&[]),
                                            item_shape: IxDyn(&oshape[..]),
                                            access_pattern_shape_settled: false,
                                            zero_regions: HashMap::default(),
                                            contains_accelerator_calls: data
                                                .contains_accelerator_calls,
                                        },
                                    ));
                                }
                                assert!(relay_shape[axis] > begin);
                                let mut oshape: Vec<_> =
                                    relay_shape.slice().iter().cloned().collect();
                                oshape[axis] = relay_shape[axis] - begin;
                                access_vec.push(MyAnalysisData::AccessPattern(AccessPatternData {
                                    shape: IxDyn(&[]),
                                    item_shape: IxDyn(&oshape[..]),
                                    access_pattern_shape_settled: false,
                                    zero_regions: HashMap::default(),
                                    contains_accelerator_calls: data.contains_accelerator_calls,
                                }));
                                MyAnalysisData::Tuple(access_vec)
                            }
                            _ => panic!("Invalid call to RelaySplit"),
                        }
                    }
                    crate::language::RelayOperator::RelayMean => {
                        let access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::Num(usize_data)] => {
                                let shape_length =
                                    a.shape.slice().len() + a.item_shape.slice().len();
                                let relay_shape = a
                                    .shape
                                    .slice()
                                    .iter()
                                    .chain(a.item_shape.slice().iter())
                                    .cloned()
                                    .collect::<Vec<_>>();
                                let axis = *usize_data;
                                assert!(usize::try_from(axis).unwrap() < shape_length);
                                if usize::try_from(axis).unwrap() == shape_length - 1 {
                                    AccessPatternData {
                                        shape: IxDyn(&[]),
                                        item_shape: IxDyn(&relay_shape[..axis.try_into().unwrap()]),
                                        zero_regions: HashMap::default(),
                                        access_pattern_shape_settled: false,
                                        contains_accelerator_calls: a.contains_accelerator_calls,
                                    }
                                } else {
                                    AccessPatternData {
                                        shape: IxDyn(&[]),
                                        item_shape: IxDyn(
                                            &[
                                                &relay_shape[..axis.try_into().unwrap()],
                                                &relay_shape[usize::try_from(axis).unwrap() + 1..],
                                            ]
                                            .concat(),
                                        ),
                                        zero_regions: HashMap::default(),
                                        access_pattern_shape_settled: false,
                                        contains_accelerator_calls: a.contains_accelerator_calls,
                                    }
                                }
                            }
                            _ => panic!("Erf only supports accepting 1 input tensor"),
                        };
                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayConv1D => {
                        let access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(data), MyAnalysisData::AccessPattern(weight), MyAnalysisData::Shape(ShapeData { shape: strides, .. }), MyAnalysisData::Shape(ShapeData { shape: padding, .. })] =>
                            {
                                let data_shape = data
                                    .shape
                                    .slice()
                                    .iter()
                                    .chain(data.item_shape.slice().iter())
                                    .cloned()
                                    .collect::<Vec<_>>();
                                let weight_shape = weight
                                    .shape
                                    .slice()
                                    .iter()
                                    .chain(weight.item_shape.slice().iter())
                                    .cloned()
                                    .collect::<Vec<_>>();
                                assert_eq!(data_shape.len(), 3);
                                assert_eq!(weight_shape.len(), 3);
                                assert_eq!(data_shape[1], weight_shape[1]);
                                assert_eq!(strides.ndim(), 1);
                                assert_eq!(padding.ndim(), 2);
                                let output_shape = IxDyn(&[
                                    data_shape[0],
                                    weight_shape[0],
                                    // Here we just compute the new shape of the
                                    // dimension after the convolution. I
                                    // realize it looks convoluted. I wanted to
                                    // use the following function because we
                                    // repeat this calculation all over the
                                    // codebase, and I want one central place
                                    // where we implement the calculation. I
                                    // match on the result just to make sure
                                    // it's the expected length.
                                    match access_windows_resulting_shape(
                                        &IxDyn(&[padding[0] + data_shape[2] + padding[1]]),
                                        &IxDyn(&[weight_shape[2]]),
                                        strides,
                                    )[..]
                                    {
                                        [result] => result,
                                        _ => panic!("unexpected length result"),
                                    },
                                ]);
                                AccessPatternData {
                                    shape: IxDyn(&[]),
                                    item_shape: IxDyn(output_shape.slice()),
                                    zero_regions: HashMap::default(),
                                    access_pattern_shape_settled: false,
                                    contains_accelerator_calls: data.contains_accelerator_calls
                                        || weight.contains_accelerator_calls,
                                }
                            }
                            _ => panic!("Incorrect conv1d arguments"),
                        };
                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayConv2D => {
                        let access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(data), MyAnalysisData::AccessPattern(weight), MyAnalysisData::Shape(strides), MyAnalysisData::Shape(padding), MyAnalysisData::Num(group), MyAnalysisData::Num(_channels), MyAnalysisData::Shape(_kernel_size), MyAnalysisData::RelayActivationLayout(act_layout), MyAnalysisData::RelayKernelLayout(ker_layout)] =>
                            {
                                let (n, c, h, w) = match (act_layout, &data.as_vec()[..]) {
                                    (
                                        crate::language::RelayActivationLayout::NCHW,
                                        &[n, c, h, w],
                                    ) => (n, c, h, w),
                                    (
                                        crate::language::RelayActivationLayout::NHWC,
                                        &[n, h, w, c],
                                    ) => (n, c, h, w),
                                    _ => panic!(),
                                };
                                let (o, i, kh, kw) = match (ker_layout, &weight.as_vec()[..]) {
                                    (crate::language::RelayKernelLayout::OIHW, &[o, i, h, w]) => {
                                        (o, i, h, w)
                                    }
                                    (crate::language::RelayKernelLayout::HWIO, &[h, w, i, o]) => {
                                        (o, i, h, w)
                                    }
                                    _ => panic!(),
                                };
                                let h = padding.shape[0] + h + padding.shape[2];
                                let w = padding.shape[1] + w + padding.shape[3];
                                assert_eq!(strides.shape.ndim(), 2);
                                match *group {
                                    1 => {
                                        assert_eq!(i, c);
                                        let access_window_shape = access_windows_resulting_shape(
                                            &IxDyn(&[h, w]),
                                            &IxDyn(&[kh, kw]),
                                            &strides.shape,
                                        );
                                        assert_eq!(access_window_shape.len(), 2);
                                        let h = access_window_shape[0];
                                        let w = access_window_shape[1];
                                        let out_shape = match act_layout {
                                            crate::language::RelayActivationLayout::NCHW => {
                                                vec![n, o, h, w]
                                            }
                                            crate::language::RelayActivationLayout::NHWC => {
                                                vec![n, h, w, o]
                                            }
                                            _ => panic!(),
                                        };
                                        AccessPatternData {
                                            shape: IxDyn(&out_shape),
                                            item_shape: IxDyn(&[]),
                                            access_pattern_shape_settled: false,
                                            zero_regions: HashMap::default(),
                                            contains_accelerator_calls: data
                                                .contains_accelerator_calls
                                                || weight.contains_accelerator_calls,
                                        }
                                    }
                                    c => {
                                        match act_layout {
                                            crate::language::RelayActivationLayout::NCHW => (),
                                            crate::language::RelayActivationLayout::NHWC => todo!("Not currently supported, supporting only NCHW for PLDI push."),
                                            crate::language::RelayActivationLayout::NCDHW => panic!()
                                        }
                                        match ker_layout {
                                            crate::language::RelayKernelLayout::OIHW => (),
                                            crate::language::RelayKernelLayout::HWIO => todo!("Not currently supported, supporting only OIHW for PLDI push."),
                                            crate::language::RelayKernelLayout::OIDHW => panic!()
                                        }

                                        assert_eq!(
                                            i,
                                            usize::try_from(c).unwrap()
                                                / usize::try_from(*group).unwrap()
                                        );

                                        let access_window_shape = access_windows_resulting_shape(
                                            &IxDyn(&[h, w]),
                                            &IxDyn(&[kh, kw]),
                                            &IxDyn(&strides.shape.slice()),
                                        );
                                        assert_eq!(access_window_shape.len(), 2);
                                        let h = access_window_shape[0];
                                        let w = access_window_shape[1];

                                        AccessPatternData {
                                            shape: IxDyn(&[n, c.try_into().unwrap(), h, w]),
                                            item_shape: IxDyn(&[]),
                                            access_pattern_shape_settled: false,
                                            zero_regions: HashMap::default(),
                                            contains_accelerator_calls: data
                                                .contains_accelerator_calls
                                                || weight.contains_accelerator_calls,
                                        }
                                    }
                                }
                            }
                            _ => panic!("Cannot parse arguments for Conv2D"),
                        };
                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayConv3DTranspose => {
                        let access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(data), MyAnalysisData::AccessPattern(weight), MyAnalysisData::Shape(strides), MyAnalysisData::Shape(padding), MyAnalysisData::Num(group), MyAnalysisData::Num(_channels), MyAnalysisData::Shape(kernel_size), MyAnalysisData::RelayActivationLayout(act_layout), MyAnalysisData::RelayKernelLayout(ker_layout)] =>
                            {
                                let (n, _c, _d, h, w) = match (act_layout, &data.as_vec()[..]) {
                                    (
                                        crate::language::RelayActivationLayout::NCDHW,
                                        &[n, c, d, h, w],
                                    ) => (n, c, d, h, w),
                                    _ => panic!(),
                                };
                                let (o, _i, kd, kh, kw) = match (ker_layout, &weight.as_vec()[..]) {
                                    (
                                        crate::language::RelayKernelLayout::OIDHW,
                                        &[o, i, d, h, w],
                                    ) => (o, i, d, h, w),
                                    _ => panic!(),
                                };
                                assert_eq!(kd, kernel_size.shape[0]);
                                assert_eq!(kh, kernel_size.shape[1]);
                                assert_eq!(kw, kernel_size.shape[2]);

                                // front, top, left, back, down, right
                                let d = padding.shape[0] + w + padding.shape[3];
                                let h = padding.shape[1] + h + padding.shape[4];
                                let w = padding.shape[2] + w + padding.shape[5];
                                assert_eq!(strides.shape.ndim(), 3);
                                match *group {
                                    1 => {
                                        // assert_eq!(i, c);
                                        // let access_window_shape = access_windows_resulting_shape(
                                        //     &IxDyn(&[d, h, w]),
                                        //     &IxDyn(&[kd, kh, kw]),
                                        //     &IxDyn(&strides.shape.slice()),
                                        // );
                                        // assert_eq!(access_window_shape.len(), 3);
                                        // let d = access_window_shape[0];
                                        // let h = access_window_shape[1];
                                        // let w = access_window_shape[2];

                                        let d = d * kernel_size.shape[0];
                                        let h = h * kernel_size.shape[1];
                                        let w = w * kernel_size.shape[2];

                                        AccessPatternData {
                                            shape: IxDyn(&[n, o.try_into().unwrap(), d, h, w]),
                                            item_shape: IxDyn(&[]),
                                            access_pattern_shape_settled: false,
                                            zero_regions: HashMap::default(),
                                            contains_accelerator_calls: data
                                                .contains_accelerator_calls
                                                || weight.contains_accelerator_calls,
                                        }
                                    }
                                    _ => todo!("groups = {}", group),
                                }
                            }
                            _ => panic!("Cannot parse arguments for Conv3DTranspose"),
                        };
                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayConv3D => {
                        let access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(data), MyAnalysisData::AccessPattern(weight), MyAnalysisData::Shape(strides), MyAnalysisData::Shape(padding), MyAnalysisData::Num(group), MyAnalysisData::Num(_channels), MyAnalysisData::Shape(_kernel_size), MyAnalysisData::RelayActivationLayout(act_layout), MyAnalysisData::RelayKernelLayout(ker_layout)] =>
                            {
                                let (n, c, _d, h, w) = match (act_layout, &data.as_vec()[..]) {
                                    (
                                        crate::language::RelayActivationLayout::NCDHW,
                                        &[n, c, d, h, w],
                                    ) => (n, c, d, h, w),
                                    _ => panic!(),
                                };
                                let (o, i, kd, kh, kw) = match (ker_layout, &weight.as_vec()[..]) {
                                    (
                                        crate::language::RelayKernelLayout::OIDHW,
                                        &[o, i, d, h, w],
                                    ) => (o, i, d, h, w),
                                    _ => panic!(),
                                };
                                // front, top, left, back, down, right
                                let d = padding.shape[0] + w + padding.shape[3];
                                let h = padding.shape[1] + h + padding.shape[4];
                                let w = padding.shape[2] + w + padding.shape[5];
                                assert_eq!(strides.shape.ndim(), 3);
                                match *group {
                                    _ => {
                                        assert_eq!(i, c);
                                        let access_window_shape = access_windows_resulting_shape(
                                            &IxDyn(&[d, h, w]),
                                            &IxDyn(&[kd, kh, kw]),
                                            &IxDyn(&strides.shape.slice()),
                                        );
                                        assert_eq!(access_window_shape.len(), 3);
                                        let d = access_window_shape[0];
                                        let h = access_window_shape[1];
                                        let w = access_window_shape[2];

                                        AccessPatternData {
                                            shape: IxDyn(&[n, o.try_into().unwrap(), d, h, w]),
                                            item_shape: IxDyn(&[]),
                                            access_pattern_shape_settled: false,
                                            zero_regions: HashMap::default(),
                                            contains_accelerator_calls: data
                                                .contains_accelerator_calls
                                                || weight.contains_accelerator_calls,
                                        }
                                    }
                                }
                            }
                            _ => panic!("Cannot parse arguments for Conv3D"),
                        };
                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayDense => {
                        let zero_regions = HashMap::default();
                        let access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::AccessPattern(b)] => {
                                let lhs_relay_shape = a
                                    .shape
                                    .slice()
                                    .iter()
                                    .chain(a.item_shape.slice().iter())
                                    .cloned()
                                    .collect::<Vec<_>>();
                                let rhs_relay_shape = b
                                    .shape
                                    .slice()
                                    .iter()
                                    .chain(b.item_shape.slice().iter())
                                    .cloned()
                                    .collect::<Vec<_>>();
                                let batch = lhs_relay_shape[0];
                                let in_feat = lhs_relay_shape[1];
                                let out_feat = rhs_relay_shape[0];
                                assert_eq!(rhs_relay_shape[1], in_feat);
                                let new_shape = [batch, out_feat];
                                AccessPatternData {
                                    shape: IxDyn(&new_shape),
                                    item_shape: IxDyn(&[]),
                                    access_pattern_shape_settled: false,
                                    zero_regions,
                                    contains_accelerator_calls: a.contains_accelerator_calls
                                        || b.contains_accelerator_calls,
                                }
                            }
                            _ => panic!("Dense current only support 2 parameters"),
                        };
                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayCast => {
                        match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(data), _] => {
                                let mut out = data.clone();
                                out.access_pattern_shape_settled = false;
                                MyAnalysisData::AccessPattern(out)
                            }
                            [MyAnalysisData::Shape(from_shape), MyAnalysisData::DataType(dtype)] => {
                                MyAnalysisData::Shape(ShapeData {
                                    shape: from_shape.shape.clone(),
                                    dtype: dtype.clone(),
                                })
                            }
                            _ => panic!("Invalid cast"),
                        }
                    }
                    crate::language::RelayOperator::RelayClip => {
                        match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(access), _, _] => {
                                let mut out = access.clone();
                                out.access_pattern_shape_settled = false;
                                MyAnalysisData::AccessPattern(out)
                            }
                            [MyAnalysisData::Shape(shape), _, _] => {
                                MyAnalysisData::AccessPattern(AccessPatternData {
                                    shape: IxDyn(&[]),
                                    item_shape: IxDyn(&shape.shape.slice()),
                                    access_pattern_shape_settled: false,
                                    zero_regions: HashMap::default(),
                                    contains_accelerator_calls: false,
                                })
                            }
                            _ => panic!("Invalid Clip"),
                        }
                    }
                    crate::language::RelayOperator::RelayReshape => {
                        let zero_regions = HashMap::default();
                        let access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(access), MyAnalysisData::Shape(shape_data)] => {
                                AccessPatternData {
                                    shape: IxDyn(&[]),
                                    item_shape: IxDyn(shape_data.shape.slice()),
                                    access_pattern_shape_settled: false,
                                    zero_regions,
                                    contains_accelerator_calls: access.contains_accelerator_calls,
                                }
                            }
                            _ => panic!("Cannot match parameters for Reshape operator"),
                        };
                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayBiasAdd => {
                        let mut access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::AccessPattern(_), MyAnalysisData::Num(_) | MyAnalysisData::Shape(_)] => {
                                a.clone()
                            }
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayBatchFlatten => {
                        let mut access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a)] => a.clone(),
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        assert!(access.shape.ndim() + access.item_shape.ndim() > 0);

                        // TODO(@gussmith23) Assuming NCHW layout
                        // TODO(@gussmith23) I'm just doing something arbitrary
                        // w/ access axis.
                        if access.shape.ndim() + access.item_shape.ndim() == 1 {
                            access.shape = IxDyn(&[access[0]]);
                        } else {
                            access.shape = IxDyn(&[
                                access[0],
                                access
                                    .shape
                                    .slice()
                                    .iter()
                                    .chain(access.item_shape.slice().iter())
                                    .skip(1)
                                    .product(),
                            ]);
                        }
                        access.item_shape = IxDyn(&[]);

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayGlobalAvgPool2D => {
                        let (mut access, layout) = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::RelayActivationLayout(l)] => {
                                (a.clone(), l.clone())
                            }
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        assert_eq!(access.shape.ndim() + access.item_shape.ndim(), 4);

                        match layout {
                            crate::language::RelayActivationLayout::NCHW => {
                                access[2] = 1;
                                access[3] = 1;
                            }
                            crate::language::RelayActivationLayout::NHWC => {
                                access[1] = 1;
                                access[2] = 1;
                            }
                            crate::language::RelayActivationLayout::NCDHW => panic!(),
                        }

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayMaxPool2D => {
                        let (mut access, pool_size, strides, padding, layout) = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::Shape(pool_size), MyAnalysisData::Shape(strides), MyAnalysisData::Shape(padding), MyAnalysisData::RelayActivationLayout(l)] => {
                                (a.clone(), pool_size, strides, padding, l)
                            }
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        assert_eq!(access.shape.ndim() + access.item_shape.ndim(), 4);
                        assert_eq!(pool_size.shape.ndim(), 2);
                        assert_eq!(strides.shape.ndim(), 2);
                        assert_eq!(padding.shape.ndim(), 4);

                        match layout {
                            crate::language::RelayActivationLayout::NCHW => {
                                // Sorry for the horrific indentation...
                                access[2] =
                                // The dimension plus padding
                                    (((padding.shape[0] + access[2] + padding.shape[2])
                                      // Get the number of spots where we could pool
                                      - (pool_size.shape[0] - 1))
                                     // Then calculate the spots we actually pool at
                                     // using the stride
                                     + strides.shape[0]
                                     - 1)
                                    / strides.shape[0];
                                access[3] = (((padding.shape[1] + access[3] + padding.shape[3])
                                              // Get the number of spots where we could pool
                                              - (pool_size.shape[1] - 1))
                                             // Then calculate the spots we actually pool at
                                             // using the stride
                                             + strides.shape[1]
                                    - 1)
                                    / strides.shape[1];
                            }
                            crate::language::RelayActivationLayout::NHWC => {
                                // Sorry for the horrific indentation...
                                access[1] =
                                // The dimension plus padding
                                    (((padding.shape[0] + access[1] + padding.shape[2])
                                      // Get the number of spots where we could pool
                                      - (pool_size.shape[0] - 1))
                                     // Then calculate the spots we actually pool at
                                     // using the stride
                                     + strides.shape[0]
                                     - 1)
                                    / strides.shape[0];
                                access[2] = (((padding.shape[1] + access[2] + padding.shape[3])
                                              // Get the number of spots where we could pool
                                              - (pool_size.shape[1] - 1))
                                             // Then calculate the spots we actually pool at
                                             // using the stride
                                             + strides.shape[1]
                                    - 1)
                                    / strides.shape[1];
                            }
                            crate::language::RelayActivationLayout::NCDHW => panic!(),
                        }

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayReLU | RelayNegative | RelaySqrt => {
                        let mut access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a)] => a.clone(),
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayLeakyReLU => {
                        let mut access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::Literal(_)] => {
                                a.clone()
                            }
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelaySigmoid
                    | crate::language::RelayOperator::RelayTanh => {
                        let mut access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a)] => a.clone(),
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayLogSoftmax
                    | crate::language::RelayOperator::RelaySoftmax => {
                        let mut access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::Num(_) | MyAnalysisData::Shape(_)] => {
                                a.clone()
                            }
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayBatchNormInference => {
                        let mut access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::AccessPattern(_), MyAnalysisData::AccessPattern(_), MyAnalysisData::AccessPattern(_), MyAnalysisData::AccessPattern(_), MyAnalysisData::Num(_) | MyAnalysisData::Shape(_), MyAnalysisData::Literal(_)] => {
                                a.clone()
                            }
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayAvgPool2D => {
                        let (mut access, pool_size, strides, padding, layout) = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::Shape(pool_size), MyAnalysisData::Shape(strides), MyAnalysisData::Shape(padding), MyAnalysisData::RelayActivationLayout(l)] => {
                                (a.clone(), pool_size, strides, padding, l)
                            }
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        assert_eq!(access.shape.ndim() + access.item_shape.ndim(), 4);
                        assert_eq!(pool_size.shape.ndim(), 2);
                        assert_eq!(strides.shape.ndim(), 2);
                        assert_eq!(padding.shape.ndim(), 4);

                        match layout {
                            crate::language::RelayActivationLayout::NCHW => {
                                // Sorry for the horrific indentation...
                                access[2] =
                                // The dimension plus padding
                                    (((padding.shape[0] + access[2] + padding.shape[2])
                                      // Get the number of spots where we could pool
                                      - (pool_size.shape[0] - 1))
                                     // Then calculate the spots we actually pool at
                                     // using the stride
                                     + strides.shape[0]
                                     - 1)
                                    / strides.shape[0];
                                access[3] = (((padding.shape[1] + access[3] + padding.shape[3])
                                              // Get the number of spots where we could pool
                                              - (pool_size.shape[1] - 1))
                                             // Then calculate the spots we actually pool at
                                             // using the stride
                                             + strides.shape[1]
                                    - 1)
                                    / strides.shape[1];
                            }
                            crate::language::RelayActivationLayout::NHWC => {
                                // Sorry for the horrific indentation...
                                access[1] =
                                // The dimension plus padding
                                    (((padding.shape[0] + access[1] + padding.shape[2])
                                      // Get the number of spots where we could pool
                                      - (pool_size.shape[0] - 1))
                                     // Then calculate the spots we actually pool at
                                     // using the stride
                                     + strides.shape[0]
                                     - 1)
                                    / strides.shape[0];
                                access[2] = (((padding.shape[1] + access[2] + padding.shape[3])
                                              // Get the number of spots where we could pool
                                              - (pool_size.shape[1] - 1))
                                             // Then calculate the spots we actually pool at
                                             // using the stride
                                             + strides.shape[1]
                                    - 1)
                                    / strides.shape[1];
                            }
                            crate::language::RelayActivationLayout::NCDHW => panic!(),
                        }

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                    crate::language::RelayOperator::RelayUpSampling => {
                        let mut access = match params[1..]
                            .iter()
                            .map(|id| &egraph[*id].data)
                            .collect::<Vec<_>>()[..]
                        {
                            [MyAnalysisData::AccessPattern(a), MyAnalysisData::Literal(scale_h), MyAnalysisData::Literal(scale_w), MyAnalysisData::RelayActivationLayout(layout)] =>
                            {
                                assert_eq!(
                                    layout.clone(),
                                    crate::language::RelayActivationLayout::NCHW,
                                    "upsampling only supports NCHW"
                                );
                                // let mut shape = array![a.shape[0], a.shape[1], scale_h.into() * shape[2], scale_w.into() * shape[w]];
                                let mut shape = a.shape.clone();
                                assert_eq!(scale_h.ndim(), 0);
                                assert_eq!(scale_w.ndim(), 0);
                                shape[2] =
                                    (scale_h.first().unwrap() * (shape[2] as f64)).round() as usize;
                                shape[3] =
                                    (scale_w.first().unwrap() * (shape[3] as f64)).round() as usize;

                                AccessPatternData {
                                    shape: shape.clone(),
                                    item_shape: a.item_shape.clone(),
                                    zero_regions: a.zero_regions.clone(),
                                    access_pattern_shape_settled: false,
                                    contains_accelerator_calls: a.contains_accelerator_calls,
                                }
                            }
                            _ => panic!("Parameters do not type check"),
                        };

                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        access.zero_regions = HashMap::default();

                        access.access_pattern_shape_settled = false;

                        MyAnalysisData::AccessPattern(access)
                    }
                }
            }
            &AccessLiteral(id) => match &egraph[id].data {
                MyAnalysisData::Literal(t) => MyAnalysisData::AccessPattern(AccessPatternData {
                    zero_regions: {
                        debug!("Zero regions unimplemented on line {}", std::line!());
                        HashMap::default()
                    },
                    shape: IxDyn(&[]),
                    item_shape: IxDyn(t.shape()),
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: false,
                }),
                _ => panic!(),
            },
            &ConstantTensor([_value, shape]) => match &egraph[shape].data {
                MyAnalysisData::Shape(s) => MyAnalysisData::Shape(s.clone()),
                _ => panic!(),
            },
            &NotNanFloat64(v) => MyAnalysisData::Literal(ndarray::arr0(v.into_inner()).into_dyn()),
            &Literal(id) => match &egraph[id].data {
                t @ MyAnalysisData::Literal(_) => t.clone(),
                _ => panic!(),
            },
            &AccessTranspose([access_id, list_id]) => {
                let access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!(),
                };
                let list = match &egraph[list_id].data {
                    MyAnalysisData::List(l) => l,
                    _ => panic!(),
                };
                assert_eq!(
                    access.shape.ndim() + access.item_shape.ndim(),
                    list.len(),
                    "Number of items in list should equal the number of axes in the first argument"
                );
                let tmp = access
                    .shape
                    .slice()
                    .iter()
                    .chain(access.item_shape.slice().iter())
                    .collect::<Vec<_>>();
                let new_shape = list.iter().map(|i| *tmp[*i]).collect::<Vec<_>>();

                // Re-sort zero regions.
                let mut new_zero_regions = HashMap::default();
                for (new_axis_index, old_axis_index) in list.iter().enumerate() {
                    if let Some(val) = access.zero_regions.get(old_axis_index) {
                        new_zero_regions.insert(new_axis_index, val.clone());
                    }
                }

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(&new_shape[..access.shape.ndim()]),
                    item_shape: IxDyn(&new_shape[access.shape.ndim()..]),
                    zero_regions: new_zero_regions,
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: access.contains_accelerator_calls,
                })
            }
            List(list) => {
                let list = list
                    .iter()
                    .map(|id| MyAnalysis::get_usize(*id, egraph))
                    .collect::<Vec<_>>();
                MyAnalysisData::List(list)
            }
            &AccessBroadcast([access_id, shape_id]) => {
                let access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!(),
                };
                let shape = match &egraph[shape_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!(
                        "Expected access shape as second argument of access-broadcast, got {:?}",
                        egraph[shape_id]
                    ),
                };

                assert_eq!(
                    access.shape.ndim() + access.item_shape.ndim(),
                    shape.shape.ndim() + shape.item_shape.ndim(),
                    "Shape we're broadcasting to should have the same number of dimensions as the shape we're broadcasting from"
                );

                let new_shape = access
                    .shape
                    .slice()
                    .iter()
                    .chain(access.item_shape.slice().iter())
                    .zip(
                        shape
                            .shape
                            .slice()
                            .iter()
                            .chain(shape.item_shape.slice().iter()),
                    )
                    .map(|(broadcast_from_dim, broadcast_to_dim): (&usize, &usize)| {
                        assert!(
                            *broadcast_from_dim == 1 || broadcast_from_dim == broadcast_to_dim,
                            "Expected broadcast_from_dim to be 1 or {}, got {}",
                            *broadcast_to_dim,
                            *broadcast_from_dim
                        );
                        *broadcast_to_dim
                    })
                    .collect::<Vec<_>>();

                if !access.zero_regions.is_empty() {
                    debug!(
                        "Throwing away zero region analysis data on line {}",
                        std::line!()
                    );
                }

                assert_eq!(
                    new_shape.len(),
                    access.shape.ndim() + access.item_shape.ndim()
                );

                MyAnalysisData::AccessPattern(AccessPatternData {
                    shape: IxDyn(&new_shape[..access.shape.ndim()]),
                    item_shape: IxDyn(&new_shape[access.shape.ndim()..]),
                    // TODO(@gussmith23) Implement zero regions
                    // It's harmless (I think) if `zero_regions` defaults to
                    // empty, but for it to be useful, we need to implement it
                    // for each operator.
                    zero_regions: {
                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        HashMap::default()
                    },
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: access.contains_accelerator_calls,
                })
            }
            &AccessInsertAxis([access_id, axis_id]) => {
                let mut access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!(),
                };
                // TODO(@gussmith23) Implement zero_regions
                if !access.zero_regions.is_empty() {
                    debug!(
                        "Throwing away zero region analysis data on line {}",
                        std::line!()
                    );
                    access.zero_regions = HashMap::default();
                }
                let axis = match egraph[axis_id].data {
                    MyAnalysisData::Num(v) => v,
                    _ => panic!(),
                };

                assert!(
                    usize::try_from(axis).unwrap()
                        <= access.shape.ndim() + access.item_shape.ndim()
                );

                if usize::try_from(axis).unwrap() <= access.shape.ndim() {
                    access.shape = IxDyn(
                        access.shape.slice()[..axis.try_into().unwrap()]
                            .iter()
                            .cloned()
                            .chain(std::iter::once(1))
                            .chain(
                                access.shape.slice()[axis.try_into().unwrap()..]
                                    .iter()
                                    .cloned(),
                            )
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );
                } else {
                    let n = access.shape.ndim();
                    access.item_shape = IxDyn(
                        access.item_shape.slice()[..usize::try_from(axis).unwrap() - n]
                            .iter()
                            .cloned()
                            .chain(std::iter::once(1))
                            .chain(
                                access.item_shape.slice()[usize::try_from(axis).unwrap() - n..]
                                    .iter()
                                    .cloned(),
                            )
                            .collect::<Vec<_>>()
                            .as_slice(),
                    );
                }

                access.access_pattern_shape_settled = all_children_are_settled(egraph, enode);

                MyAnalysisData::AccessPattern(access)
            }
            &AccessSqueeze([access_id, axis_id]) => {
                let mut access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!(),
                };
                // TODO(@gussmith23) Implement zero_regions
                if !access.zero_regions.is_empty() {
                    debug!(
                        "Throwing away zero region analysis data on line {}",
                        std::line!()
                    );
                    access.zero_regions = HashMap::default();
                }
                let axis = MyAnalysis::get_usize(axis_id, egraph);
                use ndarray::RemoveAxis;
                if axis < access.shape.ndim() {
                    assert_eq!(
                        access.shape[axis], 1,
                        "Expected axis {} of {:?} to be 1",
                        axis, access.shape
                    );
                    access.shape = access.shape.remove_axis(ndarray::Axis(axis));
                } else {
                    assert_eq!(access.item_shape[axis - access.shape.ndim()], 1);
                    access.item_shape = access
                        .item_shape
                        .remove_axis(ndarray::Axis(axis - access.shape.ndim()));
                }

                access.access_pattern_shape_settled = all_children_are_settled(egraph, enode);

                MyAnalysisData::AccessPattern(access)
            }
            &AccessPad([access_id, pad_type_id, axis_id, pad_before_id, pad_after_id]) => {
                let mut access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!("Expected AccessPattern, got {:#?}", &egraph[access_id].data),
                };
                let pad_type = match &egraph[pad_type_id].data {
                    MyAnalysisData::PadType(t) => t,
                    _ => panic!(),
                };
                let axis = MyAnalysis::get_usize(axis_id, egraph);
                assert!(axis < access.shape.ndim() + access.item_shape.ndim());
                let orig_axis_val = access[axis];
                let pad_before = MyAnalysis::get_usize(pad_before_id, egraph);
                let pad_after = MyAnalysis::get_usize(pad_after_id, egraph);
                if axis < access.shape.ndim() {
                    access.shape[axis] += pad_before + pad_after;
                } else {
                    access.item_shape[axis - access.shape.ndim()] += pad_before + pad_after;
                };

                // TODO(@gussmith23) Remove this after figuring out padding issues
                for (axis, val) in &access.zero_regions {
                    assert!(
                        val.len() <= access[*axis],
                        "{} > {}",
                        val.len(),
                        access[*axis]
                    );
                }

                // Update zero regions
                match pad_type {
                    crate::language::PadType::MinPadding => {
                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                            access.zero_regions = HashMap::default();
                        }
                    }
                    crate::language::PadType::ZeroPadding => {
                        if !access.zero_regions.contains_key(&axis) {
                            access.zero_regions.insert(axis, BoolVecRangeSet::default());
                        }
                        // Update the zero regions. Order here is important (we
                        // do the end padding first, then the beginning)
                        // TODO(@gussmith23) Written in a rush.
                        access
                            .zero_regions
                            .get_mut(&axis)
                            .unwrap()
                            .insert_elements(orig_axis_val, pad_after);
                        access
                            .zero_regions
                            .get_mut(&axis)
                            .unwrap()
                            .add_range((orig_axis_val, orig_axis_val + pad_after));
                        access
                            .zero_regions
                            .get_mut(&axis)
                            .unwrap()
                            .insert_elements(0, pad_before);
                        access
                            .zero_regions
                            .get_mut(&axis)
                            .unwrap()
                            .add_range((0, pad_before));
                    }
                }

                // TODO(@gussmith23) Remove this after figuring out padding issues
                for (axis, val) in &access.zero_regions {
                    assert!(val.len() <= access[*axis]);
                }

                access.access_pattern_shape_settled = all_children_are_settled(egraph, enode);

                MyAnalysisData::AccessPattern(access)
            }
            &AccessTensor(t_id) => {
                let shape = match &egraph[t_id].data {
                    MyAnalysisData::Shape(l) => l.shape.clone(),
                    _ => panic!(),
                };
                MyAnalysisData::AccessPattern(AccessPatternData {
                    // TODO(@gussmith23) Implement zero regions
                    // It's harmless (I think) if `zero_regions` defaults to
                    // empty, but for it to be useful, we need to implement it
                    // for each operator.
                    zero_regions: { HashMap::default() },
                    shape: shape.clone(),
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    item_shape: IxDyn(&[]),
                    contains_accelerator_calls: false,
                })
            }
            &AccessShiftRight(a_id) => {
                let a = match &egraph[a_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!(),
                };

                let combined = a
                    .shape
                    .as_array_view()
                    .iter()
                    .chain(a.item_shape.as_array_view().iter())
                    .cloned()
                    .collect::<Vec<_>>();
                MyAnalysisData::AccessPattern(AccessPatternData {
                    // TODO(@gussmith23) Implement zero regions
                    // It's harmless (I think) if `zero_regions` defaults to
                    // empty, but for it to be useful, we need to implement it
                    // for each operator.
                    zero_regions: {
                        if !a.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        HashMap::default()
                    },
                    shape: IxDyn(&combined[..(a.shape.ndim().saturating_sub(1))]),
                    item_shape: IxDyn(&combined[(a.shape.ndim().saturating_sub(1))..]),
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: a.contains_accelerator_calls,
                })
            }
            &AccessPair([a0_id, a1_id]) => {
                let (a0, a1) = match (&egraph[a0_id].data, &egraph[a1_id].data) {
                    (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => {
                        (a0, a1)
                    }
                    _ => panic!(),
                };

                // assert_eq!(a0.shape, a1.shape);
                // assert_eq!(a0.item_shape, a1.item_shape);

                MyAnalysisData::AccessPattern(AccessPatternData {
                    // TODO(@gussmith23) Implement zero regions
                    // It's harmless (I think) if `zero_regions` defaults to
                    // empty, but for it to be useful, we need to implement it
                    // for each operator.
                    zero_regions: {
                        if !a0.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        if !a1.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        HashMap::default()
                    },
                    shape: a0.shape.clone(),
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    item_shape: IxDyn(
                        std::iter::once(2)
                            .chain(a0.item_shape.as_array_view().iter().cloned())
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                    contains_accelerator_calls: a0.contains_accelerator_calls
                        || a1.contains_accelerator_calls,
                })
            }
            &AccessSlice([access_id, axis_id, low_id, high_id]) => {
                let mut new_access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!(),
                };
                let axis: usize = Self::get_usize(axis_id, egraph);
                let low: usize = Self::get_usize(low_id, egraph);
                let high: usize = Self::get_usize(high_id, egraph);
                let original_axis_value = new_access[axis];

                assert!(new_access.shape.ndim() + new_access.item_shape.ndim() > axis);
                if axis < new_access.shape.ndim() {
                    assert!(low < new_access.shape[axis]);
                    assert!(high <= new_access.shape[axis]);
                    new_access.shape[axis] = high - low;
                } else {
                    assert!(low < new_access.item_shape[axis - new_access.shape.ndim()]);
                    assert!(high <= new_access.item_shape[axis - new_access.shape.ndim()]);
                    new_access.item_shape[axis - new_access.shape.ndim()] = high - low;
                }

                // Update zero regions
                if let Some(range_set) = new_access.zero_regions.get_mut(&axis) {
                    // TODO(@gussmith23) should really just have an "envelope"
                    range_set.remove_elements(high, original_axis_value - high);
                    range_set.remove_elements(0, low - 0);
                }

                new_access.access_pattern_shape_settled = all_children_are_settled(egraph, enode);

                MyAnalysisData::AccessPattern(new_access)
            }
            &AccessConcatenate([a0_id, a1_id, axis_id]) => {
                let axis = Self::get_usize(axis_id, egraph);
                let mut new_access = match &egraph[a0_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!(),
                };
                let a1 = match &egraph[a1_id].data {
                    MyAnalysisData::AccessPattern(a) => {
                        if egraph[a1_id].nodes.iter().all(|n| match n {
                            Language::RelayOperatorCall(_) => true,
                            _ => false,
                        }) {
                            let relay_shape = IxDyn(&a.as_vec());
                            let new_axis = new_access.shape.ndim();
                            assert!(new_axis <= relay_shape.ndim());
                            AccessPatternData {
                                zero_regions: HashMap::default(),
                                shape: IxDyn(&relay_shape.slice()[..new_axis]),
                                item_shape: IxDyn(&relay_shape.slice()[new_axis..]),
                                access_pattern_shape_settled: all_children_are_settled(
                                    egraph, enode,
                                ),
                                contains_accelerator_calls: a.contains_accelerator_calls,
                            }
                        } else {
                            a.clone()
                        }
                    }
                    _ => panic!(),
                };
                // TODO(@gussmith23) Implement zero_regions
                if !new_access.zero_regions.is_empty() {
                    debug!(
                        "Throwing away zero region analysis data on line {}",
                        std::line!()
                    );
                    new_access.zero_regions = HashMap::default();
                }
                if !a1.zero_regions.is_empty() {
                    debug!(
                        "Throwing away zero region analysis data on line {}",
                        std::line!()
                    );
                }
                assert_eq!(new_access.shape.ndim(), a1.shape.ndim(),);
                assert_eq!(new_access.item_shape.ndim(), a1.item_shape.ndim(),);
                assert!(axis < a1.shape.ndim() + a1.item_shape.ndim());
                if axis < new_access.shape.ndim() {
                    new_access.shape[axis] += a1.shape[axis];
                } else {
                    new_access.item_shape[axis - new_access.shape.ndim()] +=
                        a1.item_shape[axis - new_access.shape.ndim()];
                }

                new_access.access_pattern_shape_settled = all_children_are_settled(egraph, enode);

                // new_access.relay_shape = Some(IxDyn(&[new_access.shape.slice(), new_access.item_shape.slice()].concat()));
                MyAnalysisData::AccessPattern(new_access)
            }
            &AccessShape([shape_id, item_shape_id]) => {
                MyAnalysisData::AccessPattern(AccessPatternData {
                    zero_regions: { HashMap::default() },
                    shape: match &egraph[shape_id].data {
                        MyAnalysisData::Shape(s) => s.shape.clone(),
                        _ => panic!(),
                    },
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    item_shape: match &egraph[item_shape_id].data {
                        MyAnalysisData::Shape(s) => s.shape.clone(),
                        _ => panic!(),
                    },
                    contains_accelerator_calls: false,
                })
            }
            Shape(list) => MyAnalysisData::Shape(ShapeData {
                shape: IxDyn(
                    list.iter()
                        .map(|id: &Id| MyAnalysis::get_usize(*id, egraph))
                        .collect::<Vec<_>>()
                        .as_slice(),
                ),
                dtype: crate::language::DataType::Uint(64),
            }),
            &AccessReshape([access_id, access_shape_id]) => {
                let a = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    MyAnalysisData::Shape(s) => AccessPatternData {
                        shape: s.shape.clone(),
                        item_shape: IxDyn(&[]),
                        zero_regions: HashMap::default(),
                        access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                        contains_accelerator_calls: false,
                    },
                    _ => panic!("Expected an access as the first argument to access-reshape"),
                };
                let mut new_shape = match &egraph[access_shape_id].data {
                    MyAnalysisData::AccessPattern(a) => a.clone(),
                    _ => panic!(),
                };
                // TODO(@gussmith23) Implement zero_regions
                new_shape.zero_regions = HashMap::default();
                if !a.zero_regions.is_empty() {
                    debug!(
                        "Throwing away zero region analysis data on line {}",
                        std::line!()
                    );
                }
                // TODO(@gussmith23) this should definitely not be commented out...
                // assert_eq!(
                //     a.shape.as_array_view().iter().product::<usize>(),
                //     new_shape.shape.as_array_view().iter().product::<usize>(),
                // );

                //new_shape.access_pattern_shape_settled = all_children_are_settled(egraph, enode);
                // TODO(@gussmith23) DO NOT SUBMIT this is a hack b/c reshapes
                // cause a lot of trouble w/ shapes settling
                new_shape.access_pattern_shape_settled = false;

                MyAnalysisData::AccessPattern(new_shape)
            }
            &AccessFlatten(access_id) => {
                let a = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!(),
                };
                MyAnalysisData::AccessPattern(AccessPatternData {
                    // TODO(@gussmith23) Implement zero regions
                    // It's harmless (I think) if `zero_regions` defaults to
                    // empty, but for it to be useful, we need to implement it
                    // for each operator.
                    zero_regions: {
                        if !a.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        HashMap::default()
                    },
                    shape: IxDyn(&{
                        // This helps us avoid magically inventing a dimension, e.g.
                        // flatten ((), (256, 2)) should not become ((1), (512)).
                        if a.shape.ndim() == 0 {
                            vec![]
                        } else {
                            let flattened_shape = a.shape.as_array_view().iter().product();
                            vec![flattened_shape]
                        }
                    }),
                    item_shape: IxDyn(&{
                        // This helps us avoid magically inventing a dimension, e.g.
                        // flatten ((256, 2), ()) should not become ((512), (1)).
                        if a.item_shape.ndim() == 0 {
                            vec![]
                        } else {
                            let flattened_item_shape =
                                a.item_shape.as_array_view().iter().product();
                            vec![flattened_item_shape]
                        }
                    }),
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: a.contains_accelerator_calls,
                })
            }
            ComputeType(t) => MyAnalysisData::ComputeType(t.clone()),
            &Compute([compute_type_id, access_id]) => {
                let compute_type = match &egraph[compute_type_id].data {
                    MyAnalysisData::ComputeType(t) => t,
                    _ => panic!("Argument 0 of {:?} should be a ComputeType", enode),
                };
                let a0 = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a0) => a0,
                    _ => panic!(),
                };
                // TODO(@gussmith23) Implement zero_regions
                if !a0.zero_regions.is_empty() {
                    debug!(
                        "Throwing away zero region analysis data on line {}",
                        std::line!()
                    );
                }

                match compute_type {
                    self::ComputeType::ReduceMean => {
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            // TODO(@gussmith23) Implement zero regions
                            // It's harmless (I think) if `zero_regions` defaults to
                            // empty, but for it to be useful, we need to implement it
                            // for each operator.
                            zero_regions: {
                                if !a0.zero_regions.is_empty() {
                                    debug!(
                                        "Throwing away zero region analysis data on line {}",
                                        std::line!()
                                    );
                                }
                                HashMap::default()
                            },
                            shape: a0.shape.clone(),
                            item_shape: ndarray::IxDyn(&[]),
                            access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                            contains_accelerator_calls: a0.contains_accelerator_calls,
                        })
                    }
                    self::ComputeType::Softmax => {
                        assert_eq!(
                            a0.item_shape.ndim(),
                            1,
                            "Softmax is only implemented for axis=-1"
                        );
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            // TODO(@gussmith23) Implement zero regions
                            // It's harmless (I think) if `zero_regions` defaults to
                            // empty, but for it to be useful, we need to implement it
                            // for each operator.
                            zero_regions: {
                                if !a0.zero_regions.is_empty() {
                                    debug!(
                                        "Throwing away zero region analysis data on line {}",
                                        std::line!()
                                    );
                                }
                                HashMap::default()
                            },
                            shape: a0.shape.clone(),
                            item_shape: a0.item_shape.clone(),
                            access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                            contains_accelerator_calls: a0.contains_accelerator_calls,
                        })
                    }
                    self::ComputeType::ElementwiseAdd
                    | self::ComputeType::ElementwiseMul
                    | self::ComputeType::ElementwiseDiv => {
                        assert!(a0.item_shape.ndim() >= 1);
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            // TODO(@gussmith23) Implement zero regions
                            // It's harmless (I think) if `zero_regions` defaults to
                            // empty, but for it to be useful, we need to implement it
                            // for each operator.
                            zero_regions: {
                                if !a0.zero_regions.is_empty() {
                                    debug!(
                                        "Throwing away zero region analysis data on line {}",
                                        std::line!()
                                    );
                                }
                                HashMap::default()
                            },
                            shape: a0.shape.clone(),
                            item_shape: IxDyn(&a0.item_shape.slice()[1..]),
                            access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                            contains_accelerator_calls: a0.contains_accelerator_calls,
                        })
                    }
                    self::ComputeType::DotProduct => {
                        // If it's =1, that's just a "dot product" of scalars,
                        // which is just a sum.
                        //
                        // Honestly, it could also be 0. It doesn't make much
                        // sense but it's not wrong. Can remove this later if we
                        // want those semantics.
                        assert!(a0.item_shape.ndim() >= 1);

                        // MyAnalysisData::Tensor(TensorData {
                        //     shape: a0.shape.clone(),
                        // })
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            // TODO(@gussmith23) Implement zero regions
                            // It's harmless (I think) if `zero_regions` defaults to
                            // empty, but for it to be useful, we need to implement it
                            // for each operator.
                            zero_regions: {
                                if !a0.zero_regions.is_empty() {
                                    debug!(
                                        "Throwing away zero region analysis data on line {}",
                                        std::line!()
                                    );
                                }
                                HashMap::default()
                            },
                            shape: a0.shape.clone(),
                            item_shape: IxDyn(&[]),
                            access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                            contains_accelerator_calls: a0.contains_accelerator_calls,
                        })
                    }
                    self::ComputeType::ReduceSum | self::ComputeType::ReduceMax => {
                        MyAnalysisData::AccessPattern(AccessPatternData {
                            // TODO(@gussmith23) Implement zero regions
                            // It's harmless (I think) if `zero_regions` defaults to
                            // empty, but for it to be useful, we need to implement it
                            // for each operator.
                            zero_regions: {
                                if !a0.zero_regions.is_empty() {
                                    debug!(
                                        "Throwing away zero region analysis data on line {}",
                                        std::line!()
                                    );
                                }
                                HashMap::default()
                            },
                            shape: a0.shape.clone(),
                            item_shape: IxDyn(&[]),
                            access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                            contains_accelerator_calls: a0.contains_accelerator_calls,
                        })
                    }
                    self::ComputeType::ReLU
                    | self::ComputeType::Sqrt
                    | self::ComputeType::Negative => {
                        // TODO(@gussmith23) Implement zero_regions
                        if !a0.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        let mut a = a0.clone();
                        a.zero_regions = HashMap::default();

                        a.access_pattern_shape_settled = all_children_are_settled(egraph, enode);
                        MyAnalysisData::AccessPattern(a)
                    }
                }
            }
            &AccessCartesianProduct([a0_id, a1_id]) => {
                let (a0, a1) = match (&egraph[a0_id].data, &egraph[a1_id].data) {
                    (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => {
                        (a0, a1)
                    }
                    _ => panic!(),
                };
                assert_eq!(
                    a0.item_shape, a1.item_shape,
                    "Cartesian product argument shapes must match"
                );

                let new_shape = IxDyn(
                    a0.shape
                        .as_array_view()
                        .iter()
                        .cloned()
                        .chain(a1.shape.as_array_view().iter().cloned())
                        .collect::<Vec<usize>>()
                        .as_slice(),
                );
                let new_item_shape = IxDyn(
                    std::iter::once(2)
                        .chain(a0.item_shape.as_array_view().iter().cloned())
                        .collect::<Vec<usize>>()
                        .as_slice(),
                );

                assert_eq!(
                    new_shape.as_array_view().iter().product::<usize>()
                        * new_item_shape.as_array_view().iter().product::<usize>(),
                    a0.shape.as_array_view().iter().product::<usize>()
                        * a1.shape.as_array_view().iter().product::<usize>()
                        * 2
                        * a0.item_shape.as_array_view().iter().product::<usize>()
                );

                MyAnalysisData::AccessPattern(AccessPatternData {
                    zero_regions: {
                        // TODO(@gussmith23) We only implement zero regions for
                        // item dimensions.
                        // That's all we need for now w/r/t cart prods.

                        let mut zero_regions = HashMap::new();
                        for item_dim in 0..a0.item_shape.ndim() {
                            if let (Some(range_set_0), Some(range_set_1)) = (
                                a0.zero_regions.get(&(a0.shape.ndim() + item_dim)),
                                a1.zero_regions.get(&(a1.shape.ndim() + item_dim)),
                            ) {
                                // Basically, we know a range [:, :, :, :, x] is
                                // filled with zeros if its original ranges [:,
                                // :, x] and [:, :, x] are zeros.
                                let new_range_set: BoolVecRangeSet = range_set_0
                                    .iter()
                                    .zip(range_set_1.iter())
                                    .map(|(v0, v1): (&bool, &bool)| *v0 && *v1)
                                    .collect();
                                if new_range_set.iter().any(|v| *v) {
                                    zero_regions.insert(
                                        a0.shape.ndim() + a1.shape.ndim() + 1 + item_dim,
                                        new_range_set,
                                    );
                                }
                            }
                        }

                        zero_regions
                    },
                    shape: new_shape.clone(),
                    item_shape: new_item_shape.clone(),
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: a0.contains_accelerator_calls
                        || a1.contains_accelerator_calls,
                })
            }
            &SliceShape([shape_id, dim_id]) => {
                let shape = match &egraph[shape_id].data {
                    MyAnalysisData::Shape(s) => &s.shape,
                    _ => panic!(),
                };
                let dim = MyAnalysis::get_usize(dim_id, egraph);
                MyAnalysisData::Shape(ShapeData {
                    shape: IxDyn(shape.as_array_view().slice(s![dim..]).to_slice().unwrap()),
                    dtype: crate::language::DataType::Uint(64),
                })
            }
            &ShapeInsertAxis([shape_id, dim_id]) => {
                let shape = MyAnalysis::get_shape_of_value(shape_id, egraph);
                let dim = MyAnalysis::get_usize(dim_id, egraph);
                assert!(
                    dim <= shape.ndim(),
                    "Invalid dimension {} for shape {:?}",
                    dim,
                    shape
                );
                MyAnalysisData::Shape(ShapeData {
                    shape: IxDyn(
                        shape.slice()[..dim]
                            .iter()
                            .chain(std::iter::once(&1))
                            .chain(shape.slice()[dim..].iter())
                            .cloned()
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                    dtype: crate::language::DataType::Uint(64),
                })
            }
            &ShapeRemoveAxis([shape_id, dim_id]) => {
                let shape = MyAnalysis::get_shape_of_value(shape_id, egraph);
                let dim = MyAnalysis::get_usize(dim_id, egraph);
                assert!(
                    dim < shape.ndim(),
                    "Invalid dimension {} for shape {:?}",
                    dim,
                    shape
                );
                MyAnalysisData::Shape(ShapeData {
                    shape: IxDyn(
                        shape.slice()[..dim]
                            .iter()
                            .chain(shape.slice()[dim + 1..].iter())
                            .cloned()
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                    dtype: crate::language::DataType::Uint(64),
                })
            }
            &DataType(dtype) => MyAnalysisData::DataType(dtype.clone()),
            &Access([tensor_or_access_id, dim_id]) => {
                // TODO(@gussmith23) How to access tensor literals?
                let dim = MyAnalysis::get_usize(dim_id, egraph);
                let access = match &egraph[tensor_or_access_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => panic!(),
                };
                let shape = access
                    .shape
                    .as_array_view()
                    .iter()
                    .chain(access.item_shape.as_array_view().iter())
                    .cloned()
                    .collect::<Vec<_>>();
                MyAnalysisData::AccessPattern(AccessPatternData {
                    // TODO(@gussmith23) Implement zero regions
                    // It's harmless (I think) if `zero_regions` defaults to
                    // empty, but for it to be useful, we need to implement it
                    // for each operator.
                    zero_regions: {
                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        HashMap::default()
                    },
                    shape: IxDyn(&shape[..dim]),
                    item_shape: IxDyn(&shape[dim..]),
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: access.contains_accelerator_calls,
                })
            }
            &SystolicArray([rows_id, cols_id, a0_id, a1_id])
            | &SystolicArrayWithBlocking([rows_id, cols_id, a0_id, a1_id]) => {
                let rows = Self::get_usize(rows_id, egraph);
                let cols = Self::get_usize(cols_id, egraph);

                let (a0, a1) = match (&egraph[a0_id].data, &egraph[a1_id].data) {
                    (MyAnalysisData::AccessPattern(a0), MyAnalysisData::AccessPattern(a1)) => {
                        (a0, a1)
                    }
                    _ => panic!("Expected access patterns as third and fourth arguments"),
                };

                assert_eq!(a1.shape, IxDyn(&[]));
                assert!(a0.shape.ndim() == 0 || a0.shape.ndim() == 1);

                match &enode {
                    &SystolicArray(_) => {
                        assert_eq!(a1.item_shape, IxDyn(&[rows, cols]));
                        assert_eq!(a0.item_shape, IxDyn(&[rows]));
                    }
                    &SystolicArrayWithBlocking(_) => {
                        // Scott: The input vector size should be a multiple of
                        // the systolic array's height and the output vector
                        // size should be a multiple of the systolic array's
                        // width.
                        assert_eq!(a0.item_shape.ndim(), 1);
                        assert!(a0.item_shape.slice()[0] % rows == 0);
                        assert_eq!(a1.item_shape.ndim(), 2);
                        assert_eq!(a0.item_shape.slice()[0], a1.item_shape.slice()[0]);
                        assert!(a1.item_shape.slice()[1] % cols == 0);
                    }
                    _ => unreachable!(),
                }

                MyAnalysisData::AccessPattern(AccessPatternData {
                    // TODO(@gussmith23) Implement zero regions
                    // It's harmless (I think) if `zero_regions` defaults to
                    // empty, but for it to be useful, we need to implement it
                    // for each operator.
                    zero_regions: {
                        if !a0.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        if !a1.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        HashMap::default()
                    },
                    shape: IxDyn(
                        a0.shape
                            .as_array_view()
                            .iter()
                            .chain(std::iter::once(&a1.item_shape.slice()[1]))
                            .cloned()
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                    item_shape: IxDyn(&[]),
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: a0.contains_accelerator_calls
                        || a1.contains_accelerator_calls,
                })
            }
            Num(u) => MyAnalysisData::Num(*u),
            Symbol(name) => {
                MyAnalysisData::Shape(ShapeData {
                    shape: ndarray::IxDyn(
                        &(match &name[..] {
                            "in" => vec![1, 784],
                            "w1" => vec![784, 512],
                            "w2" => vec![512, 512],
                            "w3" => vec![512, 10],
                            // TODO(@gussmith23) have to figure out a way around this.
                            // Max seems to think the tensors should just go
                            // into the egraph. I was hoping to have some kind
                            // of environment that we could wrap the egraph in
                            // (would have to be accessible from here), but Max
                            // doesn't have that nor does he plan to implement
                            // it.
                            //
                            // Update, Max is implementing something that will
                            // allow for this.
                            "single-matrix-multiply-input-a" => vec![32, 32],
                            "single-matrix-multiply-input-b" => vec![32, 32],
                            "v-32" => vec![32],
                            "t-32-32" => vec![32, 32],
                            "t-32-64" => vec![32, 64],
                            "t-64-128" => vec![64, 128],
                            "t-128-16" => vec![128, 16],
                            // A 3-channel "image" in CHW format.
                            "t-3-32-32" => vec![3, 32, 32],
                            // An OIHW set of convolution filters.
                            "t-8-3-3-3" => vec![8, 3, 3, 3],
                            "t-1024-2-256" => vec![1024, 2, 256],
                            "t-1-2-3-4" => vec![1, 2, 3, 4],
                            _ => egraph
                                .analysis
                                .name_to_shape
                                .get(name)
                                .unwrap_or_else(|| panic!("No shape defined for {}", name))
                                .clone(),
                        })[..],
                    ),
                    dtype: egraph
                        .analysis
                        .name_to_dtype
                        .get(name)
                        .unwrap_or_else(|| &crate::language::DataType::Float(32))
                        .clone(),
                })
            }
            PadType(t) => MyAnalysisData::PadType(*t),
            &AccessWindows([access_id, filters_shape_id, stride_shape_id]) => {
                let access = match &egraph[access_id].data {
                    MyAnalysisData::AccessPattern(a) => a,
                    _ => {
                        panic!("Expected an access pattern as the first argument to access-windows")
                    }
                };
                let filters_shape = MyAnalysis::get_shape_of_value(filters_shape_id, egraph);
                let stride_shape = MyAnalysis::get_shape_of_value(stride_shape_id, egraph);

                // TODO(@gussmith23) Generalize AccessWindows to other accesses
                // Right now we expect item shape to be a scalar.
                // I don't think we need this to be true.
                //assert_eq!(access.item_shape.ndim(), 0);

                MyAnalysisData::AccessPattern(AccessPatternData {
                    // TODO(@gussmith23) Implement zero regions
                    // It's harmless (I think) if `zero_regions` defaults to
                    // empty, but for it to be useful, we need to implement it
                    // for each operator.
                    zero_regions: {
                        if !access.zero_regions.is_empty() {
                            debug!(
                                "Throwing away zero region analysis data on line {}",
                                std::line!()
                            );
                        }
                        HashMap::default()
                    },
                    shape: IxDyn(
                        access
                            .shape
                            .slice()
                            .iter()
                            .cloned()
                            .chain(
                                access_windows_resulting_shape(
                                    &access.item_shape,
                                    &filters_shape,
                                    &stride_shape,
                                )
                                .as_slice()
                                .iter()
                                .cloned(),
                            )
                            .collect::<Vec<_>>()
                            .as_slice(),
                    ),
                    item_shape: filters_shape.clone(),
                    access_pattern_shape_settled: all_children_are_settled(egraph, enode),
                    contains_accelerator_calls: access.contains_accelerator_calls,
                })
            }

            &ShapeOf([tensor_id]) => MyAnalysisData::Shape(ShapeData {
                shape: Self::get_shape(tensor_id, egraph).clone(),
                dtype: Self::get_dtype(tensor_id, egraph).clone(),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use egg::RecExpr;

    use super::*;
    #[test]
    fn access_windows() {
        // TODO(@gussmith23) Could probably clean this up with a for loop
        // Would make it easier to add more tests.

        let program = "
         (access-windows (access (access-tensor t-3-32-32) 0) (slice-shape (shape-of t-8-3-3-3) 1) (shape 1 1 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 30, 30]));
                assert_eq!(a.item_shape, IxDyn(&[3, 3, 3]));
            }
            _ => panic!(),
        }

        let program = "
         (access-windows (access (access-tensor t-3-32-32) 0) (slice-shape (shape-of t-8-3-3-3) 1) (shape 1 2 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 15, 30]));
                assert_eq!(a.item_shape, IxDyn(&[3, 3, 3]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn shape_of() {
        let program = "
         (shape-of t-3-32-32)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(
            MyAnalysis::get_shape_of_value(id, &egraph),
            &IxDyn(&[3, 32, 32])
        );
    }

    #[test]
    fn access() {
        let program = "
         (access (access-tensor t-3-32-32) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[3, 32, 32]));
            }
            _ => panic!(),
        }

        let program = "
         (access (access-tensor t-3-32-32) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }

        let program = "
         (access (access-tensor t-3-32-32) 3)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn reaccess() {
        let program = "
         (access (access (access-tensor t-3-32-32) 3) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[3, 32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_invalid() {
        let program = "
         (access (access-tensor t-3-32-32) 4)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn shape_insert_axis_0() {
        let program = "
         (shape-insert-axis (shape 1 2 3) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(
            MyAnalysis::get_shape_of_value(id, &egraph),
            &IxDyn(&[1, 2, 1, 3])
        );
    }

    #[test]
    fn shape_insert_axis_1() {
        let program = "
         (shape-insert-axis (shape 1 2 3) 3)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(
            MyAnalysis::get_shape_of_value(id, &egraph),
            &IxDyn(&[1, 2, 3, 1])
        );
    }

    #[test]
    fn shape_remove_axis_0() {
        let program = "
         (shape-remove-axis (shape 1 2 3) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[2, 3]));
    }

    #[test]
    fn shape_remove_axis_1() {
        let program = "
         (shape-remove-axis (shape 1 2 3) 1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[1, 3]));
    }

    #[test]
    fn shape_remove_axis_2() {
        let program = "
         (shape-remove-axis (shape 1 2 3) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[1, 2]));
    }

    #[test]
    #[should_panic]
    fn shape_remove_axis_panic() {
        let program = "
         (shape-remove-axis (shape 1 2 3) 3)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    #[should_panic]
    fn shape_insert_axis_panic() {
        let program = "
         (shape-insert-axis (shape 1 2 3) 4)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn slice_shape() {
        let program = "
         (slice-shape (shape-of t-3-32-32) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[32]));

        let program = "
         (slice-shape (shape-of t-3-32-32) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(
            MyAnalysis::get_shape_of_value(id, &egraph),
            &IxDyn(&[3, 32, 32])
        );
    }

    #[test]
    #[should_panic]
    fn slice_shape_invalid_slice() {
        let program = "
         (slice-shape (shape-of t-3-32-32) 10)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        assert_eq!(MyAnalysis::get_shape_of_value(id, &egraph), &IxDyn(&[]));
    }

    #[test]
    fn access_cartesian_product() {
        let program = "
         (access-cartesian-product
          (access (access-tensor v-32) 0)
          (access (access-tensor t-32-32) 1)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }

        let program = "
         (access-cartesian-product
          (access (access-tensor t-32-32) 1)
          (access (access-tensor v-32) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    // TODO(@gussmith23) More tests of cart prod w/ padding
    fn access_cartesian_product_zero_padding() {
        let program = "
         (access-cartesian-product
          (access-pad
           (access (access-tensor v-32) 0)
           zero-padding 0 2 3
          )
          (access-pad
           (access (access-tensor t-32-32) 1)
           zero-padding 1 2 3
          )
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 37]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&2].len(), 37);
                assert!(a.zero_regions[&2].covered((0, 2)));
                assert!(!a.zero_regions[&2].covered((2, 34)));
                assert!(a.zero_regions[&2].covered((34, 37)));
                assert_eq!(
                    a.zero_regions[&2],
                    std::iter::repeat(true)
                        .take(2)
                        .chain(std::iter::repeat(false).take(32))
                        .chain(std::iter::repeat(true).take(3))
                        .collect::<Vec<_>>()
                )
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_dot_product() {
        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }

        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }

        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    // This may not panic in the future, if we allow dot products over empty
    // tuples.
    #[should_panic]
    #[test]
    fn compute_dot_product_panic() {
        let program = "
         (compute dot-product (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn conv2d() {
        // The following TVM/Python code will compute the correct sizes of
        // cov2ds.
        //
        // import tvm
        // from tvm import relay
        //
        // mod = relay.module.Module.from_expr(
        //     relay.nn.conv2d(relay.var('x', shape=[1, 3, 32, 32]),
        //                     relay.var('weight', shape=[8, 3, 3, 3])))
        //
        // print(mod)

        let program = "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor t-8-3-3-3) 1)
           (access-squeeze
            (access-windows
             (access (access-tensor t-3-32-32) 0)
             (shape 3 3 3)
             (shape 1 1 1)
            )
            0
           )
          )
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[8, 30, 30]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }

        let program = "
         (compute dot-product
          (access-cartesian-product
           (access (access-tensor t-8-3-3-3) 1)
           (access-squeeze
            (access-windows
             (access (access-tensor t-3-32-32) 0)
             (shape 3 3 3)
             (shape 1 1 2)
            )
            0
           )
          )
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[8, 30, 15]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn flatten_reshape() {
        let program = "
         (access-reshape
          (access-flatten (access (access-tensor t-3-32-32) 2))
          (access-shape (shape 32 3) (shape 16 2))
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 2]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn flatten_reshape_panic() {
        let program = "
         (access-reshape
          (access-flatten (access (access-tensor t-3-32-32) 2))
          (access-shape (shape 1) (shape 16 2))
         )
        "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);

        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 2]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_0() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 1) 0 0 1)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_1() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 1) 1 16 32)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_2() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 2) 2 16 32)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[16]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_zero_pad_0() {
        test_logger::ensure_env_logger_initialized();

        let program = "(access-slice (access-pad (access (access-tensor t-3-32-32) 1) zero-padding 0 2 3) 0 0 3)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&0].len(), 3);
                assert_eq!(a.zero_regions[&0], vec![true, true, false]);
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_slice_zero_pad_1() {
        test_logger::ensure_env_logger_initialized();

        let program = "
(access-slice
 (access-pad
  (access (access-tensor t-3-32-32) 1)
  zero-padding 0 2 3
 )
 0 1 7
)"
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[6]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&0].len(), 6);
                assert_eq!(
                    a.zero_regions[&0],
                    vec![true, false, false, false, true, true]
                );
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_slice_panic() {
        let program = "(access-slice (access (access-tensor t-3-32-32) 1) 3 16 32)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[16, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_concatenate_0() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 0)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[6]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_concatenate_1() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 2)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 64]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_concatenate_panic_0() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-3-32-32) 1) 3)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 64]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_concatenate_panic_1() {
        let program = "(access-concatenate (access (access-tensor t-3-32-32) 1) (access (access-tensor t-8-3-3-3) 1) 2)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 64]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_0() {
        let program = "(access-transpose (access (access-tensor t-3-32-32) 1) (list 1 2 0))"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[32, 3]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_4() {
        let program = "(access-transpose (access (access-tensor t-3-32-32) 1) (list 1 0 2))"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[3, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_5() {
        let program = "(access-transpose (access (access-tensor t-3-32-32) 1) (list 0 1 2))"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_transpose_panic_2() {
        let program = "(access-transpose (access (access-tensor t-3-32-32) 1) (list 0 1 3))"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_move_axis_panic_1() {
        let program = "(access-move-axis (access (access-tensor t-3-32-32) 1) 1 3)"
            .parse()
            .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_sum_0() {
        let program = "
         (compute reduce-sum (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_sum_1() {
        let program = "
         (compute reduce-sum (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pair() {
        let program = "
         (access-pair (access (access-tensor t-32-32) 1) (access (access-tensor t-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn access_pair_panic() {
        let program = "
         (access-pair (access (access-tensor t-32-32) 0) (access (access-tensor t-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[2, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_shift_right_0() {
        let program = "
         (access-shift-right (access (access-tensor t-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_shift_right_1() {
        let program = "
         (access-shift-right (access (access-tensor t-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_shift_right_2() {
        let program = "
         (access-shift-right (access (access-tensor t-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_relu() {
        let program = "
         (compute relu (access (access-tensor t-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_add_0() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_add_1() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_add_2() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn compute_elementwise_add_panic() {
        let program = "
         (compute elementwise-add (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn compute_elementwise_mul_0() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_mul_1() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 1))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_mul_2() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[should_panic]
    #[test]
    fn compute_elementwise_mul_panic() {
        let program = "
         (compute elementwise-mul (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn zero_padding() {
        let program = "zero-padding".parse().unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::PadType(PadType::ZeroPadding) => (),
            _ => panic!(),
        };
    }

    #[test]
    fn access_pad_zero_padding_0() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) zero-padding 0 1 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[35]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&0].len(), 35);
                assert!(a.zero_regions[&0].covered((0, 1)));
                assert!(!a.zero_regions[&0].covered((1, 33)));
                assert!(a.zero_regions[&0].covered((33, 35)));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pad_zero_padding_1() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) zero-padding 1 0 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32]));
                assert_eq!(a.item_shape, IxDyn(&[34]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&1].len(), 34);
                assert!(a.zero_regions[&1].covered((32, 34)));
                assert!(!a.zero_regions[&1].covered((0, 32)));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pad_zero_padding_2() {
        let program = "
(access-pad
 (access-pad (access (access-tensor t-32-32) 1) zero-padding 1 0 2)
 zero-padding 0 1 3
)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[36]));
                assert_eq!(a.item_shape, IxDyn(&[34]));
                assert_eq!(a.zero_regions.len(), 2);
                assert_eq!(a.zero_regions[&1].len(), 34);
                assert_eq!(a.zero_regions[&0].len(), 36);
                assert!(a.zero_regions[&1].covered((32, 34)));
                assert!(!a.zero_regions[&1].covered((0, 32)));
                assert!(a.zero_regions[&0].covered((0, 1)));
                assert!(a.zero_regions[&0].covered((33, 36)));
                assert!(!a.zero_regions[&0].covered((1, 33)));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pad_zero_padding_3() {
        let program = "
(access-pad
 (access-pad (access (access-tensor t-32-32) 1) zero-padding 0 0 2)
 zero-padding 0 1 3
)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[38]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&0].len(), 38);
                assert!(a.zero_regions[&0].covered((0, 1)));
                assert!(a.zero_regions[&0].covered((35, 38)));
                assert!(!a.zero_regions[&0].covered((1, 35)));
                // This one is key: this makes sure that the first pad's zero
                // region was shifted appropriately by the second pad (was (32,
                // 34), but should get shifted by 1)
                assert!(a.zero_regions[&0].covered((33, 35)));
                assert!(!a.zero_regions[&0].covered((0, 33)));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_pad_zero_padding_panic() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) zero-padding 2 3 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn compute_reduce_max_0() {
        let program = "
         (compute reduce-max (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_max_1() {
        let program = "
         (compute reduce-max (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_squeeze() {
        let program = "
         (access-squeeze (access (access-tensor t-1-2-3-4) 1) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[2, 3, 4]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_squeeze_panic() {
        let program = "
         (access-squeeze (access (access-tensor t-1-2-3-4) 1) 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn access_insert_axis() {
        let program = "
         (access-insert-axis (access (access-tensor t-1-2-3-4) 1) 0)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1]));
                assert_eq!(a.item_shape, IxDyn(&[2, 3, 4]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    // TODO(@gussmith) More access-insert-axis tests
    fn access_insert_axis_panic() {
        let program = "
         (access-squeeze (access (access-tensor t-1-2-3-4) 1) 5)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    // TODO(@gussmith) More access-broadcast tests
    fn access_broadcast() {
        let program = "
         (access-broadcast (access (access-tensor t-1-2-3-4) 1) (access-shape (shape 2 2 3 4) (shape)))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: HashMap::default(),
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[2]));
                assert_eq!(a.item_shape, IxDyn(&[2, 3, 4]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        let program = "
         (systolic-array 64 32
          (access (access-tensor a) 1)
          (access (access-transpose (access-tensor a) (list 1 0)) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn systolic_array_panic_0() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        // Because the second argument is not the right shape.
        let program = "
         (systolic-array 64 32
          (access (access-tensor a) 1)
          (access (access-tensor a) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        egraph.add_expr(&program);
    }

    #[test]
    fn list() {
        let program = "
         (list 1 2 3 4)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::List(l) => assert_eq!(l, &vec![1, 2, 3, 4]),
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose() {
        let program = "
         (access-transpose (access (access-tensor a) 1) (list 2 0 1))
         "
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        let name_to_dtype = [("a".into(), DataType::Float(32))]
            .iter()
            .cloned()
            .collect();
        map.insert("a".to_string(), vec![4, 5, 6]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype,
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[6]));
                assert_eq!(a.item_shape, IxDyn(&[4, 5]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_1() {
        let program = "
             (access-transpose
              (access-transpose
               (access (access-tensor t) 1)
               (list 1 3 2 0)
              )
              (list 3 2 1 0)
             )"
        .parse()
        .unwrap();
        let mut map = HashMap::new();
        let name_to_dtype = [("t".into(), DataType::Float(32))]
            .iter()
            .cloned()
            .collect();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype,
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1]));
                assert_eq!(a.item_shape, IxDyn(&[3, 4, 2]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_2() {
        let program = "
              (access-transpose
               (access (access-tensor t) 1)
               (list 1 3 2 0)
              )
             "
        .parse()
        .unwrap();
        let mut map = HashMap::new();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[2]));
                assert_eq!(a.item_shape, IxDyn(&[4, 3, 1]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_transpose_3() {
        let program = "
              (access-transpose
               (access-pad (access (access-tensor t) 1) zero-padding 1 5 0)
               (list 1 3 2 0)
              )
             "
        .parse()
        .unwrap();
        let mut map = HashMap::new();
        map.insert("t".to_string(), vec![1, 2, 3, 4]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[7]));
                assert_eq!(a.item_shape, IxDyn(&[4, 3, 1]));
                assert_eq!(a.zero_regions.len(), 1);
                assert_eq!(a.zero_regions[&0].len(), 7);
                assert!(a.zero_regions[&0].covered((0, 5)));
                assert!(!a.zero_regions[&0].covered((5, 7)));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic]
    fn access_transpose_panic_0() {
        let program = "
         (access-transpose (access (access-tensor a) 1) (list 0 1))
         "
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![4, 5, 6]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        egraph.add_expr(&program);
    }

    #[test]
    #[should_panic]
    fn access_transpose_panic_1() {
        let program = "
         (access-transpose (access (access-tensor a) 1) (list 2 1 1))
         "
        .parse()
        .unwrap();
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![4, 6]);
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        egraph.add_expr(&program);
    }

    #[test]
    #[should_panic]
    fn compute_softmax_0() {
        let program = "
         (compute softmax (access (access-tensor t-3-32-32) 3))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    #[should_panic]
    fn compute_softmax_1() {
        let program = "
         (compute softmax (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        egraph.add_expr(&program);
    }

    #[test]
    fn compute_softmax_2() {
        let program = "
         (compute softmax (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_mean_0() {
        let program = "
         (compute reduce-mean (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_reduce_mean_1() {
        let program = "
         (compute reduce-mean (access (access-tensor t-3-32-32) 2))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_pad_min_padding() {
        let program = "
         (access-pad (access (access-tensor t-32-32) 1) min-padding 0 1 2)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[35]));
                assert_eq!(a.item_shape, IxDyn(&[32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_elementwise_div() {
        let program = "
         (compute elementwise-div (access (access-tensor t-3-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn literal_0() {
        let program = "
         (literal 0.1234)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::Literal(t) => {
                assert_eq!(*t, ndarray::arr0(0.1234).into_dyn());
            }
            _ => panic!(),
        }
    }

    #[test]
    fn access_literal() {
        let program = "
         (access-literal (literal 0.1234))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_sqrt() {
        let program = "
         (compute sqrt (access (access-tensor t-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn compute_negative() {
        let program = "
         (compute negative (access (access-tensor t-32-32) 0))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis::default());
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[]));
                assert_eq!(a.item_shape, IxDyn(&[32, 32]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_with_blocking_0() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![64, 32]);
        let program = "
         (systolic-array-with-blocking 64 32
          (access (access-tensor a) 1)
          (access (access-tensor b) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_with_blocking_2() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![64, 32]);
        let program = "
         (systolic-array-with-blocking 32 32
          (access (access-tensor a) 1)
          (access (access-tensor b) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_with_blocking_3() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![64, 32]);
        let program = "
         (systolic-array-with-blocking 32 2
          (access (access-tensor a) 1)
          (access (access-tensor b) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "assertion failed: a1.item_shape.slice()[1] % cols == 0")]
    fn systolic_array_with_blocking_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![64, 32]);
        let program = "
         (systolic-array-with-blocking 32 3
          (access (access-tensor a) 1)
          (access (access-tensor b) 0)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let _id = egraph.add_expr(&program);
    }

    #[test]
    fn batch_norm_inference() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        // Just showing that, right now, these shapes don't matter at all.
        map.insert("b".to_string(), vec![32]);
        map.insert("c".to_string(), vec![10, 20]);
        map.insert("d".to_string(), vec![3, 3, 3, 3, 3, 3]);
        map.insert("e".to_string(), vec![]);
        let program = "
         (relay-operator-call relay-batch-norm-inference
          (access-tensor a)
          (access-tensor b)
          (access-tensor c)
          (access-tensor d)
          (access-tensor e)
          1 1e-5)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn batch_norm_inference_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        // Just showing that, right now, these shapes don't matter at all.
        map.insert("b".to_string(), vec![32]);
        map.insert("c".to_string(), vec![10, 20]);
        map.insert("d".to_string(), vec![3, 3, 3, 3, 3, 3]);
        map.insert("e".to_string(), vec![]);
        let program = "
         (relay-operator-call relay-batch-norm-inference
          (access-tensor a)
          (access-tensor b)
          (access-tensor c)
          (access-tensor d)
          (access-tensor e)
          1e-5 1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_softmax() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        let program = "
         (relay-operator-call relay-softmax
          (access-tensor a)
          1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_softmax_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![32]);
        let program = "
         (relay-operator-call relay-softmax
          (access-tensor a)
          (access-tensor b)
          )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_relu() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        let program = "
         (relay-operator-call relay-relu
          (access-tensor a))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_relu_panic() {
        let program = "
         (relay-operator-call relay-relu 1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: HashMap::default(),
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_max_pool2d_nchw() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-max-pool2d
          (access-tensor a)
          (shape 1 2)
          (shape 3 4)
          (shape 5 6 7 8)
          relay-activation-layout-nchw
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 3, 15, 20]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_max_pool2d_nhwc() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-max-pool2d
          (access-tensor a)
          (shape 1 2)
          (shape 3 4)
          (shape 5 6 7 8)
          relay-activation-layout-nhwc
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 5, 12, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_max_pool2d_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-max-pool2d
          (access-tensor a)
          (access-tensor a)
          (shape 3 4)
          (shape 5 6 7 8)
          relay-activation-layout-nhwc
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_global_avg_pool2d_nchw() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-global-avg-pool2d
          (access-tensor a)
          relay-activation-layout-nchw
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 3, 1, 1]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_global_avg_pool2d_nhwc() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-global-avg-pool2d
          (access-tensor a)
          relay-activation-layout-nhwc
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1, 1, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_global_avg_pool2d_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-max-pool2d
          (access-tensor a)
          (access-tensor a)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_batch_flatten() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-batch-flatten
          (access-tensor a)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 3 * 32 * 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_batch_flatten_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 3, 32, 64]);
        let program = "
         (relay-operator-call relay-batch-flatten
          (access-tensor a)
          (access-tensor a)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_bias_add() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 32]);
        map.insert("b".to_string(), vec![32]);
        let program = "
         (relay-operator-call relay-bias-add
          (access-tensor a)
          (access-tensor b)
          1
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_bias_add_panic() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 32]);
        let program = "
         (relay-operator-call relay-bias-add
          (access-tensor a)
          2 3
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_add() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![3, 1, 32, 32]);
        map.insert("b".to_string(), vec![64, 1, 32]);
        let program = "
         (relay-operator-call relay-add
          (access-tensor a)
          (access-tensor b)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 64, 32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Shapes can't be broadcast")]
    fn relay_operator_call_add_panic_broadcast() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![3, 2, 32, 32]);
        map.insert("b".to_string(), vec![64, 1, 32]);
        let program = "
         (relay-operator-call relay-add
          (access-tensor a)
          (access-tensor b)
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 64, 32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    #[should_panic(expected = "Parameters do not type check")]
    fn relay_operator_call_add_panic_types() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![3, 2, 32, 32]);
        let program = "
         (relay-operator-call relay-add
          (access-tensor a)
          1
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[3, 64, 32, 32]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_leaky_relu() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        let program = "
         (relay-operator-call relay-leaky-relu
          (access-tensor a) 0.1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[32, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_avgpool2d() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 1280, 7, 7]);
        let program = "
         (relay-operator-call relay-avg-pool2d
          (access-tensor a)
          (shape 7 7)
          (shape 1 1)
          (shape 0 0 0 0)
          relay-activation-layout-nchw)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1280, 1, 1]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_upsampling() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 1280, 7, 7]);
        let program = "
         (relay-operator-call relay-upsampling
          (access-tensor a)
          2.0 2.0
          (relay-activation-layout-nchw))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1280, 14, 14]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_sigmoid() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 1280, 7, 7]);
        let program = "
         (relay-operator-call relay-sigmoid
          (access-tensor a))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1280, 7, 7]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_split() {
        let names_to_shapes = [("data".into(), vec![1, 5, 4])]
            .iter()
            .cloned()
            .collect::<HashMap<_, _>>();
        let mut program = egg::RecExpr::default();
        let operator_id = program.add(Language::RelayOperator(RelayOperator::RelaySplit));
        let tensor_id = program.add(Language::Symbol("data".into()));
        let access_data = program.add(Language::AccessTensor(tensor_id));
        let sections = program.add(Language::Num(5));
        let axis = program.add(Language::Num(1));
        let _relay_operator_call = program.add(Language::RelayOperatorCall(
            vec![operator_id, access_data, sections, axis].into_boxed_slice(),
        ));
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: names_to_shapes,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        let section_data = MyAnalysisData::AccessPattern(AccessPatternData {
            shape: IxDyn(&[]),
            item_shape: IxDyn(&[1, 1, 4]),
            access_pattern_shape_settled: false,
            zero_regions: HashMap::default(),
            contains_accelerator_calls: false,
        });
        match &egraph[id].data {
            MyAnalysisData::Tuple(tup) => tup
                .iter()
                .zip(vec![section_data.clone(), section_data.clone(), section_data].iter())
                .for_each(|t| assert_eq!(t.0, t.1)),
            _ => panic!("Split should outputs a tuple"),
        }
    }

    #[test]
    fn relay_operator_call_maximum() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 1280, 7, 7]);
        map.insert("b".to_string(), vec![1, 1280, 1, 1]);
        let program = "
         (relay-operator-call relay-maximum
          (access-tensor a) (access-tensor b))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1280, 7, 7]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_operator_call_minimum() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![1, 1280, 7, 7]);
        map.insert("b".to_string(), vec![1, 1280, 1, 1]);
        let program = "
         (relay-operator-call relay-minimum
          (access-tensor a) (access-tensor b))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 1280, 7, 7]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_conv2d_nchw_oihw_with_blocking() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![1, 32, 44, 78]);
        map.insert("weights".to_string(), vec![64, 32, 1, 2]);
        let program = "
         (systolic-array-conv2d-nchw-oihw-with-blocking
          32 32
          (access-tensor weights)
          (access-tensor data)
          1 2
          3 4
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 64, 15, 20]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_conv2d_nhwc_hwio_with_blocking() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![1, 44, 78, 32]);
        map.insert("weights".to_string(), vec![1, 2, 32, 64]);
        let program = "
         (systolic-array-conv2d-nhwc-hwio-with-blocking
          32 32
          (access-tensor weights)
          (access-tensor data)
          1 2
          3 4
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 15, 20, 64]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_conv2d_im2col_nhwc_hwio_with_blocking() {
        let mut map = HashMap::default();
        // Note that we don't need any multiples of rows/cols here.
        // Systolic array should handle tail padding.
        map.insert("data".to_string(), vec![1, 44, 78, 33]);
        map.insert("weights".to_string(), vec![1, 2, 33, 65]);
        let program = "
         (systolic-array-conv2d-im2col-nhwc-hwio-with-blocking
          32 32
          (access-tensor weights)
          (access-tensor data)
          1 2
          3 4
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 15, 20, 65]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn systolic_array_conv2d_im2col_nchw_oihw_with_blocking() {
        let mut map = HashMap::default();
        // Note that we don't need any multiples of rows/cols here.
        // Systolic array should handle tail padding.
        map.insert("data".to_string(), vec![1, 33, 44, 78]);
        map.insert("weights".to_string(), vec![65, 33, 1, 2]);
        let program = "
         (systolic-array-conv2d-im2col-nchw-oihw-with-blocking
          32 32
          (access-tensor weights)
          (access-tensor data)
          1 2
          3 4
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape, IxDyn(&[1, 65, 15, 20]));
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn construct_tuple() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![16, 32]);
        let program = "
         (construct-tuple (access-tensor a) (access-tensor b))
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::Tuple(a) => {
                assert_eq!(a.len(), 2);
                match (a[0].clone(), a[1].clone()) {
                    (MyAnalysisData::AccessPattern(t1), MyAnalysisData::AccessPattern(t2)) => {
                        assert_eq!(t1.shape, IxDyn(&[32, 64]));
                        assert_eq!(t2.shape, IxDyn(&[16, 32]));
                    }
                    _ => panic!(),
                }
            }
            _ => panic!(),
        }
    }

    #[test]
    fn tuple_get_item() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![32, 64]);
        map.insert("b".to_string(), vec![16, 32]);
        let program = "
         (tuple-get-item (construct-tuple (access-tensor a) (access-tensor b)) 1)
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(b) => {
                assert_eq!(b.shape, IxDyn(&[16, 32]));
            }
            _ => panic!(),
        }
    }

    // >>> data = relay.var('data', shape=(2, 3, 32, 32))
    // >>> weights = relay.var('weights', shape=(3, 1, 5, 5))
    // >>> program = relay.nn.conv2d(data, weights, strides=(2, 3), padding=(1, 2, 3, 4), groups=3, channels=3)
    // >>> mod = tvm.IRModule.from_expr(program)
    // >>> program = relay.nn.conv2d(data, weights, strides=(2, 3), padding=(1, 2, 3, 4), groups=3, channels=3)
    // >>> mod = relay.transform.InferType()(mod)
    // >>> mod
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(2, 3, 32, 32), float32], %weights: Tensor[(3, 1, 5, 5), float32]) -> Tensor[(2, 3, 16, 12), float32] {
    //   nn.conv2d(%data, %weights, strides=[2, 3], padding=[1, 2, 3, 4], groups=3, channels=3) /* ty=Tensor[(2, 3, 16, 12), float32] */
    // }
    #[test]
    fn conv2d_depthwise_0() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![2, 3, 32, 32]);
        map.insert("weights".to_string(), vec![3, 1, 5, 5]);

        let program = "
         (relay-operator-call relay-conv2d
            (access-tensor data)
            (access-tensor weights)
            (shape 2 3)
            (shape 1 2 3 4)
            3
            3
            (shape 3 5 5)
            relay-activation-layout-nchw
            relay-kernel-layout-oihw
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(b) => {
                assert_eq!(b.shape, IxDyn(&[2, 3, 16, 12]));
            }
            _ => panic!(),
        }
    }
    // >>> import tvm
    // >>> from tvm import relay
    // >>> data = relay.var('data', shape=(2, 3, 32, 32))
    // >>> weights = relay.var('weights', shape=(3, 1, 5, 5))
    // >>> program = relay.nn.conv2d(data, weights, strides=(4, 1), padding=(0, 2, 1, 5), groups=3, channels=3)
    // >>> mod = tvm.IRModule.from_expr(program)
    // >>> mod = relay.transform.InferType()(mod)
    // >>> mod
    // #[version = "0.0.5"]
    // def @main(%data: Tensor[(2, 3, 32, 32), float32], %weights: Tensor[(3, 1, 5, 5), float32]) -> Tensor[(2, 3, 8, 35), float32] {
    //   nn.conv2d(%data, %weights, strides=[4, 1], padding=[0, 2, 1, 5], groups=3, channels=3) /* ty=Tensor[(2, 3, 8, 35), float32] */
    // }
    #[test]
    fn conv2d_depthwise_1() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![2, 3, 32, 32]);
        map.insert("weights".to_string(), vec![3, 1, 5, 5]);

        let program = "
         (relay-operator-call relay-conv2d
            (access-tensor data)
            (access-tensor weights)
            (shape 4 1)
            (shape 0 2 1 5)
            3
            3
            (shape 3 5 5)
            relay-activation-layout-nchw
            relay-kernel-layout-oihw
         )
         "
        .parse()
        .unwrap();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&program);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(b) => {
                assert_eq!(b.shape, IxDyn(&[2, 3, 8, 35]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn test_relay_cast() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![2, 3, 32, 32]);
        let dtypes = [("data".into(), crate::language::DataType::Int(32))]
            .iter()
            .cloned()
            .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-cast data float32)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::Shape(shape) => {
                assert_eq!(shape.dtype, crate::language::DataType::Float(32))
            }
            _ => panic!("Not a valid cast"),
        }
    }
    #[test]
    fn relay_take_0() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![5, 6, 7]);
        map.insert("indices".to_string(), vec![2, 3]);
        let dtypes = [
            ("data".into(), crate::language::DataType::Float(32)),
            ("indices".into(), crate::language::DataType::Int(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-take (access-tensor data) (access-tensor indices) 0)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[2, 3, 6, 7]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    /// def @main(%data: Tensor[(5, 6, 7), float32], %indices: Tensor[(2, 3), int32]) -> Tensor[(5, 6, 2, 3), float32] {
    ///   take(%data, %indices, axis=-1) /* ty=Tensor[(5, 6, 2, 3), float32] */
    /// }
    #[test]
    fn relay_take_1() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![5, 6, 7]);
        map.insert("indices".to_string(), vec![2, 3]);
        let dtypes = [
            ("data".into(), crate::language::DataType::Float(32)),
            ("indices".into(), crate::language::DataType::Int(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-take (access-tensor data) (access-tensor indices) 1)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 2, 3, 7]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    /// def @main(%data: Tensor[(5, 6, 7), float32], %indices: Tensor[(2, 3), int32]) -> Tensor[(5, 6, 2, 3), float32] {
    ///   take(%data, %indices, axis=2) /* ty=Tensor[(5, 6, 2, 3), float32] */
    /// }
    #[test]
    fn relay_take_2() {
        let mut map = HashMap::default();
        map.insert("data".to_string(), vec![5, 6, 7]);
        map.insert("indices".to_string(), vec![2, 3]);
        let dtypes = [
            ("data".into(), crate::language::DataType::Float(32)),
            ("indices".into(), crate::language::DataType::Int(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-take (access-tensor data) (access-tensor indices) 2)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 6, 2, 3]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_stack_0() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![5, 6, 7]);
        map.insert("b".to_string(), vec![5, 6, 7]);
        map.insert("c".to_string(), vec![5, 6, 7]);
        let dtypes = [
            ("a".into(), crate::language::DataType::Float(32)),
            ("b".into(), crate::language::DataType::Float(32)),
            ("c".into(), crate::language::DataType::Float(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-stack (access-tensor a) (access-tensor b) (access-tensor c) 0)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[3, 5, 6, 7]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn relay_stack_1() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![5, 6, 7]);
        map.insert("b".to_string(), vec![5, 6, 7]);
        map.insert("c".to_string(), vec![5, 6, 7]);
        let dtypes = [
            ("a".into(), crate::language::DataType::Float(32)),
            ("b".into(), crate::language::DataType::Float(32)),
            ("c".into(), crate::language::DataType::Float(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-stack (access-tensor a) (access-tensor b) (access-tensor c) 1)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 3, 6, 7]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }
    #[test]
    fn relay_stack_2() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![5, 6, 7]);
        map.insert("b".to_string(), vec![5, 6, 7]);
        map.insert("c".to_string(), vec![5, 6, 7]);
        let dtypes = [
            ("a".into(), crate::language::DataType::Float(32)),
            ("b".into(), crate::language::DataType::Float(32)),
            ("c".into(), crate::language::DataType::Float(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-stack (access-tensor a) (access-tensor b) (access-tensor c) 2)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 6, 3, 7]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }
    #[test]
    fn relay_stack_3() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![5, 6, 7]);
        map.insert("b".to_string(), vec![5, 6, 7]);
        map.insert("c".to_string(), vec![5, 6, 7]);
        let dtypes = [
            ("a".into(), crate::language::DataType::Float(32)),
            ("b".into(), crate::language::DataType::Float(32)),
            ("c".into(), crate::language::DataType::Float(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-stack (access-tensor a) (access-tensor b) (access-tensor c) 3)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 6, 7, 3]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }
    #[test]
    fn relay_stack_4() {
        let mut map = HashMap::default();
        map.insert("a".to_string(), vec![5, 6, 7]);
        map.insert("b".to_string(), vec![5, 6, 7]);
        map.insert("c".to_string(), vec![5, 6, 7]);
        let dtypes = [
            ("a".into(), crate::language::DataType::Float(32)),
            ("b".into(), crate::language::DataType::Float(32)),
            ("c".into(), crate::language::DataType::Float(32)),
        ]
        .iter()
        .cloned()
        .collect();
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: dtypes,
        });
        let program = "
    (relay-operator-call relay-stack (access-tensor a) (access-tensor b) (access-tensor c) -1)
    ";
        let id = egraph.add_expr(&program.parse().unwrap());
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(a) => {
                assert_eq!(a.shape.slice(), &[5, 6, 7, 3]);
                assert_eq!(a.item_shape, IxDyn(&[]));
            }
            _ => panic!(),
        }
    }

    #[test]
    fn conv3d() {
        let data_shape = vec![1, 3, 32, 2, 34];
        let weights_shape = vec![8, 3, 2, 1, 23];
        let mut map = HashMap::default();
        map.insert("data".to_string(), data_shape.clone());
        map.insert("weights".to_string(), weights_shape.clone());

        let mut expr = RecExpr::default();
        let data_id = expr.add(Language::Symbol("data".into()));
        let data_id = expr.add(Language::AccessTensor(data_id));
        let weights_id = expr.add(Language::Symbol("weights".into()));
        let weights_id = expr.add(Language::AccessTensor(weights_id));
        crate::language::from_relay::conv3d(
            &mut expr,
            data_id,
            &data_shape,
            weights_id,
            &weights_shape,
            &[1, 2, 3],
            &[1, 2, 3, 4, 5, 6],
            &[1, 1, 1],
            1,
            "NCDHW",
            "OIDHW",
            "",
            false,
        );
        let mut egraph = egg::EGraph::<Language, MyAnalysis>::new(MyAnalysis {
            name_to_shape: map,
            name_to_dtype: HashMap::default(),
        });
        let id = egraph.add_expr(&expr);
        match &egraph[id].data {
            MyAnalysisData::AccessPattern(b) => {
                assert_eq!(b.as_vec(), vec![1, 8, 36, 5, 7]);
            }
            _ => panic!(),
        }
    }
}
