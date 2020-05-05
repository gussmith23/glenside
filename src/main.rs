use egg::define_language;
use either::*;
use log::{debug, info};
use ndarray::s;
use ndarray::Dimension;
use num_traits::identities::Zero;
use std::collections::HashMap;

fn main() {
    env_logger::init();
    //dot_product()
    //mlp()
    single_matrix_multiply()
}

egg::define_language! {
    enum MlpLanguage {
        // TODO(gus) do we need this to be an intrinsic? Is this cheating?
        // Without this I'm not sure how to represent a matmul w/o using a lambda.
        // TODO(gus) a more updated todo here would be: do we want to
        // consider (sooner rather than later) whether we can break down
        // dot product into its component parts (that is, whether it would
        // be valuable to, and then, whether it would be possible)
        Dotprod = "dotprod",
        Rows = "rows",
        Cols = "cols",
        CartesianProduct = "cartesian-product",
        // Map over a shaped list
        ShapedMap = "shaped-map",
        BsgSystolicArray = "bsg_systolic_array_weight_stationary",
        // Slice into list/tensor/whatever we're calling them
        Slice = "slice",
        ShapedAdd = "shaped-add",
        Concat = "concat",
        // TODO(gus) this will probably need to be signed at some point?
        Usize(usize),
        Symbol(String),
    }
}

type Environment<'a> = HashMap<&'a str, Value>;

// The type system of our program. We support tensors (which support values,
// as 0-dim tensors) and lists. We could imagine adding other datatypes in
// the future (e.g. trees).
// TODO(gus) how to represent user-defined ADTs?
// StreamValue are the actual value types that can appear in streams.
#[derive(Clone, PartialEq, Debug)]
enum ListValue {
    Scalar(DataType),
    // TODO(gus) Doing this may open up a huge can of worms. I'm not quite
    // sure how to design the language though, and this is the easiest way
    // to get what I want: I want to be able to have Streams of Values.
    //Value(Value),
    //CartesianProductElement(ndarray::ArrayD<DataType>, ndarray::ArrayD<DataType>),
    Value(Value),
}
#[derive(Clone, PartialEq, Debug)]
enum Value {
    List(std::vec::Vec<ListValue>),
    // Box needed for indirection, otherwise we have recursive structure.
    Tuple2(Box<ListValue>, Box<ListValue>),
    ShapedList(ndarray::ArrayD<ListValue>),
    Function(fn(ListValue) -> ListValue),
    // TODO(gus) is this a problem? If we have this, we might as well remove
    // ListValue entirely, right?
    //Scalar(DataType),
    // Ok, maybe that later, but we actually need integers:
    Usize(usize),
}

impl std::default::Default for ListValue {
    fn default() -> Self {
        ListValue::Scalar(DataType::default())
    }
}

impl std::ops::Add for ListValue {
    type Output = ListValue;

    fn add(self: ListValue, other: ListValue) -> ListValue {
        match self {
            ListValue::Scalar(s) => match other {
                ListValue::Scalar(s2) => ListValue::Scalar(s + s2),
                _ => panic!(),
            },
            ListValue::Value(v) => match other {
                ListValue::Value(v2) => ListValue::Value(v + v2),
                _ => panic!(),
            },
        }
    }
}

// impl num_traits::identities::Zero for ListValue {
//     fn zero() {
//         // TODO(gus) This seems bad
//         ListValue::Scalar(DataType::zero())
// }

impl std::ops::Add for Value {
    type Output = Value;

    fn add(self: Value, other: Value) -> Value {
        match self {
            Value::List(v) => match other {
                Value::List(v2) => Value::List({
                    assert_eq!(v.len(), v2.len());
                    v.iter()
                        .zip(v2.iter())
                        .map(|(a, b)| a.clone() + b.clone())
                        .collect()
                }),
                _ => panic!(),
            },
            Value::ShapedList(t) => match other {
                Value::ShapedList(t2) => Value::ShapedList(t + t2),
                _ => panic!(),
            },
            Value::Tuple2(a, b) => match other {
                Value::Tuple2(a2, b2) => Value::Tuple2(Box::new(*a + *a2), Box::new(*b + *b2)),
                _ => panic!(),
            },
            Value::Usize(u) => match other {
                Value::Usize(u2) => Value::Usize(u + u2),
                _ => panic!(),
            },
            Value::Function(_) => panic!(),
        }
    }
}

#[derive(Clone, Debug)]
enum MemoizedInterpreterResult {
    InterpretedValue(Value),
    StillInterpreting,
    // When an eclass cannot be interpreted, because all of its enodes
    // recursively depend on themselves.
    CanNotInterpret,
}

type MemoizationMap = std::collections::HashMap<egg::Id, MemoizedInterpreterResult>;

fn interpret_eclass<M: egg::Metadata<MlpLanguage>>(
    egraph: &egg::EGraph<MlpLanguage, M>,
    eclass: &egg::EClass<MlpLanguage, M>,
    env: &Environment,
    // TODO(gus) shoot, i think i'm mixing up where to put the "mut". Should
    // probably not be on both.
    mut memo_map: &mut MemoizationMap,
) -> MemoizedInterpreterResult {
    info!("Interpreting eclass {}", eclass.id);
    debug!("{:?}", eclass);

    if memo_map.contains_key(&eclass.id) {
        // TODO(gus) is this right?
        return memo_map[&eclass.id].clone();
    }

    // Insert a flag saying that we are now interpreting this eclass. As we
    // recurse, if we hit this eclass again, we'll know we have a recursive
    // cycle, which we handle later in this function.
    match memo_map.insert(eclass.id, MemoizedInterpreterResult::StillInterpreting) {
        Some(_) => panic!(),
        None => (),
    }

    assert!(eclass.nodes.len() > 0);
    let mut results: std::vec::Vec<MemoizedInterpreterResult> = eclass
        .nodes
        .iter()
        .map(|enode: &egg::ENode<MlpLanguage>| interpret_enode(egraph, enode, env, &mut memo_map))
        .collect();
    let pre_filtered_length = results.len();

    // At this point, we'll have a MemoizedInterpreterResult for every enode in
    // the eclass. These all represent potential values for the entire eclass.
    // That is, the whole point of the egraph is that eclasses contain
    // expressions with equivalent values. So, if our interpreter is built
    // correctly, each MemoizedInterpreterResult will be one of two things:
    // 1. an InterpretedValue holding the eclass's correct value, or
    // 2. a StillInterpreting, if the enode recursively depends on the eclass.

    // Now, we check two things:
    // 1. There's at least one InterpretedValue---i.e. at least one of the
    //    enodes resolved to a concrete value and doesn't recursively depend on
    //    its own eclass. Without this, we have no way to even guess the value
    //    of this eclass.
    // 2. All of the InterpretedValues have the same value. This should be true
    //    if the interpreter is built correctly and if the rewrites are correct.

    let filtered_results: std::vec::Vec<Value> = results
        .drain(..)
        .filter(|result| match result {
            MemoizedInterpreterResult::InterpretedValue(_) => true,
            _ => false,
        })
        .map(|result| match result {
            MemoizedInterpreterResult::InterpretedValue(v) => v,
            _ => unreachable!(),
        })
        .collect();
    debug!("{:?}", pre_filtered_length);
    debug!("{:?}", filtered_results.len());
    debug!("{:?}", eclass);
    //assert!(filtered_results.len() > 0, "This eclass's enodes all depend recursively on this eclass! Cannot interpret to a concrete value!");
    if filtered_results.len() == 0 {
        debug!(
            "eclass {} evaluates to CanNotInterpret. Hoping for the best!",
            eclass.id
        );
        match memo_map.insert(eclass.id, MemoizedInterpreterResult::CanNotInterpret) {
            Some(MemoizedInterpreterResult::StillInterpreting) => (),
            _ => panic!(),
        }
        return MemoizedInterpreterResult::CanNotInterpret;
    }
    assert!(
        filtered_results.iter().all(|v| *v == filtered_results[0]),
        "This class's enodes don't all evaluate to the same value!"
    );

    if filtered_results.len() < pre_filtered_length {
        // Some StillInterpreting values were cut out. Might want to log
        // something here later on.
    }

    // Now, if there were any StillInterpreting results, we will "guess" their
    // value to be the concrete value that the rest of the enodes interpreted
    // to. We do this by officially setting the eclass's value in the map.
    assert!(match &memo_map.get(&eclass.id).unwrap() {
        &MemoizedInterpreterResult::StillInterpreting => true,
        _ => panic!(),
    });
    // TODO(gus) cloning, inefficient and lazy
    match memo_map.insert(
        eclass.id,
        MemoizedInterpreterResult::InterpretedValue(filtered_results[0].clone()),
    ) {
        Some(MemoizedInterpreterResult::StillInterpreting) => (),
        _ => panic!(),
    }

    // After we guess, check that things all come out to the same value.
    // We may want to disable this for running-time reasons.
    let check_after_guess = true;
    if check_after_guess {
        // We're essentially just doing the same thing that we did above!
        let results: std::vec::Vec<MemoizedInterpreterResult> = eclass
            .nodes
            .iter()
            .map(|enode: &egg::ENode<MlpLanguage>| {
                interpret_enode(egraph, enode, env, &mut memo_map)
            })
            .collect();

        assert!(results.iter().all(|result| match result {
            MemoizedInterpreterResult::InterpretedValue(v) =>
                v == match memo_map.get(&eclass.id).unwrap() {
                    // TODO(gus) probably inefficient
                    MemoizedInterpreterResult::InterpretedValue(v) => v,
                    _ => panic!(),
                },
            MemoizedInterpreterResult::CanNotInterpret => {
                // TODO(gus) remove this once I figure out if this is expected.
                debug!("CanNotInterpret found while checking");
                true
            }
            MemoizedInterpreterResult::StillInterpreting =>
                panic!("After guessing, we still get a StillInterpreting result!"),
        }));
    }

    // TODO(gus) probably inefficient
    memo_map.get(&eclass.id).unwrap().clone()
}

// I'm just doing this for now; it may not actually be what we want in the
// end.
//type Result = Meta;
// Woah, this is giving a crazy error that is pointing to the
// define_language macro usage. Not Donna deal with that right now.
// TODO I'm wondering if the metadata can just act as an interpreter? It
// kind of serves that purpose already.

fn interpret_enode<M: egg::Metadata<MlpLanguage>>(
    egraph: &egg::EGraph<MlpLanguage, M>,
    enode: &egg::ENode<MlpLanguage>,
    env: &Environment,
    memo_map: &mut MemoizationMap,
) -> MemoizedInterpreterResult {
    debug!("interpreting enode: {:?}", enode);
    use MlpLanguage::*;
    match &enode.op {
        Symbol(name) => {
            debug!("interpreting symbol");
            MemoizedInterpreterResult::InterpretedValue(env[&name[..]].clone())
        }
        Dotprod => {
            // Evaluating Dotprod produces different results based on
            // whether it gets arguments. Actually, this is true of all
            // functions. If it doesn't get any arguments, then it should
            // evaluate to a callable function.
            match enode.children.len() {
                0 => {
                    // Expects a StreamValue::Pair as input.
                    fn dotprod(pair: ListValue) -> ListValue {
                        // Unpack the tensors to dot-product.
                        let (left, right): (
                            ndarray::ArrayD<ListValue>,
                            ndarray::ArrayD<ListValue>,
                        ) = match pair {
                            ListValue::Value(v) => match v {
                                Value::Tuple2(left, right) => match (*left, *right) {
                                    (ListValue::Value(left), ListValue::Value(right)) => {
                                        match (left, right) {
                                            (Value::ShapedList(left), Value::ShapedList(right)) => {
                                                (left, right)
                                            }
                                            _ => panic!(),
                                        }
                                    }
                                    _ => panic!(),
                                },
                                _ => panic!(),
                            },
                            _ => panic!(),
                        };

                        assert_eq!(left.ndim(), 1);
                        assert_eq!(right.ndim(), 1);

                        use std::iter::FromIterator;
                        fn unpack_and_flatten(
                            t: ndarray::ArrayD<ListValue>,
                        ) -> ndarray::Array1<DataType> {
                            ndarray::Array::from_iter(
                                t.iter()
                                    .map(|el| match el {
                                        ListValue::Scalar(s) => s,
                                        _ => panic!(),
                                    })
                                    .cloned(),
                            )
                        };
                        let left = unpack_and_flatten(left);
                        let right = unpack_and_flatten(right);

                        ListValue::Scalar(left.dot(&right))
                    }
                    MemoizedInterpreterResult::InterpretedValue(Value::Function(dotprod))
                }
                _ => panic!(),
            }
        }
        Rows => {
            // There should be one arg: a single tensor.
            assert_eq!(enode.children.len(), 1);

            // Expect that the result of interpreting it is a tensor.
            // TODO(gus) clean up this syntax.
            let arg_as_tensor = interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map);
            let arg_as_tensor = match arg_as_tensor {
                // Return early if we've found a recursive cycle
                MemoizedInterpreterResult::StillInterpreting => {
                    return MemoizedInterpreterResult::StillInterpreting
                }
                MemoizedInterpreterResult::CanNotInterpret => {
                    return MemoizedInterpreterResult::CanNotInterpret
                }
                MemoizedInterpreterResult::InterpretedValue(v) => v,
            };
            let arg_as_tensor: ndarray::ArrayD<ListValue> = match arg_as_tensor {
                Value::ShapedList(t) => t,
                _ => panic!(),
            };

            MemoizedInterpreterResult::InterpretedValue(Value::List(
                arg_as_tensor
                    .axis_iter(ndarray::Axis(0))
                    .map(|view| ListValue::Value(Value::ShapedList(view.to_owned())))
                    .collect::<std::vec::Vec<ListValue>>(),
            ))
        }
        Cols => {
            // There should be one arg: a single tensor.
            assert_eq!(enode.children.len(), 1);
            // Expect that the result of interpreting it is a tensor.
            let arg_as_tensor: ndarray::ArrayD<ListValue> =
                match interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    },
                };

            MemoizedInterpreterResult::InterpretedValue(Value::List(
                arg_as_tensor
                    .axis_iter(ndarray::Axis(1))
                    .map(|view| ListValue::Value(Value::ShapedList(view.to_owned())))
                    .collect::<std::vec::Vec<ListValue>>(),
            ))
        }
        CartesianProduct => {
            // Semantics of cartesian product:
            // Rightmost thing varies the fastest.

            // There should be two args, both of which should be lists.
            assert_eq!(enode.children.len(), 2);
            let left: std::vec::Vec<ListValue> =
                match interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::List(l) => l,
                        _ => panic!(),
                    },
                };
            let right: std::vec::Vec<ListValue> =
                match interpret_eclass(egraph, &egraph[enode.children[1]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::List(l) => l,
                        _ => panic!(),
                    },
                };

            // TODO(gus) figuring out exactly what should be the output of
            // CartesianProduct. It's a headache. I see three options:
            // 1. A 1-d List of pairs (Lists) of Tensors.
            // 2. A 2-d List of pairs (Lists) of Tensors.
            // 3. A Tensor whose elements are pairs (Lists) of Tensors.
            // I have a lot of notes on my thinking. Going to go with some
            // variant of 3.
            // Scratch that. That would require adding another tensor type
            // to our list of types. Don't want to do that for some reason.
            // Going with option 1 now.
            // For what it's worth, I'm back on doing 3.

            let new_shape = vec![left.len(), right.len()];

            use itertools::iproduct;
            let product_vector: std::vec::Vec<ListValue> = iproduct!(left, right)
                .map(|tuple| {
                    ListValue::Value(Value::Tuple2(
                        Box::<_>::new(tuple.0),
                        Box::<_>::new(tuple.1),
                    ))
                })
                .collect();

            let reshaped_into_tensor: ndarray::ArrayD<ListValue> =
                ndarray::ArrayD::<ListValue>::from_shape_vec(new_shape, product_vector).unwrap();

            MemoizedInterpreterResult::InterpretedValue(Value::ShapedList(reshaped_into_tensor))
        }
        ShapedMap => {
            assert_eq!(enode.children.len(), 2);

            // The first child of map is a function, passed as an argument.
            // For example, a dot-product node with no children. In the
            // future we may support other things that will evaluate to
            // functions.
            // The node should evaluate to a function value in our
            // interpreter.
            let function: fn(ListValue) -> ListValue =
                match interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::Function(f) => f,
                        _ => panic!(),
                    },
                };

            let input_shaped_list: ndarray::ArrayD<ListValue> =
                match interpret_eclass(egraph, &egraph[enode.children[1]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    },
                };

            MemoizedInterpreterResult::InterpretedValue(Value::ShapedList(
                ndarray::ArrayD::<ListValue>::from_shape_vec(
                    input_shaped_list.shape(),
                    input_shaped_list.iter().cloned().map(function).collect(),
                )
                .unwrap(),
            ))
        }
        Slice => {
            // TODO(gus) this is true for our minimal working example, not
            // expected to be true in the future, definitely not.
            assert_eq!(enode.children.len(), 5);

            let tensor: ndarray::ArrayD<_> =
                match interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    },
                };

            assert_eq!(tensor.shape().len(), 2);

            let ((row_slice_start, row_slice_end), (col_slice_start, col_slice_end)): (
                (usize, usize),
                (usize, usize),
            ) = match enode.children[1..5]
                .iter()
                .map(|eclass_id: &u32| {
                    match interpret_eclass(egraph, &egraph[*eclass_id], env, memo_map) {
                        // TODO(gus) this panic is me just being lazy. if
                        // StillInterpreting is found, we should return
                        MemoizedInterpreterResult::InterpretedValue(v) => match v {
                            Value::Usize(u) => u,
                            _ => panic!(),
                        },
                        _ => panic!(),
                    }
                })
                .collect::<std::vec::Vec<usize>>()
                .as_slice()
            {
                [row_slice_start, row_slice_end, col_slice_start, col_slice_end] => (
                    (*row_slice_start, *row_slice_end),
                    (*col_slice_start, *col_slice_end),
                ),
                _ => panic!(),
            };

            // A  consistent problem I was having with this library
            // was with converting to dynamic-dimension tensors.
            // Found out I can use into_dyn().
            MemoizedInterpreterResult::InterpretedValue(Value::ShapedList(
                tensor
                    .slice_move(s![
                        row_slice_start..row_slice_end,
                        col_slice_start..col_slice_end
                    ])
                    .into_dyn(),
            ))
        }
        Usize(u) => MemoizedInterpreterResult::InterpretedValue(Value::Usize(*u)),
        ShapedAdd => {
            let tensors: std::vec::Vec<ndarray::ArrayD<_>> = enode
                .children
                .iter()
                .map(
                    |eclass| match interpret_eclass(egraph, &egraph[*eclass], env, memo_map) {
                        MemoizedInterpreterResult::InterpretedValue(v) => match v {
                            Value::ShapedList(t) => t,
                            _ => panic!(),
                        },
                        // TODO(gus) this panic is me just being lazy.
                        _ => panic!(),
                    },
                )
                .collect();

            assert!(tensors.len() > 0);

            MemoizedInterpreterResult::InterpretedValue(Value::ShapedList(tensors.iter().fold(
                ndarray::Array::from_elem(tensors[0].shape(), ListValue::Scalar(DataType::zero())),
                |acc, t| acc + t,
            )))
        }
        BsgSystolicArray => panic!(),
        Concat => {
            // TODO(gus) this is true for our minimal working example, not
            // expected to be true in the future, definitely not.
            assert!(enode.children.len() >= 3);

            println!("Interpreting: {:?}", enode);
            // TODO(gus) it seems like there's a loop.
            // concat a has concat b as a child. concat b has concat a as a child.
            let mut tensors: std::vec::Vec<MemoizedInterpreterResult> = enode.children
                [0..enode.children.len() - 1]
                .iter()
                .map(|child| interpret_eclass(egraph, &egraph[*child], env, memo_map))
                .collect();
            // TODO(gus) the order of this and the next if block matters!
            if tensors.iter().any(|t| match t {
                MemoizedInterpreterResult::CanNotInterpret => true,
                _ => false,
            }) {
                return MemoizedInterpreterResult::CanNotInterpret;
            }
            if tensors.iter().any(|t| match t {
                MemoizedInterpreterResult::StillInterpreting => true,
                _ => false,
            }) {
                return MemoizedInterpreterResult::StillInterpreting;
            }
            let tensors: std::vec::Vec<ndarray::ArrayD<_>> = tensors
                .drain(..)
                .map(|result| match result {
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::ShapedList(t) => t,
                        _ => panic!(),
                    },
                    _ => panic!(),
                })
                .collect();

            let concat_axis: usize = match interpret_eclass(
                egraph,
                &egraph[enode.children[enode.children.len() - 1]],
                env,
                memo_map,
            ) {
                MemoizedInterpreterResult::StillInterpreting => {
                    return MemoizedInterpreterResult::StillInterpreting
                }
                MemoizedInterpreterResult::CanNotInterpret => {
                    return MemoizedInterpreterResult::CanNotInterpret
                }
                MemoizedInterpreterResult::InterpretedValue(v) => match v {
                    Value::Usize(u) => u,
                    _ => panic!(),
                },
            };

            // Have to stack manually, because ndarray::stack doesn't actually
            // support tensors holding values that don't implement Copy! :(

            let shapes: std::vec::Vec<&[usize]> = tensors.iter().map(|t| t.shape()).collect();
            // All have equal number of dimensions
            assert!(shapes.iter().all(|shape| shape.len() == shapes[0].len()));
            // All dimensions match, except for along the concat axis.
            assert!((0..shapes[0].len())
                .all(|i| i == concat_axis
                    || ((0..shapes.len()).all(|j| shapes[j][i] == shapes[0][i]))));

            let mut new_shape: std::vec::Vec<usize> = shapes[0].to_vec();
            new_shape[concat_axis] += (1..shapes.len())
                .map(|i| shapes[i][concat_axis])
                .sum::<usize>();
            let new_shape = new_shape;

            let mut new_tensor: ndarray::ArrayD<_> = ndarray::Array::default(new_shape.to_vec());
            let mut current_start_index: usize = 0;
            for i in 0..tensors.len() {
                let original_concat_axis_size: usize = shapes[i][concat_axis];
                use std::convert::TryFrom;
                let slices: std::vec::Vec<ndarray::SliceOrIndex> = (0..new_shape.len())
                    .map(|axis| {
                        if axis == concat_axis {
                            ndarray::SliceOrIndex::Slice {
                                start: isize::try_from(current_start_index).unwrap(),
                                end: Some(
                                    isize::try_from(
                                        current_start_index + original_concat_axis_size,
                                    )
                                    .unwrap(),
                                ),
                                step: 1,
                            }
                        } else {
                            ndarray::SliceOrIndex::Slice {
                                start: 0,
                                end: None,
                                step: 1,
                            }
                        }
                    })
                    .collect();
                let slice_info: ndarray::SliceInfo<
                    std::vec::Vec<ndarray::SliceOrIndex>,
                    ndarray::IxDyn,
                > = ndarray::SliceInfo::new(slices).unwrap();

                new_tensor
                    .slice_mut(slice_info.as_ref())
                    .assign(&tensors[i]);

                current_start_index += original_concat_axis_size;
            }

            assert_eq!(current_start_index, new_shape[concat_axis]);

            MemoizedInterpreterResult::InterpretedValue(Value::ShapedList(new_tensor))
        }
    }
}

// TODO(gus) is this possible? what is this called?
// FWIW, this isn't possible as-is, as rust complains of a cycle.
//type Shape = Vec<Either<i64, Shape>>;
// Only one level deep, will add more as needed. Need to figure out how to
// achieve this in general.
type Shape = Vec<Either<i64, Vec<Either<i64, Vec<i64>>>>>;

/// Given a function in MLPLanguage (i.e. dotprod) and an input shape,
/// what's the output shape?
fn infer_shape(
    node: &MlpLanguage,
    inner_shape: &Vec<Either<i64, Vec<i64>>>,
) -> Either<i64, Vec<Either<i64, Vec<i64>>>> {
    use MlpLanguage::*;
    match node {
        Dotprod => {
            // TODO(gus) a BIG assumption that's making our lives easier
            // right now and probably hurting us in the long run is that
            // dotprod doesn't take two arguments, but it takes one argument
            // which is a pair of vectors in a "tuple".
            //println!("dotprod input shape: {:?}", inner_shape);
            // Check for tuple of size 2
            assert_eq!(inner_shape.len(), 2);
            assert_eq!(*inner_shape[0].as_ref().left().unwrap(), 2);
            let right: &Vec<_> = inner_shape[1].as_ref().right().unwrap();
            assert_eq!(right.len(), 1);
            Left(1)
        }
        _ => {
            //println!("Unrecognized node type: {:?}", node);
            panic!()
        }
    }
}

type DataType = f64;

#[derive(Debug, Clone)]
struct Meta {
    // TODO(gus) implement tensors with many dimensions by using a vector
    // here. I just didn't want to mess around with figuring out Vecs in
    // Rust, and refcounting, etc etc.
    shape: Option<Shape>,
    scalar_value: Option<i64>,
    // If this eclass represents an op in the language, what op?
    // I'm adding this because I am using ops in the language as
    // first-class citizens (e.g. I can pass Dotprod to ShapedMap).
    // When it comes time to inspect a ShapedMap that takes an op, for example, I
    // need to be able to quickly figure out what the op represented by the
    // eclass is.
    // I specifically need this in the following context: I have an
    // infer_shape function that takes an op and an input shape and tells me
    // what the output shape will be. I need to be able to get the op.
    // TODO(gus) I think this is a hack honestly---what I should do is have
    // a metadata slot for "infer shape" functions. When metadata merges, we
    // can combine the list of these functions. When we need to infer a
    // shape, we can run all of the functions and make sure their outputs
    // all match.
    //op: Option<MlpLanguage>,
    // Actually, for right now, I'm going to take the even easier route and
    // assume that any eclass who is expected to be a usage of an op as a
    // first-class type will only have a single enode, which should be an op.
    // Tensor value. Should replace scalar_value.
    value: Option<ndarray::ArrayD<DataType>>,
}
impl PartialEq for Meta {
    fn eq(&self, other: &Self) -> bool {
        return self.shape == other.shape && self.scalar_value == other.scalar_value;
    }
}
impl egg::Metadata<MlpLanguage> for Meta {
    type Error = ();

    fn merge(&self, other: &Self) -> Self {
        assert_eq!(self, other);
        self.clone()
    }

    fn make(egraph: &egg::EGraph<MlpLanguage, Self>, enode: &egg::ENode<MlpLanguage>) -> Self {
        // We only know the value in the case of a Num.
        use MlpLanguage::*;
        match &enode.op {
            CartesianProduct => {
                assert_eq!(enode.children.len(), 2);
                let initial_shape_left: &Shape =
                    egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                assert_eq!(initial_shape_left.len(), 2);
                let initial_shape_right: &Shape =
                    egraph[enode.children[1]].metadata.shape.as_ref().unwrap();
                assert_eq!(initial_shape_left.len(), 2);
                // println!(
                //     "CartesianProduct: {:?}, {:?}",
                //     initial_shape_left, initial_shape_right
                // );
                // TODO(gus) i think right now we need the innermost dims to
                // match. While this makes sense for a matmul implemented
                // with a cartesian product, it doesn't make sense for
                // cartesian product in general.
                assert_eq!(
                    initial_shape_left[1].as_ref().right().unwrap(),
                    initial_shape_right[1].as_ref().right().unwrap(),
                );
                let innermost_shape: Vec<i64> = initial_shape_left[1]
                    .as_ref()
                    .right()
                    .unwrap()
                    .iter()
                    .map(|i| i.clone().left().unwrap())
                    .collect();
                let new_inner_shape: Vec<Either<i64, Vec<i64>>> =
                    vec![Left(2), Right(innermost_shape)];
                let new_shape: Shape = vec![
                    Left(*initial_shape_left[0].as_ref().left().unwrap()),
                    Left(*initial_shape_right[0].as_ref().left().unwrap()),
                    Right(new_inner_shape),
                ];
                Meta {
                    shape: Some(new_shape),
                    scalar_value: None,
                    value: None,
                }
            }
            Rows => {
                assert_eq!(enode.children.len(), 1);
                let initial_shape: &Shape =
                    egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                //println!("{:?}", initial_shape);
                assert_eq!(initial_shape.len(), 2);
                // TODO(gus) should probably check that the shape is of a
                // shape we expect
                // Transform initial shape: (row (tensor...)) becomes a list
                // which has shape (list num_rows (list other_dim...))
                // TODO(gus) we're also doing "rows" and "columns" right
                // now, which is very specific to 2 dimensions.
                // We can generalize this later
                let all_but_first: Vec<Either<i64, Vec<i64>>> = initial_shape[1..]
                    .as_ref()
                    .into_iter()
                    .map(|i| Left(i.clone().left().unwrap()))
                    .collect();
                let new_shape: Shape = vec![
                    Left(initial_shape[0].clone().left().unwrap()),
                    Right(all_but_first),
                ];
                Meta {
                    shape: Some(new_shape),
                    scalar_value: None,
                    value: None,
                }
            }
            Cols => {
                assert_eq!(enode.children.len(), 1);
                let initial_shape: &Shape =
                    egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                //println!("Cols: {:?}", initial_shape);
                assert_eq!(initial_shape.len(), 2);
                let all_but_second: Vec<Either<i64, Vec<i64>>> = initial_shape[0..1]
                    .as_ref()
                    .into_iter()
                    .map(|i| Left(i.clone().left().unwrap()))
                    .collect();
                let new_shape: Shape = vec![
                    Left(initial_shape[1].clone().left().unwrap()),
                    Right(all_but_second),
                ];
                Meta {
                    shape: Some(new_shape),
                    scalar_value: None,
                    value: None,
                }
            }
            ShapedMap => {
                assert_eq!(enode.children.len(), 2);
                let shape: &Shape = egraph[enode.children[1]].metadata.shape.as_ref().unwrap();

                // Get the MlpLanguage op that is being mapped onto the
                // input.
                let class: &egg::EClass<MlpLanguage, Meta> = &egraph[enode.children[0]];
                // Assume that the class only has one child, which is an
                // MlpLanguage op with no children. The fact that it has no
                // children means it's not a call.
                assert_eq!(class.len(), 1);
                let node: &egg::ENode<MlpLanguage> = &class.nodes[0];
                assert_eq!(node.children.len(), 0);
                let op: &MlpLanguage = &node.op;

                let mut new_shape: Shape = shape.clone();
                // TODO(gus)
                // we assume the last thing in the top level shape describes
                // the list's elements' shape. (All things in the list have
                // the same shape; that's a limitation of our shape type
                // system.)
                // TODO(gus) if infer_shape returns Left(1), what do we do?
                // do we remove the dimension entirely?
                // that's what i'm going to do now, because it is solving a
                // headache for me---but it may cause more down the road.
                // I think the type system is rotten. Worth a try but lists
                // and tensors should probably just be different.
                let inferred_shape =
                    infer_shape(op, shape.last().unwrap().as_ref().right().unwrap());
                if inferred_shape == Left(1) {
                    new_shape.remove(new_shape.len() - 1);
                } else {
                    new_shape[shape.len() - 1] = inferred_shape;
                }
                Meta {
                    shape: Some(new_shape),
                    scalar_value: None,
                    value: None,
                }
            }
            Dotprod => {
                assert_eq!(enode.children.len(), 0);
                //println!("Dotprod");
                Meta {
                    shape: None,
                    scalar_value: None,
                    value: None,
                }
            }
            BsgSystolicArray => {
                assert_eq!(enode.children.len(), 2);
                let left_shape = &egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                let right_shape = &egraph[enode.children[1]].metadata.shape.as_ref().unwrap();
                // TODO(gus) very rough approx of what's actually right.
                assert_eq!(left_shape[1], right_shape[0]);
                assert_eq!(left_shape[2..], right_shape[2..]);
                assert!(left_shape[2..]
                    .iter()
                    .all(|i| *i.as_ref().left().unwrap() == 1));
                let new_shape: Shape =
                    [&left_shape[0..1], &right_shape[1..2], &left_shape[2..]].concat();
                Meta {
                    shape: Some(new_shape),
                    scalar_value: None,
                    value: None,
                }
            }
            Slice => {
                // TODO(gus) left off here.
                let shape_to_be_sliced: &Shape =
                    &egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                // It should be a vector of scalars only.
                let shape_to_be_sliced: std::vec::Vec<i64> = shape_to_be_sliced
                    .iter()
                    .map(|i| *i.as_ref().left().unwrap())
                    .collect();
                let slice_indices: std::vec::Vec<usize> = enode.children[1..]
                    .iter()
                    .map(|id| {
                        match interpret_eclass(
                            &egraph,
                            &egraph[*id],
                            &HashMap::default(),
                            &mut MemoizationMap::default(),
                        ) {
                            MemoizedInterpreterResult::InterpretedValue(v) => match v {
                                Value::Usize(u) => u,
                                _ => panic!(),
                            },
                            _ => panic!(),
                        }
                    })
                    .collect();

                // For every dimension, there should be two slice indices:
                // ( [beginning, end) )
                // Note that this is a pretty restrictive syntax for now.
                assert_eq!(0, slice_indices.len() % 2);
                assert_eq!(shape_to_be_sliced.len(), slice_indices.len() / 2);

                let mut new_shape = shape_to_be_sliced.clone();

                for dim_i in 0..shape_to_be_sliced.len() {
                    let dim_val: i64 = shape_to_be_sliced[dim_i];
                    let slice_start: usize = slice_indices[dim_i * 2];
                    let slice_end: usize = slice_indices[dim_i * 2 + 1];
                    use std::convert::TryInto;
                    assert!(slice_end <= dim_val.try_into().unwrap());
                    assert!(slice_start <= slice_end);
                    if slice_end - slice_start > 0 {
                        // If the slice actually needs to produce values...
                        assert!(slice_start < dim_val.try_into().unwrap());
                    }

                    new_shape[dim_i] = (slice_end - slice_start).try_into().unwrap();
                }
                let new_shape: Shape = new_shape.iter().map(|i| Left(*i)).collect();

                Meta {
                    shape: Some(new_shape),
                    scalar_value: None,
                    value: None,
                }
            }
            Concat => {
                // Need at least two arrays and always need one axis
                assert!(enode.children.len() >= 3);
                let shapes: std::vec::Vec<&Shape> = (0..(enode.children.len() - 1))
                    .map(|i| egraph[enode.children[i]].metadata.shape.as_ref().unwrap())
                    .collect();

                // Figure out if they're tensors or lists.
                // They should be vectors of scalars only.
                // Okay, but what if we're concatting lists?
                let shapes: std::vec::Vec<std::vec::Vec<i64>> = shapes
                    .iter()
                    .map(|shape| shape.iter().map(|i| *i.as_ref().left().unwrap()).collect())
                    .collect();
                let concat_axis: usize = match interpret_eclass(
                    &egraph,
                    &egraph[enode.children[enode.children.len() - 1]],
                    &HashMap::default(),
                    &mut MemoizationMap::default(),
                ) {
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::Usize(u) => u,
                        _ => panic!(),
                    },
                    _ => panic!(),
                };

                assert!((0..shapes.len()).all(|i| shapes[i].len() == shapes[0].len()));
                // The two shapes must be equal, except for along the concat
                // axis.
                assert!((0..shapes[0].len()).all(|i| i == concat_axis
                    || ((0..shapes.len()).all(|j| shapes[j][i] == shapes[0][i]))));

                let mut new_shape = shapes[0].clone();
                new_shape[concat_axis] += (1..shapes.len())
                    .map(|i| shapes[i][concat_axis])
                    .sum::<i64>();
                let new_shape: Shape = new_shape.iter().map(|i| Left(*i)).collect();
                //println!("concat input shapes: {:?}", shapes);
                //println!("concat output shape: {:?}", new_shape);

                Meta {
                    shape: Some(new_shape),
                    scalar_value: None,
                    value: None,
                }
            }
            Usize(_) => Meta {
                shape: None,
                // TODO(gus) ugh, this isn't necessarily right, is it?
                scalar_value: None,
                value: None,
            },
            ShapedAdd => panic!(),
            Symbol(name) => {
                //println!("Symbol");
                Meta {
                    shape: Some(match &name[..] {
                        "in" => vec![Left(1), Left(784)],
                        "w1" => vec![Left(784), Left(512)],
                        "w2" => vec![Left(512), Left(512)],
                        "w3" => vec![Left(512), Left(10)],
                        // TODO(gus) have to figure out a way around this. Max
                        // seems to think the tensors should just go into the
                        // egraph. I was hoping to have some kind of environment
                        // that we could wrap the egraph in (would have to be
                        // accessible from here), but Max doesn't have that nor
                        // does he plan to implement it.
                        "single-matrix-multiply-input-a" => vec![Left(32), Left(32)],
                        "single-matrix-multiply-input-b" => vec![Left(32), Left(32)],
                        _ => panic!("No shape defined for {}", name),
                    }),
                    scalar_value: None,
                    value: None,
                }
            }
        }
    }
}

fn load_npy(path: &str) -> ndarray::ArrayD<DataType> {
    ndarray_npy::read_npy::<_, ndarray::ArrayD<DataType>>(path).unwrap()
}
fn pack_interpreter_input(array: ndarray::ArrayD<DataType>) -> Value {
    Value::ShapedList(
        ndarray::ArrayD::<ListValue>::from_shape_vec(
            array.shape(),
            array
                .iter()
                .cloned()
                .map(|scalar| ListValue::Scalar(scalar))
                .collect(),
        )
        .unwrap(),
    )
}
fn unpack_interpreter_output(output: MemoizedInterpreterResult) -> ndarray::ArrayD<DataType> {
    match output {
        MemoizedInterpreterResult::InterpretedValue(v) => match v {
            Value::ShapedList(t) => ndarray::ArrayD::<DataType>::from_shape_vec(
                t.shape(),
                t.iter()
                    .cloned()
                    .map(|list_val| match list_val {
                        ListValue::Scalar(s) => s,
                        _ => panic!(),
                    })
                    .collect(),
            )
            .unwrap(),
            _ => panic!(),
        },
        _ => panic!(),
    }
}

egg::define_language! {
    enum SingleMatrixMultiplyLanguage {
        Rows = "rows",
        Cols = "cols",
        CartesianProduct = "cartesian-product",
        // Map dot product:
        // for a tensor with shape
        // [a1, ..., an, 2, b],
        // the result is a new tensor with shape
        // [a1, ..., an]
        // Whose elements are the dot product of the two b-length vectors at
        // each position in the original array.
        MapDotProduct = "map-dot-product",
        BsgSystolicArray = "bsg_systolic_array_weight_stationary",
        // Slice into list/tensor/whatever we're calling them
        Slice = "slice",
        Concat = "concat",
        // TODO(gus) this will probably need to be signed at some point?
        Usize(usize),
        Symbol(String),
    }
}

#[derive(Debug, Clone, PartialEq)]
struct SingleMatrixMultiplyMeta {
    shape: Option<ndarray::IxDyn>,
    usize_value: Option<usize>,
}
impl egg::Metadata<SingleMatrixMultiplyLanguage> for SingleMatrixMultiplyMeta {
    type Error = ();

    fn merge(&self, other: &Self) -> Self {
        assert_eq!(self, other);
        self.clone()
    }

    fn make(
        egraph: &egg::EGraph<SingleMatrixMultiplyLanguage, Self>,
        enode: &egg::ENode<SingleMatrixMultiplyLanguage>,
    ) -> Self {
        // We only know the value in the case of a Num.
        use SingleMatrixMultiplyLanguage::*;
        match &enode.op {
            CartesianProduct => {
                // This cartesian product works a little differently from
                // before, given the new, simplified shape system.
                // It wants to pair up the very last dimension of the two
                // input arrays. I.e. it views the two input arrays as
                // having shapes
                // [a1, a2, ..., an, c]
                // [b1, b2, ..., bn, c]
                // And sees them essentially as two tensors of vectors:
                // input 1 is a [a1, ..., an] sized tensor of c-length vectors
                // similar for input 2.
                // So I think our only requirement is that the last dimension
                // is the same size. And then the resulting size is
                // [a1, ... an, b1, ..., bn, 2, c].
                // Right now i'm just implementing it for input arrays with 2
                // dimensions, though.
                assert_eq!(enode.children.len(), 2);
                let initial_shape_left: &ndarray::IxDyn =
                    &egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                assert_eq!(initial_shape_left.as_array_view().len(), 2);
                let initial_shape_right: &ndarray::IxDyn =
                    &egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                assert_eq!(initial_shape_right.as_array_view().len(), 2);
                assert_eq!(
                    initial_shape_left[initial_shape_left.as_array_view().len() - 1],
                    initial_shape_right[initial_shape_right.as_array_view().len() - 1],
                );

                // New shape is [a1, ..., an, b1, ..., bn, c].
                let mut new_shape: Vec<usize> = initial_shape_left
                    .as_array_view()
                    .iter()
                    .take(initial_shape_left.as_array_view().len() - 1)
                    .copied()
                    .collect();
                new_shape.extend(
                    initial_shape_right
                        .as_array_view()
                        .iter()
                        .take(initial_shape_right.as_array_view().len() - 1),
                );
                new_shape.push(2);
                new_shape.push(initial_shape_left[initial_shape_left.as_array_view().len() - 1]);
                let new_shape: ndarray::IxDyn = ndarray::IxDyn(&new_shape[..]);
                assert_eq!(
                    new_shape.as_array_view().len(),
                    initial_shape_left.as_array_view().len() - 1
                        + initial_shape_right.as_array_view().len()
                        - 1
                        + 1
                        + 1
                );
                SingleMatrixMultiplyMeta {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            Rows => {
                assert_eq!(enode.children.len(), 1);
                let initial_shape: ndarray::IxDyn = egraph[enode.children[0]]
                    .metadata
                    .shape
                    .as_ref()
                    .unwrap()
                    .clone();
                // Doesn't have to be true in the future.
                assert_eq!(initial_shape.as_array_view().len(), 2);
                // Our new, simpler system makes this way easier!
                SingleMatrixMultiplyMeta {
                    shape: Some(initial_shape),
                    usize_value: None,
                }
            }
            Cols => {
                assert_eq!(enode.children.len(), 1);
                let mut initial_shape: ndarray::IxDyn = egraph[enode.children[0]]
                    .metadata
                    .shape
                    .as_ref()
                    .unwrap()
                    .clone();
                // Doesn't have to be true in the future.
                assert_eq!(initial_shape.as_array_view().len(), 2);

                // The column dimension gets moved first. For a two-dimensional
                // array, it's a transpose!
                let cols_val: usize = initial_shape[1];
                initial_shape[1] = initial_shape[0];
                initial_shape[0] = cols_val;

                SingleMatrixMultiplyMeta {
                    shape: Some(initial_shape),
                    usize_value: None,
                }
            }
            MapDotProduct => {
                assert_eq!(enode.children.len(), 1);
                let shape: &ndarray::IxDyn =
                    egraph[enode.children[0]].metadata.shape.as_ref().unwrap();

                assert!(shape.as_array_view().len() >= 3);
                assert_eq!(shape[shape.as_array_view().len() - 2], 2);

                let new_shape: ndarray::IxDyn = ndarray::IxDyn(
                    &shape
                        .as_array_view()
                        .iter()
                        .take(shape.as_array_view().len() - 2)
                        .copied()
                        .collect::<Vec<usize>>()[..],
                );

                SingleMatrixMultiplyMeta {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            BsgSystolicArray => panic!(),
            Slice => {
                let shape_to_be_sliced: &ndarray::IxDyn =
                    &egraph[enode.children[0]].metadata.shape.as_ref().unwrap();
                let slice_indices: std::vec::Vec<usize> = enode.children[1..]
                    .iter()
                    .map(|id| egraph[*id].metadata.usize_value.unwrap())
                    .collect();

                // For every dimension, there should be two slice indices:
                // ( [beginning, end) )
                // Note that this is a pretty restrictive syntax for now.
                assert_eq!(0, slice_indices.len() % 2);
                assert_eq!(
                    shape_to_be_sliced.as_array_view().len(),
                    slice_indices.len() / 2
                );

                let mut new_shape = shape_to_be_sliced.clone();

                for dim_i in 0..shape_to_be_sliced.as_array_view().len() {
                    let dim_val: usize = shape_to_be_sliced[dim_i];
                    let slice_start: usize = slice_indices[dim_i * 2];
                    let slice_end: usize = slice_indices[dim_i * 2 + 1];
                    use std::convert::TryInto;
                    assert!(slice_end <= dim_val.try_into().unwrap());
                    assert!(slice_start <= slice_end);
                    if slice_end - slice_start > 0 {
                        // If the slice actually needs to produce values...
                        assert!(slice_start < dim_val.try_into().unwrap());
                    }

                    new_shape[dim_i] = (slice_end - slice_start).try_into().unwrap();
                }

                SingleMatrixMultiplyMeta {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            Concat => {
                // Need at least two arrays and always need one axis
                assert!(enode.children.len() >= 3);
                let shapes: std::vec::Vec<&ndarray::IxDyn> = (0..(enode.children.len() - 1))
                    .map(|i| egraph[enode.children[i]].metadata.shape.as_ref().unwrap())
                    .collect();

                let concat_axis: usize = egraph[enode.children[enode.children.len() - 1]]
                    .metadata
                    .usize_value
                    .unwrap();

                assert!((0..shapes.len())
                    .all(|i| shapes[i].as_array_view().len() == shapes[0].as_array_view().len()));
                // The two shapes must be equal, except for along the concat
                // axis.
                assert!(
                    (0..shapes[0].as_array_view().len()).all(|i| i == concat_axis
                        || ((0..shapes.len()).all(|j| shapes[j][i] == shapes[0][i])))
                );

                let mut new_shape = shapes[0].clone();
                new_shape[concat_axis] += (1..shapes.len())
                    .map(|i| shapes[i][concat_axis])
                    .sum::<usize>();
                //println!("concat input shapes: {:?}", shapes);
                //println!("concat output shape: {:?}", new_shape);

                SingleMatrixMultiplyMeta {
                    shape: Some(new_shape),
                    usize_value: None,
                }
            }
            Usize(u) => SingleMatrixMultiplyMeta {
                shape: None,
                usize_value: Some(*u),
            },
            Symbol(name) => {
                //println!("Symbol");
                SingleMatrixMultiplyMeta {
                    shape: Some(ndarray::IxDyn(
                        &(match &name[..] {
                            "in" => vec![1, 784],
                            "w1" => vec![784, 512],
                            "w2" => vec![512, 512],
                            "w3" => vec![512, 10],
                            // TODO(gus) have to figure out a way around this. Max
                            // seems to think the tensors should just go into the
                            // egraph. I was hoping to have some kind of environment
                            // that we could wrap the egraph in (would have to be
                            // accessible from here), but Max doesn't have that nor
                            // does he plan to implement it.
                            "single-matrix-multiply-input-a" => vec![32, 32],
                            "single-matrix-multiply-input-b" => vec![32, 32],
                            _ => panic!("No shape defined for {}", name),
                        })[..],
                    )),
                    usize_value: None,
                }
            }
        }
    }
}

fn single_matrix_multiply() {
    struct SplitConcatApplier {
        a: egg::Var,
    };
    impl egg::Applier<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta> for SplitConcatApplier {
        fn apply_one(
            &self,
            egraph: &mut egg::EGraph<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta>,
            _id: egg::Id,
            subst: &egg::Subst,
        ) -> std::vec::Vec<egg::Id> {
            let a: egg::Id = subst[&self.a];
            let shape = egraph[a].metadata.shape.as_ref().unwrap().clone();
            //println!("{:?}", shape);

            assert_eq!(shape.as_array_view().len(), 2);
            assert_eq!(0, shape[0] % 16);
            assert_eq!(0, shape[1] % 16);

            let mut to_be_concatted_along_axis_0 = std::vec::Vec::default();
            for i in 0..shape[0] / 16 {
                let mut to_be_concatted_along_axis_1 = std::vec::Vec::default();
                for j in 0..shape[1] / 16 {
                    use std::convert::TryInto;
                    let x_slice_start = (16 * i).try_into().unwrap();
                    let x_slice_end = (16 * (i + 1)).try_into().unwrap();
                    let y_slice_start = (16 * j).try_into().unwrap();
                    let y_slice_end = (16 * (j + 1)).try_into().unwrap();
                    let x_slice_start_id: egg::Id = egraph.add(egg::ENode::leaf(
                        SingleMatrixMultiplyLanguage::Usize(x_slice_start),
                    ));
                    let x_slice_end_id: egg::Id = egraph.add(egg::ENode::leaf(
                        SingleMatrixMultiplyLanguage::Usize(x_slice_end),
                    ));
                    let y_slice_start_id: egg::Id = egraph.add(egg::ENode::leaf(
                        SingleMatrixMultiplyLanguage::Usize(y_slice_start),
                    ));
                    let y_slice_end_id: egg::Id = egraph.add(egg::ENode::leaf(
                        SingleMatrixMultiplyLanguage::Usize(y_slice_end),
                    ));
                    to_be_concatted_along_axis_1.push(egraph.add(egg::ENode::new(
                        SingleMatrixMultiplyLanguage::Slice,
                        vec![
                            a,
                            x_slice_start_id,
                            x_slice_end_id,
                            y_slice_start_id,
                            y_slice_end_id,
                        ],
                    )));
                }
                // Args should be a list of the sliced arrays, plus the axis
                // along which to stitch them back together.
                let mut args: std::vec::Vec<egg::Id> = to_be_concatted_along_axis_1;
                args.push(egraph.add(egg::ENode::leaf(SingleMatrixMultiplyLanguage::Usize(1))));
                to_be_concatted_along_axis_0
                    .push(egraph.add(egg::ENode::new(SingleMatrixMultiplyLanguage::Concat, args)));
            }
            let mut args: std::vec::Vec<egg::Id> = to_be_concatted_along_axis_0;
            args.push(egraph.add(egg::ENode::leaf(SingleMatrixMultiplyLanguage::Usize(0))));
            let concat_id: egg::Id =
                egraph.add(egg::ENode::new(SingleMatrixMultiplyLanguage::Concat, args));

            vec![concat_id]
        }
    }
    fn has_shape(
        var: &'static str,
    ) -> impl Fn(
        &mut egg::EGraph<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta>,
        egg::Id,
        &egg::Subst,
    ) -> bool {
        let var = var.parse().unwrap();
        move |egraph, _, subst| !egraph[subst[&var]].metadata.shape.is_none()
    }
    fn is_symbol(
        var: &'static str,
    ) -> impl Fn(
        &mut egg::EGraph<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta>,
        egg::Id,
        &egg::Subst,
    ) -> bool {
        let var = var.parse().unwrap();
        // TODO(gus) should this be `all` or `any` or something else entirely?
        move |egraph, _, subst| {
            egraph[subst[&var]]
                .nodes
                .iter()
                .map(|enode| match enode.op {
                    SingleMatrixMultiplyLanguage::Symbol(_) => true,
                    _ => false,
                })
                .all(|x| x)
        }
    }
    fn dimension_greater_than(
        var: &'static str,
        axis: usize,
        greater_than: usize,
    ) -> impl Fn(
        &mut egg::EGraph<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta>,
        egg::Id,
        &egg::Subst,
    ) -> bool {
        let var = var.parse().unwrap();
        move |egraph, _, subst| {
            egraph[subst[&var]].metadata.shape.as_ref().unwrap()[axis] > greater_than
        }
    }
    fn dimension_is_even(
        var: &'static str,
        axis: usize,
    ) -> impl Fn(
        &mut egg::EGraph<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta>,
        egg::Id,
        &egg::Subst,
    ) -> bool {
        let var = var.parse().unwrap();
        move |egraph, _, subst| egraph[subst[&var]].metadata.shape.as_ref().unwrap()[axis] % 2 == 0
    }

    struct RewriteNonMatchingCartConcatApplier {
        a1: egg::Var,
        a2: egg::Var,
        a_axis: usize,
        b1: egg::Var,
        b2: egg::Var,
        b_axis: usize,
    }
    impl egg::Applier<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta>
        for RewriteNonMatchingCartConcatApplier
    {
        fn apply_one(
            &self,
            egraph: &mut egg::EGraph<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta>,
            _id: egg::Id,
            subst: &egg::Subst,
        ) -> std::vec::Vec<egg::Id> {
            // For now, just want to handle these cases.
            assert!(self.a_axis == 0 || self.a_axis == 1);
            assert!(self.b_axis == 0 || self.b_axis == 1);
            assert_ne!(self.a_axis, self.b_axis);

            // We will break up the as into smaller chunks and the bs into
            // smaller chunks, so that they all match in size.
            // The goal is to have the innermost concats be along axis 0, and
            // the outermost concats to be along axis 1. Additionally, the goal
            // is that the result should only involve cartesian products of
            // concats, where the left and right concat use the same axis.
            // Then, existing rewrites can be used to bubble the concats up
            // through the cartesian products.

            // Each a needs to be split into 4; each b needs to be split into 4.

            // First we want to construct all of the concats along the 1 axis.
            // These will become our innermost concats.
            // One of these is already concatted along the 1 axis!

            // TODO(gus) left off here, I think I should actually do something
            // simpler here and just rewrite the two concats that are the
            // children of this cartesian product.
            // It needs some information from elsewhere in the graph, though,
            // that's the tough thing.

            // So we're going to slice-and-concat all 4 tensors. We'll slice the
            // as based on the bs size, and slice the bs based on the as size.
            // TODO(gus) I could write an even simpler rewrite rule that slices
            // more indiscriminately, everywhere. Right now I'm using some
            // context clue (the overarching cartesian product) to only apply
            // this where needed.

            // All I actually want to do is to rewrite that second concat.
            //  (cartesian-product
            //   (concat ?a1 ?a2 0)
            //   (concat ?b1 ?b2 1)
            //  )
            //  (cartesian-product
            //   (concat ?a1 ?a2 0)
            //   (concat (concat (slice ?b1) (slice ?b1)  0)
            //  )
            //

            vec![]
        }
    }

    struct SplitApplier {
        a: egg::Var,
        axis: usize,
    }
    impl egg::Applier<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta> for SplitApplier {
        fn apply_one(
            &self,
            egraph: &mut egg::EGraph<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta>,
            id: egg::Id,
            _subst: &egg::Subst,
        ) -> std::vec::Vec<egg::Id> {
            let shape: ndarray::IxDyn = egraph[id].metadata.shape.as_ref().unwrap().clone();
            assert_eq!(shape[self.axis] % 2, 0);
            let low_bound = 0;
            let low_bound_id = egraph.add(egg::ENode::leaf(SingleMatrixMultiplyLanguage::Usize(
                low_bound,
            )));
            let high_bound = shape[self.axis];
            let high_bound_id = egraph.add(egg::ENode::leaf(SingleMatrixMultiplyLanguage::Usize(
                high_bound,
            )));
            let middle_bound = high_bound / 2;
            let middle_bound_id = egraph.add(egg::ENode::leaf(
                SingleMatrixMultiplyLanguage::Usize(middle_bound),
            ));

            let mut slice_0_indices = std::vec::Vec::new();
            for i in 0..shape.as_array_view().len() {
                if i == self.axis {
                    // If this is the axis we're splitting on, then access the
                    // first half.
                    slice_0_indices.push(low_bound_id);
                    slice_0_indices.push(middle_bound_id);
                } else {
                    // Otherwise, access the whole axis.
                    slice_0_indices
                        .push(egraph.add(egg::ENode::leaf(SingleMatrixMultiplyLanguage::Usize(0))));
                    slice_0_indices.push(egraph.add(egg::ENode::leaf(
                        SingleMatrixMultiplyLanguage::Usize(shape[i]),
                    )));
                }
            }

            let mut slice_1_indices = std::vec::Vec::new();
            for i in 0..shape.as_array_view().len() {
                if i == self.axis {
                    // If this is the axis we're splitting on, then access the
                    // second half.
                    slice_1_indices.push(middle_bound_id);
                    slice_1_indices.push(high_bound_id);
                } else {
                    // Otherwise, access the whole axis.
                    slice_1_indices
                        .push(egraph.add(egg::ENode::leaf(SingleMatrixMultiplyLanguage::Usize(0))));
                    slice_1_indices.push(egraph.add(egg::ENode::leaf(
                        SingleMatrixMultiplyLanguage::Usize(shape[i]),
                    )));
                }
            }

            let mut slice_0_children = std::vec::Vec::new();
            slice_0_children.push(id);
            slice_0_children.append(&mut slice_0_indices);

            let mut slice_1_children = std::vec::Vec::new();
            slice_1_children.push(id);
            slice_1_children.append(&mut slice_1_indices);

            let slice_0_id = egraph.add(egg::ENode::new(
                SingleMatrixMultiplyLanguage::Slice,
                slice_0_children,
            ));
            let slice_1_id = egraph.add(egg::ENode::new(
                SingleMatrixMultiplyLanguage::Slice,
                slice_1_children,
            ));
            //println!("{:?}", egraph[slice_0_id]);
            //println!("{:?}", egraph[slice_1_id]);

            let axis_usize_id = egraph.add(egg::ENode::leaf(SingleMatrixMultiplyLanguage::Usize(
                self.axis,
            )));

            // Add
            // (concat )
            let id: egg::Id = egraph.add(egg::ENode::new(
                SingleMatrixMultiplyLanguage::Concat,
                vec![slice_0_id, slice_1_id, axis_usize_id],
            ));
            vec![id]
        }
    }

    let rws = vec![
        // TODO(gus) damn it, I still think that usize-halve won't even be enough.
        // TODO(gus) the if statements actually run backwards.
        egg::rewrite!("split-x"; "?a" => {SplitApplier{axis: 0, a:"?a".parse().unwrap()}} if dimension_greater_than("?a", 0, 16) if dimension_is_even("?a", 0) if has_shape("?a")),
        egg::rewrite!("split-y"; "?a" => {SplitApplier{axis: 1, a:"?a".parse().unwrap()}} if dimension_greater_than("?a", 1, 16) if dimension_is_even("?a", 1) if has_shape("?a")),
        egg::rewrite!("split-concat"; "?a" => {SplitConcatApplier{a:"?a".parse().unwrap()}} if has_shape("?a") if is_symbol("?a")),
        egg::rewrite!("bubble-concat-through-rows-axis-0"; "(rows (concat ?a ?b 0))"
                      => "(concat (rows ?a) (rows ?b) 0)"),
        egg::rewrite!("bubble-concat-through-rows-axis-1"; "(rows (concat ?a ?b 1))"
                      => "(concat (rows ?a) (rows ?b) 1)"),
        egg::rewrite!("bubble-concat-through-cols-axis-0"; "(cols (concat ?a ?b 0))"
                      => "(concat (cols ?a) (cols ?b) 1)"),
        egg::rewrite!("bubble-concat-through-cols-axis-1"; "(cols (concat ?a ?b 1))"
                      => "(concat (cols ?a) (cols ?b) 0)"),
        // TODO(gus) this isn't the only way this could be done.
        // Also there's gotta be a name for this in terms of algebraic rules
        // TODO(gus) would it make our pattern-matching life easier if (1) we
        // put the axes at the start of concat and (2) we used cons cells?
        egg::rewrite!("bubble-concat-through-cartesian-product-axes-0-0";
                      "(cartesian-product (concat ?a1 ?a2 0) (concat ?b1 ?b2 0))"
                      // TODO(gus) check this
                      => "(concat
                           (concat (cartesian-product ?a1 ?b1)
                                   (cartesian-product ?a1 ?b2) 1)
                           (concat (cartesian-product ?a2 ?b1)
                                   (cartesian-product ?a2 ?b2) 1)
                           0)"
        ),
        egg::rewrite!(
        "rewrite-nonmatching-cartesian-product-concat";
        "(cartesian-product
              (concat ?a1 ?a2 0)
              (concat ?b1 ?b2 1)
             )" =>
        {RewriteNonMatchingCartConcatApplier{
            a1:"?a1".parse().unwrap(),
            a2:"?a2".parse().unwrap(),
            a_axis:0,
            b1:"?b1".parse().unwrap(),
            b2:"?b2".parse().unwrap(),
            b_axis:1,
        }}),
        // egg::rewrite!("bubble-concat-through-cartesian-product"; "(cartesian-product (concat ?a ?b ?c ?d ?axis) (concat ?e ?f ?g ?h ?axis))" =>
        // // TODO(gus) I think this one's where the magic happens :)
        // {BubbleConcatThroughCartesianProductApplier{
        //     a:"?a".parse().unwrap(),
        //     b:"?b".parse().unwrap(),
        //     c:"?c".parse().unwrap(),
        //     d:"?d".parse().unwrap(),
        //     e:"?e".parse().unwrap(),
        //     f:"?f".parse().unwrap(),
        //     g:"?g".parse().unwrap(),
        //     h:"?h".parse().unwrap(),
        //     axis:"?axis".parse().unwrap(),

        // }}),
    ];

    let program = "
     (map-dot-product
      (cartesian-product
       (rows single-matrix-multiply-input-a)
       (cols single-matrix-multiply-input-b)
      )
     )
     "
    .parse()
    .unwrap();

    let (egraph, id) =
        egg::EGraph::<SingleMatrixMultiplyLanguage, SingleMatrixMultiplyMeta>::from_expr(&program);
    egraph
        .dot()
        .to_svg("single-matrix-multiply-before-rewrites.svg")
        .unwrap();
    let runner = egg::Runner::new().with_egraph(egraph).run(&rws);
    runner
        .egraph
        .dot()
        .to_svg("single-matrix-multiply-after-rewrites.svg")
        .unwrap();

    // let out = interpret_eclass(
    //     &runner.egraph,
    //     &runner.egraph[id],
    //     &env,
    //     &mut MemoizationMap::new(),
    // );
    // let out = unpack_interpreter_output(out);
    // assert!(out_true.abs_diff_eq(&out, 1e-8));
}

fn _mlp() {
    let slice_test_program_1 = "(slice a 0 1 0 1)".parse().unwrap();
    let slice_test_program_2 = "(slice a 1 2 0 2)".parse().unwrap();
    let input = pack_interpreter_input(
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap(),
    );
    let mut env = Environment::new();
    env.insert("a", input);
    let (egraph, id) = egg::EGraph::<MlpLanguage, ()>::from_expr(&slice_test_program_1);
    let out = unpack_interpreter_output(interpret_eclass(
        &egraph,
        &egraph[id],
        &env,
        &mut MemoizationMap::new(),
    ));
    assert_eq!(
        out,
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![1, 1], vec![1.]).unwrap()
    );
    let (egraph, id) = egg::EGraph::<MlpLanguage, ()>::from_expr(&slice_test_program_2);
    let out = unpack_interpreter_output(interpret_eclass(
        &egraph,
        &egraph[id],
        &env,
        &mut MemoizationMap::new(),
    ));
    assert_eq!(
        out,
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![1, 2], vec![3., 4.]).unwrap()
    );

    let shaped_add_test_program_1 = "(shaped-add a b)".parse().unwrap();
    let a = pack_interpreter_input(
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap(),
    );
    let b = pack_interpreter_input(
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![2, 2], vec![2., 3., -4., -5.]).unwrap(),
    );
    let mut env = Environment::new();
    env.insert("a", a);
    env.insert("b", b);
    let (egraph, id) = egg::EGraph::<MlpLanguage, ()>::from_expr(&shaped_add_test_program_1);
    let out = unpack_interpreter_output(interpret_eclass(
        &egraph,
        &egraph[id],
        &env,
        &mut MemoizationMap::new(),
    ));
    assert_eq!(
        out,
        ndarray::ArrayD::<DataType>::from_shape_vec(vec![2, 2], vec![3., 5., -1., -1.]).unwrap()
    );

    let program = "
     (shaped-map dotprod
      (cartesian-product
       (rows
        (shaped-map dotprod
         (cartesian-product
          (rows
           (shaped-map dotprod (cartesian-product (rows in)
                                           (cols w1))))
          (cols w2)
         )
        )
       )
       (cols w3)
      )
     )
     "
    .parse()
    .unwrap();

    fn load_npy(path: &str) -> ndarray::ArrayD<DataType> {
        ndarray_npy::read_npy::<_, ndarray::ArrayD<DataType>>(path).unwrap()
    }
    fn pack_interpreter_input(array: ndarray::ArrayD<DataType>) -> Value {
        Value::ShapedList(
            ndarray::ArrayD::<ListValue>::from_shape_vec(
                array.shape(),
                array
                    .iter()
                    .cloned()
                    .map(|scalar| ListValue::Scalar(scalar))
                    .collect(),
            )
            .unwrap(),
        )
    }
    let in_val = pack_interpreter_input(load_npy("in.npy"));
    let w1_val = pack_interpreter_input(load_npy("w1.npy"));
    let w2_val = pack_interpreter_input(load_npy("w2.npy"));
    let w3_val = pack_interpreter_input(load_npy("w3.npy"));
    let out_true = load_npy("out.npy");
    let mut env = Environment::new();
    env.insert("in", in_val);
    env.insert("w1", w1_val);
    env.insert("w2", w2_val);
    env.insert("w3", w3_val);
    let (egraph, id) = egg::EGraph::<MlpLanguage, Meta>::from_expr(&program);
    egraph.dot().to_svg("mlp-before-rewrites.svg").unwrap();
    let out = interpret_eclass(&egraph, &egraph[id], &env, &mut MemoizationMap::new());
    let out = unpack_interpreter_output(out);

    use approx::AbsDiffEq;
    assert!(out_true.abs_diff_eq(&out, 1e-8));

    // TODO(gus) metadata? using area?
    // TODO(gus) here's a problem: if we represent area as metadata, how do
    // we capture the different areas of different designs that might share
    // an e-class?

    let rewrite = egg::rewrite!("tensorize-dot-product";
    "(shaped-map dotprod
      (cartesian-product
       (rows ?t1)
       (cols ?t2)
      )
     )"=>"(bsg_systolic_array_weight_stationary ?t1 ?t2)");

    let rules: &[egg::Rewrite<MlpLanguage, Meta>] = &[rewrite];

    let runner = egg::Runner::new().with_expr(&program).run(&rules);

    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );
    runner
        .egraph
        .dot()
        .to_svg("mlp-after-rewrites.svg")
        .unwrap();
}

fn _dot_product() {
    egg::define_language! {
        enum DotProductLanguage {
            Num(i32),
            Mul = "*",
            For = "for",
            Sum = "sum",
            Symbol(String),
        }
    }

    // Metadata is simply the i32 value of the class, if known.
    type Meta = Option<i32>;
    impl egg::Metadata<DotProductLanguage> for Meta {
        type Error = ();

        fn merge(&self, other: &Self) -> Self {
            assert_eq!(
                self.expect("During metadata merge, expected value in self"),
                other.expect("During metadata merge, expected value in other")
            );
            *self
        }

        fn make(
            _egraph: &egg::EGraph<DotProductLanguage, Self>,
            enode: &egg::ENode<DotProductLanguage>,
        ) -> Self {
            // We only know the value in the case of a Num.
            match enode.op {
                DotProductLanguage::Num(i) => Some(i),
                _ => None,
            }
        }
    }

    #[derive(Debug)]
    struct Splitter {
        bound_var: egg::Var,
        bound: egg::Var,
        body: egg::Var,
    }

    impl egg::Applier<DotProductLanguage, Meta> for Splitter {
        fn apply_one(
            &self,
            egraph: &mut egg::EGraph<DotProductLanguage, Meta>,
            _eclass: egg::Id,
            subst: &egg::Subst,
        ) -> Vec<egg::Id> {
            // Figure out what the actual value of bound is.
            // Expect it to be known, for now.
            let bound_id: egg::Id = subst[&self.bound];
            let bound_val = &egraph[bound_id]
                .metadata
                .expect("Bound's exact value should be known, for now");

            //println!("{}", bound_val);

            let _bound_var_id: egg::Id = subst[&self.bound_var];

            // Decide how to split the loop up: by all of its factors
            // https://gist.github.com/qolop/71ef78c394db822756d58cac9993db77
            let factors = (2..*bound_val)
                .into_iter()
                .filter(|&x| bound_val % x == 0)
                .collect::<Vec<i32>>();
            for factor in factors {
                let outer_bound = factor;
                let _outer_for_loop_bound_id = egraph.add(egg::ENode::new(
                    DotProductLanguage::Num(outer_bound),
                    vec![],
                ));

                let inner_bound = bound_val / factor;
                let _inner_for_loop_bound_id = egraph.add(egg::ENode::new(
                    DotProductLanguage::Num(inner_bound),
                    vec![],
                ));

                // I want to rewrite the original body so that, anywhere where
                // the original loop var is used, the new loop var will be used.

                // TODO stopped here.
                // rewrite! {
                //     "replace-loop-var";
                //     // Searcher should just search for the initial loop var
                // }

                // What is the new loop index variable?
                // let inner_for_loop_id =
                //     egraph.add(ENode::new(DotProductLanguage::For), vec![])
            }

            vec![]
            /*
            // Get strings for sexpr
            let index: QuestionMarkName = "?index".parse().unwrap();
            let bound = "?bound".parse().unwrap();
            let body = "?body".parse().unwrap();
            // Look up node IDs
            let index = subst[&self.bound_var];
            let bound = map[&bound][0];
            let bodyid = map[&body][0];
            // Get the int from the node for bound
            let boundval = egraph[bound].nodes[0].op.clone();
            let boundval = match boundval {
                Lang::Num(val) => val,
                _ => panic!(),
            };
            // Get the index variable string
            let indexstr = egraph[index].nodes[0].op.clone();
            let indexstr = match indexstr {
                Lang::String(name) => name,
                _ => panic!(),
            };

            // I don't know if we need to get anything other than an id for the  body.

            // Get list of factors for this bound and the new index (index') to use
            let factors = get_factors(boundval);
            // let newindex = egraph.add(Expr::unit(Lang::String(indexstr + "'"))).id;

            let mut res = Vec::new();
            for fact in factors {
                if fact != 1 && fact != boundval {
                    let secondbound = boundval / fact;
                    let firstboundid = egraph.add(Expr::unit(Lang::Num(fact))).id;
                    let secondboundid = egraph.add(Expr::unit(Lang::Num(secondbound))).id;
                    let childnode =
                    // TODO does it need the ID of a child that may have already existed?
                    egraph.add(Expr::new(Lang::For, smallvec![index, secondboundid, bodyid]));
                    let newnode = egraph.add(Expr::new(
                        Lang::For,
                        smallvec![index, firstboundid, childnode.id],
                    ));
                    res.push(newnode);
                }
            }
            res
            */
        }
    }

    let dot_product_rules: &[egg::Rewrite<DotProductLanguage, Meta>] = &[
        egg::rewrite!("loop-split"; "(for ?bound_var ?bound ?body)" => {
            Splitter {
                bound_var : "?bound_var".parse().unwrap(),
                bound : "?bound".parse().unwrap(),
                body : "?body".parse().unwrap(),
            }
        }),
        egg::rewrite!("hw-dot-prod-8"; "(for ?i 8 (* (index ?a ?i) (index ?b ?i)))" => "(hw-dot-prod-8 ?a ?b)"),
        egg::rewrite!("hw-dot-prod-8"; "(for ?i 8 (* (index ?a ?i) (index ?b ?i)))" => "(hw-dot-prod-8 ?a ?b)"),
    ];

    let program = "(sum (for i 10 (* (index a i) (index b i))))"
        .parse()
        .unwrap();

    let runner = egg::Runner::new()
        .with_expr(&program)
        .run(&dot_product_rules);

    println!(
        "Stopped after {} iterations, reason: {:?}",
        runner.iterations.len(),
        runner.stop_reason
    );
    runner.egraph.dot().to_svg("dot-product.svg").unwrap();
}

// TODO implement with "residual" computations
// TODO implement also with zero padding
// TODO use fold instead of for
