use super::*;
use log::{debug, info};
use ndarray::s;
use std::collections::hash_map::HashMap;

type Environment<'a> = HashMap<&'a str, Value>;

type DataType = f64;

// TODO(gus) not sure this should actually be pub; I'm being lazy
/// Values are `ndarray::ArrayD` tensors or `usize`s.
#[derive(Clone, PartialEq, Debug)]
pub enum Value {
    Tensor(ndarray::ArrayD<DataType>),
    Usize(usize),
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
            Value::Tensor(t1) => match other {
                Value::Tensor(t2) => Value::Tensor(t1 + t2),
                _ => panic!(),
            },
            Value::Usize(u1) => match other {
                Value::Usize(u2) => Value::Usize(u1 + u2),
                _ => panic!(),
            },
        }
    }
}

// TODO(gus) not sure this should actually be pub; I'm being lazy
#[derive(Clone, Debug)]
pub enum MemoizedInterpreterResult {
    InterpretedValue(Value),
    StillInterpreting,
    // When an eclass cannot be interpreted, because all of its enodes
    // recursively depend on themselves.
    CanNotInterpret,
}

type MemoizationMap = std::collections::HashMap<egg::Id, MemoizedInterpreterResult>;

fn interpret_eclass<M: egg::Metadata<Language>>(
    egraph: &egg::EGraph<Language, M>,
    eclass: &egg::EClass<Language, M>,
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
        .map(|enode: &egg::ENode<Language>| interpret_enode(egraph, enode, env, &mut memo_map))
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
            .map(|enode: &egg::ENode<Language>| interpret_enode(egraph, enode, env, &mut memo_map))
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

fn interpret_enode<M: egg::Metadata<Language>>(
    egraph: &egg::EGraph<Language, M>,
    enode: &egg::ENode<Language>,
    env: &Environment,
    memo_map: &mut MemoizationMap,
) -> MemoizedInterpreterResult {
    use language::Language::*;
    debug!("interpreting enode: {:?}", enode);
    match &enode.op {
        Symbol(name) => {
            debug!("interpreting symbol");
            MemoizedInterpreterResult::InterpretedValue(env[&name[..]].clone())
        }
        MapDotProduct => {
            assert_eq!(enode.children.len(), 1);

            // Get the argument as a tensor.
            let arg: MemoizedInterpreterResult =
                interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map);
            let arg: Value = match arg {
                // Return early if we've found a recursive cycle
                MemoizedInterpreterResult::StillInterpreting => {
                    return MemoizedInterpreterResult::StillInterpreting
                }
                MemoizedInterpreterResult::CanNotInterpret => {
                    return MemoizedInterpreterResult::CanNotInterpret
                }
                MemoizedInterpreterResult::InterpretedValue(v) => v,
            };
            let arg: ndarray::ArrayD<DataType> = match arg {
                Value::Tensor(t) => t,
                _ => panic!(),
            };

            // Check that it has the correct shape.
            // Basically, we're looking for a shape:
            // [d1, ... , dn, 2, v]
            // where v is the length of the vectors to be dot-prodded together.
            // The resulting shape will be
            // [d1, ... , dn]
            let initial_shape = arg.shape();
            assert_eq!(initial_shape[arg.shape().len() - 2], 2);

            fn map_dot_product<T: ndarray::LinalgScalar + Copy>(
                t: ndarray::ArrayViewD<T>,
            ) -> ndarray::ArrayD<T> {
                if t.shape().len() > 2 {
                    // TODO(gus) Definitely not performant
                    let to_be_stacked = t
                        .axis_iter(ndarray::Axis(0))
                        .map(map_dot_product)
                        .collect::<Vec<ndarray::ArrayD<T>>>();
                    // Not sure if this should ever be the case.
                    assert!(to_be_stacked.len() > 0);
                    if to_be_stacked[0].shape().len() == 0 {
                        use std::iter::FromIterator;
                        ndarray::ArrayBase::from_iter(to_be_stacked.iter().cloned().map(|t| {
                            t.into_dimensionality::<ndarray::Ix0>()
                                .unwrap()
                                .into_scalar()
                        }))
                        .into_dyn()
                    } else {
                        let mut to_be_stacked = to_be_stacked
                            .iter()
                            .map(|t| t.view())
                            .collect::<Vec<ndarray::ArrayViewD<T>>>();
                        ndarray::stack(ndarray::Axis(0), &to_be_stacked[..]).unwrap()
                    }
                } else {
                    assert_eq!(t.shape()[0], 2);

                    let v1: ndarray::Array1<T> = t
                        .index_axis(ndarray::Axis(0), 0)
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix1>()
                        .unwrap();
                    let v2: ndarray::Array1<T> = t
                        .index_axis(ndarray::Axis(0), 1)
                        .to_owned()
                        .into_dimensionality::<ndarray::Ix1>()
                        .unwrap();
                    // TODO(gus) left off here, no idea how to fix the current
                    // compiler error
                    ndarray::arr0(v1.dot(&v2)).into_dyn()
                }
            }

            MemoizedInterpreterResult::InterpretedValue(Value::Tensor(map_dot_product(arg.view())))
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
            let arg_as_tensor: ndarray::ArrayD<DataType> = match arg_as_tensor {
                Value::Tensor(t) => t,
                _ => panic!(),
            };

            MemoizedInterpreterResult::InterpretedValue(Value::Tensor(arg_as_tensor))
        }
        Cols => {
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
            let arg_as_tensor: ndarray::ArrayD<DataType> = match arg_as_tensor {
                Value::Tensor(t) => t,
                _ => panic!(),
            };

            let transpose = arg_as_tensor.t().to_owned();

            MemoizedInterpreterResult::InterpretedValue(Value::Tensor(transpose))
        }
        CartesianProduct => {
            // Semantics of cartesian product:
            // Rightmost thing varies the fastest.

            // There should be two args, both of which should be lists.
            assert_eq!(enode.children.len(), 2);
            let mut left: ndarray::ArrayD<DataType> =
                match interpret_eclass(egraph, &egraph[enode.children[0]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::Tensor(t) => t,
                        _ => panic!(),
                    },
                };
            let mut right: ndarray::ArrayD<DataType> =
                match interpret_eclass(egraph, &egraph[enode.children[1]], env, memo_map) {
                    MemoizedInterpreterResult::StillInterpreting => {
                        return MemoizedInterpreterResult::StillInterpreting
                    }
                    MemoizedInterpreterResult::CanNotInterpret => {
                        return MemoizedInterpreterResult::CanNotInterpret
                    }
                    MemoizedInterpreterResult::InterpretedValue(v) => match v {
                        Value::Tensor(t) => t,
                        _ => panic!(),
                    },
                };

            assert!(left.shape().len() > 0);
            assert!(right.shape().len() > 0);
            assert_eq!(left.shape().last().unwrap(), right.shape().last().unwrap());

            if left.shape().len() > 2 || right.shape().len() > 2 {
                todo!("Cartesian product not implemented for more than 2 dimensions");
            }

            let new_shape = left
                .shape()
                .iter()
                .take(left.shape().len() - 1)
                .chain(right.shape().iter().take(right.shape().len() - 1))
                .chain(&[2, *left.shape().last().unwrap()])
                .cloned()
                .collect::<Vec<ndarray::Ix>>();

            //let new_shape = vec![left.shape()[0], right.shape()[0], 2, left.shape()[1]];

            // TODO(gus) this whole chain of things is definitely not performant
            if left.shape().len() == 1 {
                left = left.insert_axis(ndarray::Axis(0));
            }

            if right.shape().len() == 1 {
                right = right.insert_axis(ndarray::Axis(0));
            }

            let left = left.axis_iter(ndarray::Axis(0));
            let right = right.axis_iter(ndarray::Axis(0));
            use itertools::iproduct;
            let to_be_reshaped = iproduct!(left, right);
            let to_be_reshaped = to_be_reshaped.map(|tuple| {
                ndarray::stack(ndarray::Axis(0), &[tuple.0.view(), tuple.1.view()]).unwrap()
            });
            let vec = to_be_reshaped.collect::<Vec<ndarray::ArrayD<DataType>>>();
            let vec = vec
                .iter()
                .map(|t| t.view())
                .collect::<Vec<ndarray::ArrayViewD<DataType>>>();

            let to_be_reshaped: ndarray::ArrayD<DataType> =
                ndarray::stack(ndarray::Axis(0), &vec[..]).unwrap();

            // TODO(gus) probably unnecessary cloning happening here.
            let reshaped_into_tensor: ndarray::ArrayD<DataType> =
                ndarray::ArrayD::<DataType>::from_shape_vec(
                    new_shape,
                    to_be_reshaped.iter().cloned().collect::<Vec<DataType>>(),
                )
                .unwrap();

            MemoizedInterpreterResult::InterpretedValue(Value::Tensor(reshaped_into_tensor))
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
                        Value::Tensor(t) => t,
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
            MemoizedInterpreterResult::InterpretedValue(Value::Tensor(
                tensor
                    .slice_move(s![
                        row_slice_start..row_slice_end,
                        col_slice_start..col_slice_end
                    ])
                    .into_dyn(),
            ))
        }
        Usize(u) => MemoizedInterpreterResult::InterpretedValue(Value::Usize(*u)),
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
                        Value::Tensor(t) => t,
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
            // TODO(gus) given that i changed to a simpler value system, I can
            // actually go back to stacking with their code.

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

            MemoizedInterpreterResult::InterpretedValue(Value::Tensor(new_tensor))
        }
    }
}
pub fn pack_interpreter_input(array: ndarray::ArrayD<DataType>) -> Value {
    Value::Tensor(array)
}
pub fn unpack_interpreter_output(output: MemoizedInterpreterResult) -> ndarray::ArrayD<DataType> {
    match output {
        MemoizedInterpreterResult::InterpretedValue(v) => match v {
            Value::Tensor(t) => t,
            _ => panic!(),
        },
        _ => panic!(),
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    fn load_npy(path: &str) -> ndarray::ArrayD<DataType> {
        ndarray_npy::read_npy::<_, ndarray::ArrayD<DataType>>(path).unwrap()
    }

    #[test]
    fn test_slice() {
        let slice_test_program_1 = "(slice a 0 1 0 1)".parse().unwrap();
        let slice_test_program_2 = "(slice a 1 2 0 2)".parse().unwrap();
        let input = pack_interpreter_input(
            ndarray::ArrayD::<DataType>::from_shape_vec(vec![2, 2], vec![1., 2., 3., 4.]).unwrap(),
        );
        let mut env = Environment::new();
        env.insert("a", input);
        let (egraph, id) = egg::EGraph::<Language, ()>::from_expr(&slice_test_program_1);
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
        let (egraph, id) = egg::EGraph::<Language, ()>::from_expr(&slice_test_program_2);
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
    }

    #[test]
    fn test_mlp() {
        let program = "
     (map-dot-product
      (cartesian-product
       (rows
        (map-dot-product
         (cartesian-product
          (rows
           (map-dot-product (cartesian-product (rows in)
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
        let in_val = pack_interpreter_input(load_npy("data/in.npy"));
        let w1_val = pack_interpreter_input(load_npy("data/w1.npy"));
        let w2_val = pack_interpreter_input(load_npy("data/w2.npy"));
        let w3_val = pack_interpreter_input(load_npy("data/w3.npy"));
        let out_true = load_npy("data/out.npy");
        let mut env = Environment::new();
        env.insert("in", in_val);
        env.insert("w1", w1_val);
        env.insert("w2", w2_val);
        env.insert("w3", w3_val);
        let (egraph, id) = egg::EGraph::<Language, Meta>::from_expr(&program);
        let out = interpret_eclass(&egraph, &egraph[id], &env, &mut MemoizationMap::new());
        let out = unpack_interpreter_output(out);

        use approx::AbsDiffEq;
        println!("{:?}", out);
        println!("{:?}", out_true);
        assert!(out_true.abs_diff_eq(&out, 1e-8));
    }
}
