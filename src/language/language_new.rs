use std::collections::HashSet;
use std::iter::FromIterator;

use egg::{define_language, Analysis, Id};

define_language! {
    pub enum Glenside {
        // (apply <program> <args>)
        // Apply a program represented in the egraph to concrete arguments.
        "apply" = Apply([Id; 2]),

        // (input <name: String>
        //        <dimension-set: DimensionSet>)
        "input" = Input([Id; 2]),

        // (d <dimension-identifier: String> <length: Usize>)
        //
        // The definition of a dimension. A dimension is uniquely identified by
        // some identifier, and a length.
        "d" = Dimension([Id; 2]),

        // (dimension-set <dimension-identifier: Id>...)
        //
        // A set of dimension identifiers. Should not contain duplicates. In its
        // canonical form, it should be sorted.
        //
        // TODO(@gussmith23) Make a rewrite to convert to canonical form.
        "ds" = DimensionSet(Box<[Id]>),

        // (dot-product <i0> <i1>)
        // Reduce identical dimensions using dot product. In the result, only
        // the non-shared dimensions will remain.
        "dot-product" = DotProduct([Id; 2]),

        Usize(usize),
        String(String),
    }
}

pub type Dimension = (String, usize);

// Inputs should have single type, one set of dimensions.
// functions should be like {set1} -> {set2} -> {set3}
// for example, and given set1 and set2, results in type set3.
struct TypePayload {}

#[derive(Debug, PartialEq)]
enum GlensideType {
    ///
    Type(),
    /// Represents a function taking in specific dimensions and outputting a new
    /// dimension set.
    Function {
        inputs: HashSet<Id>,
        output: Id,
    },
    /// Uniquely identifies a dimension. A dimension is an `(id, length)` pair,
    /// where `id` is an [`Id`] pointing to an eclass that uniquely identifies
    /// the dimension (for now, just a String).
    /// TODO(@gussmith23) Explain why we need length.
    Dimension(Dimension),
    DimensionSet(HashSet<Dimension>),

    Node {
        /// The unbound/in
        unbound: HashSet<Id>,
    },

    Usize(usize),
    String(String),
}

struct GlensideTypeAnalysis;
impl Analysis<Glenside> for GlensideTypeAnalysis {
    type Data = GlensideType;

    fn make(egraph: &egg::EGraph<Glenside, Self>, enode: &Glenside) -> Self::Data {
        /// Helper macro for defining match arms.
        ///
        /// It's formatted this way so that we can use rustfmt to format macro
        /// invocations!
        macro_rules! make_glenside_type {
            ($egraph:ident, match $ids:ident { ( $($p:pat),* $(,)? ) => $body:expr } ) => {
                {{
                    match $ids.iter().map(|id| &$egraph[*id].data).collect::<Vec<_>>().as_slice() {
                        [$($p),*] => $body,
                        _ => panic!("Does not type check"),
                    }
                }}
            };
        }

        match &enode {
            Glenside::Apply(_ids) => todo!(),
            Glenside::Dimension(ids) => make_glenside_type!(
                egraph,
                match ids {
                    // In the future, dimension identifiers may not all be
                    // Strings.
                    (GlensideType::String(name), GlensideType::Usize(length)) => {
                        GlensideType::Dimension((name.clone(), *length))
                    }
                }
            ),
            Glenside::Input(ids) => make_glenside_type!(
                egraph,
                match ids {
                    (GlensideType::String(_name), GlensideType::DimensionSet(set)) => {
                        GlensideType::DimensionSet(set.clone())
                    }
                }
            ),
            Glenside::Usize(u) => GlensideType::Usize(*u),
            Glenside::String(s) => GlensideType::String(s.clone()),
            Glenside::DimensionSet(ids) => {
                let mut vec = ids.to_vec();
                vec.sort_unstable();
                let len_before_dedup = vec.len();
                vec.dedup();
                assert_eq!(vec.len(), len_before_dedup, "Set contained duplicates");

                GlensideType::DimensionSet(HashSet::from_iter(vec.drain(..).map(|id| {
                    match &egraph[id].data {
                        GlensideType::Dimension(d) => d.clone(),
                        _ => panic!(),
                    }
                })))
            }
            Glenside::DotProduct(ids) => make_glenside_type!(
                egraph,
                match ids {
                    (GlensideType::DimensionSet(i0), GlensideType::DimensionSet(i1)) => {
                        GlensideType::DimensionSet(i0.symmetric_difference(i1).cloned().collect())
                    }
                }
            ),
        }
    }

    fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
        assert_eq!(*to, from);
        egg::merge_if_different(to, from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! test {
        ($name:ident, $glenside_str:literal, match result { $result_pat:pat => $check_block:expr }) => {
            #[test]
            fn $name() {
                let mut egraph =
                    egg::EGraph::<Glenside, GlensideTypeAnalysis>::new(GlensideTypeAnalysis);
                let id = egraph.add_expr(&$glenside_str.parse().unwrap());
                match &egraph[id].data {
                    $result_pat => $check_block,
                    _ => panic!(),
                }
            }
        };
    }

    test!(
        dimension,
        "(d N 1)",
        match result {
            GlensideType::Dimension((name, length)) => {
                assert_eq!(name, "N");
                assert_eq!(*length, 1);
            }
        }
    );

    test!(
        dimension_set,
        "(ds
          (d N 1)
          (d C 3)
          (d H 32)
          (d W 64)
         )",
        match result {
            GlensideType::DimensionSet(set) => {
                assert_eq!(
                    set,
                    &HashSet::from_iter(
                        vec![
                            ("N".to_string(), 1),
                            ("C".to_string(), 3),
                            ("H".to_string(), 32),
                            ("W".to_string(), 64),
                        ]
                        .drain(..)
                    )
                )
            }
        }
    );

    test!(
        dimension_identifier,
        "N",
        match result {
            GlensideType::String(name) => {
                assert_eq!(name, "N");
            }
        }
    );

    test!(
        dot_product,
        "
        (dot-product
         (input
          input_MxN
          (ds
           (d M 16)
           (d N 32)
          )
         )
         (input
          input_NxO
          (ds
           (d N 32)
           (d O 64)
          )
         )
        )",
        match result {
            GlensideType::DimensionSet(set) => {
                assert_eq!(
                    set,
                    &HashSet::from_iter(
                        vec![("M".to_string(), 16), ("O".to_string(), 64),].drain(..)
                    )
                )
            }
        }
    );
}
