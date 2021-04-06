use std::collections::HashMap;

use egg::{define_language, Analysis, Id};

define_language! {
    pub enum Glenside {
        // (input <name: String> <dimension-set: DimensionSet>)
        "input" = Input([Id; 2]),

        // (dimension-set <dimension: Dimension>...)
        "dimension-set" = DimensionSet(Box<[Id]>),

        // (dimension <name: String> <length: Usize>)
        "dimension" = Dimension([Id; 2]),

        Usize(usize),
        String(String),
    }
}

#[derive(Debug, PartialEq)]
enum GlensideType {
    DimensionSet(HashMap<String, usize>),
    Dimension { name: String, length: usize },
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
            Glenside::DimensionSet(ids) => {
                let mut out = HashMap::new();

                for (k, v) in ids
                    .iter()
                    .map(|id| match &egraph[*id].data {
                        GlensideType::Dimension { name, length } => (name, length),
                        _ => panic!(),
                    })
                    .collect::<Vec<_>>()
                {
                    out.insert(k.clone(), *v);
                }

                GlensideType::DimensionSet(out)
            }
            Glenside::Dimension(ids) => make_glenside_type!(
                egraph,
                match ids {
                    (GlensideType::String(name), GlensideType::Usize(length)) => {
                        GlensideType::Dimension {
                            name: name.clone(),
                            length: *length,
                        }
                    }
                }
            ),
            Glenside::Input(ids) => make_glenside_type!(
                egraph,
                match ids {
                    (GlensideType::String(_name), GlensideType::DimensionSet(map)) => {
                        GlensideType::DimensionSet(map.clone())
                    }
                }
            ),
            Glenside::Usize(u) => GlensideType::Usize(*u),
            Glenside::String(s) => GlensideType::String(s.clone()),
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
        "(dimension N 1)",
        match result {
            GlensideType::Dimension { name, length } => {
                assert_eq!(name, "N");
                assert_eq!(*length, 1);
            }
        }
    );

    test!(
        dimension_set,
        "(dimension-set (dimension N 1) (dimension C 3) (dimension H 32) (dimension W 64))",
        match result {
            GlensideType::DimensionSet(map) => {
                assert_eq!(map.len(), 4);
                assert_eq!(map.get(&"N".to_string()), Some(&1));
                assert_eq!(map.get(&"C".to_string()), Some(&3));
                assert_eq!(map.get(&"H".to_string()), Some(&32));
                assert_eq!(map.get(&"W".to_string()), Some(&64));
            }
        }
    );
}
