use std::collections::HashMap;

use egg::{define_language, Analysis, Id};

define_language! {
    pub enum Glenside {
        // (input <name: String>
        //        <dimension-map-definition: DimensionMapDefinition>)
        "input" = Input([Id; 2]),

        // (dimension-map-defintion
        //   <dimension-definition: DimensionDefinition>...)
        "dimension-map-definition" = DimensionMapDefinition(Box<[Id]>),

        // (dimension-definition <dimension-identifier: String>
        //                       <length: Usize>)
        "dimension-definition" = DimensionDefinition([Id; 2]),

        // (dimension-list
        //   <dimension-identifier: String>...)
        "dimension-list" = DimensionList(Box<[Id]>),

        // (pair <i0> <i1>)
        //
        // Pair dimensions of two inputs. Dimensions whose identifiers are
        // equivalent are paired, non-equivalent dimensions are left alone.
        "pair" = Pair([Id; 2]),

        Usize(usize),
        String(String),
    }
}

#[derive(Debug, PartialEq)]
enum GlensideType {
    DimensionMap(HashMap<String, usize>),
    DimensionDefinition { name: String, length: usize },
    DimensionList(Vec<String>),
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
            Glenside::DimensionMapDefinition(ids) => {
                let mut out = HashMap::new();

                for (k, v) in ids
                    .iter()
                    .map(|id| match &egraph[*id].data {
                        GlensideType::DimensionDefinition { name, length } => (name, length),
                        _ => panic!(),
                    })
                    .collect::<Vec<_>>()
                {
                    out.insert(k.clone(), *v);
                }

                GlensideType::DimensionMap(out)
            }
            Glenside::DimensionDefinition(ids) => make_glenside_type!(
                egraph,
                match ids {
                    // In the future, dimension identifiers may not all be
                    // Strings.
                    (GlensideType::String(name), GlensideType::Usize(length)) => {
                        GlensideType::DimensionDefinition {
                            name: name.clone(),
                            length: *length,
                        }
                    }
                }
            ),
            Glenside::Input(ids) => make_glenside_type!(
                egraph,
                match ids {
                    (GlensideType::String(_name), GlensideType::DimensionMap(map)) => {
                        GlensideType::DimensionMap(map.clone())
                    }
                }
            ),
            Glenside::Usize(u) => GlensideType::Usize(*u),
            Glenside::String(s) => GlensideType::String(s.clone()),
            Glenside::DimensionList(ids) => GlensideType::DimensionList(
                ids.iter()
                    .map(|id| match &egraph[*id].data {
                        // In the future, dimension identifiers may not all be
                        // Strings.
                        GlensideType::String(name) => name,
                        _ => panic!(),
                    })
                    .cloned()
                    .collect::<Vec<_>>(),
            ),
            Glenside::Pair(ids) => make_glenside_type!(
                egraph,
                match ids {
                    (
                        GlensideType::DimensionMap(in_dims_0),
                        GlensideType::DimensionMap(in_dims_1),
                    ) => {
                        let mut out_map = HashMap::new();

                        for (dim_name, dim_len) in in_dims_0.iter() {
                            out_map.insert(dim_name.clone(), *dim_len);
                        }

                        for (dim_name, dim_len) in in_dims_1.iter() {
                            if out_map.contains_key(dim_name) {
                                assert_eq!(out_map.get(dim_name).unwrap(), dim_len);
                            } else {
                                out_map.insert(dim_name.clone(), *dim_len);
                            }
                        }

                        // Add tuple dimension
                        // TODO(@gussmith23) How to name the tuple dim?
                        // For now just going with "T" by default.
                        assert!(!out_map.contains_key("T"));
                        out_map.insert("T".to_string(), 2);

                        GlensideType::DimensionMap(out_map)
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
        dimension_definition,
        "(dimension-definition N 1)",
        match result {
            GlensideType::DimensionDefinition { name, length } => {
                assert_eq!(name, "N");
                assert_eq!(*length, 1);
            }
        }
    );

    test!(
        dimension_map_definition,
        "(dimension-map-definition
            (dimension-definition N 1)
            (dimension-definition C 3)
            (dimension-definition H 32)
            (dimension-definition W 64)
        )",
        match result {
            GlensideType::DimensionMap(map) => {
                assert_eq!(map.len(), 4);
                assert_eq!(map.get(&"N".to_string()), Some(&1));
                assert_eq!(map.get(&"C".to_string()), Some(&3));
                assert_eq!(map.get(&"H".to_string()), Some(&32));
                assert_eq!(map.get(&"W".to_string()), Some(&64));
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
        dimension_identifier_list,
        "(dimension-list N C H W)",
        match result {
            GlensideType::DimensionList(list) => {
                assert_eq!(list, &vec!["N", "C", "H", "W"]);
            }
        }
    );

    test!(
        pair,
        "
        (pair
         (input
          input_MxN
          (dimension-map-definition
           (dimension-definition M 16)
           (dimension-definition N 32)
          )
         )
         (input
          input_NxO
          (dimension-map-definition
           (dimension-definition N 32)
           (dimension-definition O 64)
          )
         )
        )",
        match result {
            GlensideType::DimensionMap(map) => {
                assert_eq!(map.len(), 4);
                assert_eq!(map.get(&"M".to_string()), Some(&16));
                assert_eq!(map.get(&"N".to_string()), Some(&32));
                assert_eq!(map.get(&"O".to_string()), Some(&64));
                assert_eq!(map.get(&"T".to_string()), Some(&2));
            }
        }
    );
}
