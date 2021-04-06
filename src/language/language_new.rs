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

        // (dimension-definition <dimension-identifier: DimensionIdentifier>
        //                       <length: Usize>)
        "dimension-definition" = DimensionDefinition([Id; 2]),

        // (dimension-identifier-list
        //   <dimension-identifier: DimensionIdentifier>...)
        "dimension-identifier-list" = DimensionIdentifierList(Box<[Id]>),

        // (dimension-identifier <name: String>)
        "dimension-identifier" = DimensionIdentifier([Id; 1]),

        Usize(usize),
        String(String),
    }
}

#[derive(Debug, PartialEq)]
enum GlensideType {
    DimensionMap(HashMap<String, usize>),
    Dimension { name: String, length: usize },
    DimensionIdentifier(String),
    DimensionIdentifierList(Vec<String>),
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
                        GlensideType::Dimension { name, length } => (name, length),
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
                    (GlensideType::DimensionIdentifier(name), GlensideType::Usize(length)) => {
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
                    (GlensideType::String(_name), GlensideType::DimensionMap(map)) => {
                        GlensideType::DimensionMap(map.clone())
                    }
                }
            ),
            Glenside::Usize(u) => GlensideType::Usize(*u),
            Glenside::String(s) => GlensideType::String(s.clone()),
            Glenside::DimensionIdentifierList(ids) => GlensideType::DimensionIdentifierList(
                ids.iter()
                    .map(|id| match &egraph[*id].data {
                        GlensideType::DimensionIdentifier(name) => name,
                        _ => panic!(),
                    })
                    .cloned()
                    .collect::<Vec<_>>(),
            ),
            Glenside::DimensionIdentifier(ids) => make_glenside_type!(
                egraph,
                match ids {
                    (GlensideType::String(name)) => {
                        GlensideType::DimensionIdentifier(name.clone())
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
        "(dimension-definition (dimension-identifier N) 1)",
        match result {
            GlensideType::Dimension { name, length } => {
                assert_eq!(name, "N");
                assert_eq!(*length, 1);
            }
        }
    );

    test!(
        dimension_map_definition,
        "(dimension-map-definition
            (dimension-definition (dimension-identifier N) 1)
            (dimension-definition (dimension-identifier C) 3)
            (dimension-definition (dimension-identifier H) 32)
            (dimension-definition (dimension-identifier W) 64)
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
        "(dimension-identifier N)",
        match result {
            GlensideType::DimensionIdentifier(name) => {
                assert_eq!(name, "N");
            }
        }
    );

    test!(
        dimension_identifier_list,
        "(dimension-identifier-list
            (dimension-identifier N)
            (dimension-identifier C)
            (dimension-identifier H)
            (dimension-identifier W)
        )",
        match result {
            GlensideType::DimensionIdentifierList(list) => {
                assert_eq!(list, &vec!["N", "C", "H", "W"]);
            }
        }
    );
}
