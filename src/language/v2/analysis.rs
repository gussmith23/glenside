use super::language::{GlensideLanguage, *};
use egg::{Analysis, EGraph, Id};
use std::collections::HashSet;
use std::iter::FromIterator;

type EG = EGraph<GlensideLanguage, GlensideAnalysis>;

type DimensionSet = HashSet<Dimension>;

#[derive(Debug, PartialEq)]
pub struct AccessPatternData {
    dimensions: DimensionSet,
}

#[derive(Debug, PartialEq)]
pub enum GlensideAnalysis {
    AccessPattern(AccessPatternData),
    DimensionSet(DimensionSet),
    Dimension(Dimension),
}

impl Analysis<GlensideLanguage> for GlensideAnalysis {
    type Data = GlensideAnalysis;

    fn merge(&self, to: &mut Self::Data, from: Self::Data) -> bool {
        assert_eq!(*to, from);
        egg::merge_if_different(to, from)
    }

    fn make(egraph: &EG, enode: &GlensideLanguage) -> Self::Data {
        use GlensideLanguage::*;
        match enode {
            AccessInput([dim_set_id]) => {
                let dim_set = match &egraph[*dim_set_id].data {
                    GlensideAnalysis::DimensionSet(s) => s,
                    _ => panic!("expected dimension set"),
                };

                GlensideAnalysis::AccessPattern(AccessPatternData {
                    dimensions: dim_set.clone(),
                })
            }
            Access(_) => todo!(),
            Dimension(_) => todo!(),
            DimensionSet(dims) => {
                GlensideAnalysis::DimensionSet(HashSet::from_iter(dims.iter().map(|&id: &Id| {
                    match &egraph[id].data {
                        GlensideAnalysis::Dimension(d) => d.clone(),
                        _ => panic!("expected dimension"),
                    }
                })))
            }
        }
    }
}
