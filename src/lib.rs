#![feature(test)]
//! Parts of Glenside (namely extraction; see [`glenside::extraction`]) rely on
//! IBM's CPLEX solver. You can disable these components by not enabling
//! Glenside's `cplex` feature.

pub mod codegen;
pub mod extraction;
pub mod hw_design_language;
pub mod language;
