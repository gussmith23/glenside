#![feature(is_sorted)]
#![feature(test)]
//! ## Glenside Feature Flags
//!
//! Glenside generally leans towards *enabling* all features by default, purely
//! because it makes testing on my machine (where I have all of the dependencies
//! set up) easier. To disable glenside features, you should use
//! `--no-default-features` and enable features manually.
//!
//! Parts of Glenside (namely extraction; see [`glenside::extraction`]) rely on
//! IBM's CPLEX solver. You can disable these components by not enabling
//! Glenside's `cplex` feature.

pub mod codegen;
pub mod extraction;
pub mod hw_design_language;
pub mod language;
