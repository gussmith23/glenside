use egg::{define_language, Id};

define_language! {
    pub enum GlensideLanguage {
        // (access <op> <params> <args>)
        // Applies access operator <op> with parameters <params> to <args>.
        "access" = Access([Id; 3]),

        // (access-input <dimension-set: DimensionSet>)
        // Generates an access pattern.
        "access-input" = AccessInput([Id; 1]),

        // (dimension-set <dim: Dimension>...)
        // Set of dimensions (unordered).
        "dimension-set" = DimensionSet(Box<[Id]>),

        // Dimensions are symbols starting with d.
        // E.g. dN, dC, dH, dW.
        // TODO(@gussmith23) This could bite us down the road if we are trying
        // to name stuff with "d".
        Dimension(Dimension),
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Clone)]
pub struct Dimension(String);
impl std::fmt::Display for Dimension {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "d{}", self.0)
    }
}
impl std::str::FromStr for Dimension {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        assert!(s.len() > 1, "dimension must have a name");
        assert_eq!(
            s.chars().nth(0).unwrap(),
            'd',
            "dimension must start with \"d\""
        );
        Ok(Self(s[1..].to_string()))
    }
}
