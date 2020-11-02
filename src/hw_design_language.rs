use serde::Serialize;
use serde_json::map::Map;
use serde_json::{json, Value};

#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
pub enum DType {
    Int8 = 0,
    Int16 = 1,
    Int32 = 2,
    Uint8 = 3,
    Uint16 = 4,
    Uint32 = 5,
    Bf16 = 6,
    Fp16 = 7,
    Fp32 = 8,
    Fp64 = 9,
}

impl DType {
    /// Return the equivalent C type name for a given `dtype`, as a String.
    /// ```
    /// use glenside::hw_design_language::*;
    /// assert_eq!(DType::Fp32.to_c_type_string(), "float");
    /// ```
    pub fn to_c_type_string(&self) -> String {
        match &self {
            &DType::Fp32 => "float",
            &DType::Int32 => "int",
            _ => panic!(),
        }
        .to_string()
    }
}

pub struct SystolicArrayWeightStationaryParams {
    pub dtype: DType,
    pub rows: usize,
    pub cols: usize,
}

pub enum AtomConfig {
    SystolicArrayWeightStationary(SystolicArrayWeightStationaryParams),
}

pub struct Atom {
    pub name: String,
    pub id: usize,
    pub config: AtomConfig,
}

pub struct HardwareDesign {
    pub atoms: Vec<Atom>,
}

pub fn design_to_json(design: &HardwareDesign) -> Value {
    Value::Array(
        design
            .atoms
            .iter()
            .map(|atom: &Atom| atom_to_json(atom))
            .collect(),
    )
}

pub fn atom_to_json(atom: &Atom) -> Value {
    let mut map = Map::default();
    map.append(&mut match &atom.config {
        AtomConfig::SystolicArrayWeightStationary(params) => {
            let mut map = Map::default();
            map.insert(
                "atom".to_string(),
                json!("bsg_systolic_array_weight_stationary"),
            );
            map.insert("dtype".to_string(), json!(params.dtype));
            map.insert("rows".to_string(), json!(params.rows));
            map.insert("cols".to_string(), json!(params.cols));
            map
        }
    });
    map.insert("name".to_string(), json!(atom.name));
    map.insert("id".to_string(), json!(atom.id));

    Value::Object(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serialize() {
        let design = HardwareDesign {
            atoms: vec![
                Atom {
                    name: "multiplier1".to_string(),
                    id: 1,
                    config: AtomConfig::SystolicArrayWeightStationary(
                        SystolicArrayWeightStationaryParams {
                            dtype: DType::Int8,
                            rows: 16,
                            cols: 16,
                        },
                    ),
                },
                Atom {
                    name: "multiplier2".to_string(),
                    id: 2,
                    config: AtomConfig::SystolicArrayWeightStationary(
                        SystolicArrayWeightStationaryParams {
                            dtype: DType::Int8,
                            rows: 16,
                            cols: 16,
                        },
                    ),
                },
                Atom {
                    name: "multiplier3".to_string(),
                    id: 3,
                    config: AtomConfig::SystolicArrayWeightStationary(
                        SystolicArrayWeightStationaryParams {
                            dtype: DType::Int8,
                            rows: 16,
                            cols: 16,
                        },
                    ),
                },
            ],
        };

        assert_eq!(
            design_to_json(&design),
            json!(
                [
                    { "name" : "multiplier1",
                       "atom" : "bsg_systolic_array_weight_stationary",
                       "id" : 1,
                       "dtype" : "int8",
                       "cols" : 16,
                       "rows" : 16,
                    },
                    { "name" : "multiplier2",
                       "atom" : "bsg_systolic_array_weight_stationary",
                       "id" : 2,
                       "dtype" : "int8",
                       "cols" : 16,
                       "rows" : 16,
                    },
                    { "name" : "multiplier3",
                       "atom" : "bsg_systolic_array_weight_stationary",
                       "id" : 3,
                       "dtype" : "int8",
                       "cols" : 16,
                       "rows" : 16,
                    },
                ]
            )
        );
    }
}
