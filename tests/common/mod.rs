use ndarray_npy::ReadableElement;

#[cfg(feature = "ndarray-npy")]
pub fn load_npy<DataType: ReadableElement>(path: &str) -> ndarray::ArrayD<DataType> {
    ndarray_npy::read_npy::<_, ndarray::ArrayD<DataType>>(path).unwrap()
}
