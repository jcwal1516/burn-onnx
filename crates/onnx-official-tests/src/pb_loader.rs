//! Load `TensorProto` `.pb` files shipped with upstream ONNX node tests.
//!
//! ONNX backend tests serialize each input/output tensor as a single
//! `TensorProto` message. This module decodes those files into plain
//! `(shape, Vec<f32>)` tuples for FLOAT tensors, which is all the
//! current scaffold needs. Other dtypes will be added as coverage
//! widens.

use onnx_ir::TensorProto;
use protobuf::Message;
use std::path::{Path, PathBuf};
use thiserror::Error;

/// `TensorProto.DataType` enum value for an undefined / unset dtype.
/// Per the ONNX spec this signals a malformed or default-constructed
/// proto, not a dtype the consumer hasn't implemented.
const DATA_TYPE_UNDEFINED: i32 = 0;
/// `TensorProto.DataType` enum value for FLOAT (32-bit IEEE 754).
const DATA_TYPE_FLOAT: i32 = 1;

/// A decoded FLOAT tensor: row-major dimensions and a flat value buffer.
#[derive(Debug, Clone)]
pub struct FloatTensor {
    pub shape: Vec<usize>,
    pub values: Vec<f32>,
}

impl FloatTensor {
    /// Total number of scalar elements implied by `shape`.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Errors that can occur while loading a `.pb` reference tensor.
///
/// `path` fields are formatted with `{:?}` because `PathBuf` does not
/// implement `Display`; the Debug form is human-readable enough for an
/// error message and avoids pulling in a wrapper type.
#[derive(Debug, Error)]
pub enum LoadError {
    #[error("io error reading {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("protobuf decode error in {path:?}: {source}")]
    Proto {
        path: PathBuf,
        #[source]
        source: protobuf::Error,
    },
    #[error("{path:?}: TensorProto has UNDEFINED data_type, file is likely malformed or truncated")]
    UndefinedDataType { path: PathBuf },
    #[error(
        "{path:?}: unsupported TensorProto data_type {data_type} (only FLOAT=1 is currently handled)"
    )]
    UnsupportedDataType { path: PathBuf, data_type: i32 },
    #[error(
        "{path:?}: TensorProto dim[{index}] = {dim} cannot be converted to usize \
         (must be non-negative and fit in the host pointer width)"
    )]
    InvalidDimension {
        path: PathBuf,
        index: usize,
        dim: i64,
    },
    #[error("{path:?}: shape {dims:?} has an element count that overflows usize")]
    ShapeOverflow { path: PathBuf, dims: Vec<i64> },
    #[error(
        "{path:?}: raw_data has {len} bytes, which is not a multiple of 4 \
         and cannot be a buffer of f32 values"
    )]
    UnalignedRawData { path: PathBuf, len: usize },
    #[error(
        "{path:?}: tensor element count mismatch: shape {shape:?} implies {expected} elements, \
         decoded {actual}"
    )]
    LengthMismatch {
        path: PathBuf,
        shape: Vec<usize>,
        expected: usize,
        actual: usize,
    },
}

/// Decode a single `TensorProto` `.pb` file as a FLOAT tensor.
///
/// Per the ONNX spec, `TensorProto` stores values in one of two
/// mutually exclusive places: the type-specific field (`float_data` for
/// FLOAT) or the dtype-agnostic `raw_data` field, which holds values in
/// fixed-width little-endian byte order. Upstream backend tests
/// empirically use `raw_data`, but we accept either to keep the loader
/// general.
pub fn load_float_tensor(path: &Path) -> Result<FloatTensor, LoadError> {
    let bytes = std::fs::read(path).map_err(|source| LoadError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let proto = TensorProto::parse_from_bytes(&bytes).map_err(|source| LoadError::Proto {
        path: path.to_path_buf(),
        source,
    })?;

    match proto.data_type {
        DATA_TYPE_FLOAT => {}
        DATA_TYPE_UNDEFINED => {
            return Err(LoadError::UndefinedDataType {
                path: path.to_path_buf(),
            });
        }
        other => {
            return Err(LoadError::UnsupportedDataType {
                path: path.to_path_buf(),
                data_type: other,
            });
        }
    }

    // Validate each dim individually so a negative value (e.g. from a
    // malformed proto) does not wrap into a huge usize, and use checked
    // multiplication so the element count cannot overflow silently into
    // a small number that subsequently passes the length check.
    let mut shape: Vec<usize> = Vec::with_capacity(proto.dims.len());
    let mut expected: usize = 1;
    for (index, &dim) in proto.dims.iter().enumerate() {
        let dim_usize = usize::try_from(dim).map_err(|_| LoadError::InvalidDimension {
            path: path.to_path_buf(),
            index,
            dim,
        })?;
        expected = expected
            .checked_mul(dim_usize)
            .ok_or_else(|| LoadError::ShapeOverflow {
                path: path.to_path_buf(),
                dims: proto.dims.clone(),
            })?;
        shape.push(dim_usize);
    }

    // Prefer raw_data if present (this is how upstream tests ship), and
    // require it to be a clean multiple of 4 so a truncated buffer
    // cannot silently parse as a shorter-but-valid f32 sequence.
    let values: Vec<f32> = if !proto.raw_data.is_empty() {
        if proto.raw_data.len() % 4 != 0 {
            return Err(LoadError::UnalignedRawData {
                path: path.to_path_buf(),
                len: proto.raw_data.len(),
            });
        }
        proto
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    } else {
        // Note: an empty `float_data` is legitimate for shapes whose
        // product is zero (e.g. `[0]` or `[2, 0, 3]`); the length check
        // below will accept that case and reject anything else.
        proto.float_data.clone()
    };

    if values.len() != expected {
        return Err(LoadError::LengthMismatch {
            path: path.to_path_buf(),
            shape,
            expected,
            actual: values.len(),
        });
    }

    Ok(FloatTensor { shape, values })
}
