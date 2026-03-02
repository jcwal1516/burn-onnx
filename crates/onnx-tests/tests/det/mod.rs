// Import the shared macro
use crate::include_models;
include_models!(det);

#[cfg(test)]
mod tests {
    use super::*;

    use crate::backend::TestBackend;

    #[test]
    fn det_2x2() {
        // det([[1, 2], [3, 4]]) = 1*4 - 2*3 = -2
        let device = Default::default();
        let model: det::Model<TestBackend> = det::Model::new(&device);

        let input =
            burn::tensor::Tensor::<TestBackend, 2>::from_floats([[1.0f32, 2.0], [3.0, 4.0]], &device);

        let output: f32 = model.forward(input);
        let expected = -2.0f32;

        let diff = (output - expected).abs();
        assert!(
            diff < 1e-4,
            "Expected det = {expected}, got {output}, diff = {diff}"
        );
    }
}
