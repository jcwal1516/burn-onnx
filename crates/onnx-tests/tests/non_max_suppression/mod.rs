use crate::include_models;
include_models!(
    non_max_suppression,
    non_max_suppression_center,
    non_max_suppression_minimal,
    non_max_suppression_missing_middle,
    non_max_suppression_missing_score_threshold,
    non_max_suppression_multi_class
);

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Int, Tensor, TensorData};

    use crate::backend::TestBackend;

    fn corner_boxes(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 3> {
        Tensor::<TestBackend, 3>::from_floats(
            [[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, -0.1, 1.0, 0.9],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
                [0.0, 100.0, 1.0, 101.0],
            ]],
            device,
        )
    }

    fn center_boxes(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 3> {
        Tensor::<TestBackend, 3>::from_floats(
            [[
                [0.5, 0.5, 1.0, 1.0],
                [0.6, 0.5, 1.0, 1.0],
                [0.4, 0.5, 1.0, 1.0],
                [10.5, 0.5, 1.0, 1.0],
                [10.6, 0.5, 1.0, 1.0],
                [100.5, 0.5, 1.0, 1.0],
            ]],
            device,
        )
    }

    fn scores(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 3> {
        Tensor::<TestBackend, 3>::from_floats([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], device)
    }

    fn two_corner_boxes(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 3> {
        Tensor::<TestBackend, 3>::from_floats(
            [[[0.0, 0.0, 1.0, 1.0], [10.0, 10.0, 11.0, 11.0]]],
            device,
        )
    }

    fn negative_scores(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 3> {
        Tensor::<TestBackend, 3>::from_floats([[[-0.1, -0.2]]], device)
    }

    fn separated_boxes(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 3> {
        Tensor::<TestBackend, 3>::from_floats(
            [[
                [0.0, 0.0, 1.0, 1.0],
                [10.0, 10.0, 11.0, 11.0],
                [20.0, 20.0, 21.0, 21.0],
            ]],
            device,
        )
    }

    fn equality_scores(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 3> {
        Tensor::<TestBackend, 3>::from_floats([[[0.5, 0.6, 0.4]]], device)
    }

    fn reversed_corner_boxes(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 3> {
        Tensor::<TestBackend, 3>::from_floats(
            [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.1, 1.0, 1.1]]],
            device,
        )
    }

    fn overlapping_scores(
        device: &<TestBackend as burn::tensor::backend::Backend>::Device,
    ) -> Tensor<TestBackend, 3> {
        Tensor::<TestBackend, 3>::from_floats([[[0.9, 0.8]]], device)
    }

    #[test]
    fn non_max_suppression_corner() {
        let device = Default::default();
        let model: non_max_suppression::Model<TestBackend> =
            non_max_suppression::Model::new(&device);

        let max_output_boxes_per_class = Tensor::<TestBackend, 1, Int>::from_ints([3], &device);
        let iou_threshold = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let score_threshold = Tensor::<TestBackend, 1>::from_floats([0.0], &device);

        let output = model.forward(
            corner_boxes(&device),
            scores(&device),
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        );

        let expected = TensorData::from([[0i64, 0, 3], [0, 0, 0], [0, 0, 5]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn non_max_suppression_center() {
        let device = Default::default();
        let model: non_max_suppression_center::Model<TestBackend> =
            non_max_suppression_center::Model::new(&device);

        let max_output_boxes_per_class = Tensor::<TestBackend, 1, Int>::from_ints([3], &device);
        let iou_threshold = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let score_threshold = Tensor::<TestBackend, 1>::from_floats([0.0], &device);

        let output = model.forward(
            center_boxes(&device),
            scores(&device),
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        );

        let expected = TensorData::from([[0i64, 0, 3], [0, 0, 0], [0, 0, 5]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn non_max_suppression_missing_middle_optional_input() {
        let device = Default::default();
        let model: non_max_suppression_missing_middle::Model<TestBackend> =
            non_max_suppression_missing_middle::Model::new(&device);

        let max_output_boxes_per_class = Tensor::<TestBackend, 1, Int>::from_ints([3], &device);
        let score_threshold = Tensor::<TestBackend, 1>::from_floats([0.8], &device);

        let output = model.forward(
            corner_boxes(&device),
            scores(&device),
            max_output_boxes_per_class,
            score_threshold,
        );

        let expected = TensorData::from([[0i64, 0, 3], [0, 0, 0]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn non_max_suppression_missing_trailing_score_threshold_keeps_negative_scores() {
        let device = Default::default();
        let model: non_max_suppression_missing_score_threshold::Model<TestBackend> =
            non_max_suppression_missing_score_threshold::Model::new(&device);

        let max_output_boxes_per_class = Tensor::<TestBackend, 1, Int>::from_ints([1], &device);
        let iou_threshold = Tensor::<TestBackend, 1>::from_floats([0.5], &device);

        let output = model.forward(
            two_corner_boxes(&device),
            negative_scores(&device),
            max_output_boxes_per_class,
            iou_threshold,
        );

        let expected = TensorData::from([[0i64, 0, 0]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn non_max_suppression_score_threshold_keeps_equal_scores() {
        let device = Default::default();
        let model: non_max_suppression::Model<TestBackend> =
            non_max_suppression::Model::new(&device);

        let max_output_boxes_per_class = Tensor::<TestBackend, 1, Int>::from_ints([3], &device);
        let iou_threshold = Tensor::<TestBackend, 1>::from_floats([0.0], &device);
        let score_threshold = Tensor::<TestBackend, 1>::from_floats([0.5], &device);

        let output = model.forward(
            separated_boxes(&device),
            equality_scores(&device),
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        );

        let expected = TensorData::from([[0i64, 0, 1], [0, 0, 0]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn non_max_suppression_multi_class() {
        let device = Default::default();
        let model: non_max_suppression_multi_class::Model<TestBackend> =
            non_max_suppression_multi_class::Model::new(&device);

        // 1 batch, 4 boxes: two overlapping pairs far apart
        let boxes = Tensor::<TestBackend, 3>::from_floats(
            [[
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.1, 1.0, 1.1],
                [0.0, 10.0, 1.0, 11.0],
                [0.0, 10.1, 1.0, 11.1],
            ]],
            &device,
        );
        // 2 classes with different score rankings
        let scores = Tensor::<TestBackend, 3>::from_floats(
            [[
                [0.9, 0.8, 0.7, 0.6], // class 0: boxes 0,1 rank high
                [0.5, 0.6, 0.9, 0.8], // class 1: boxes 2,3 rank high
            ]],
            &device,
        );
        let max_output_boxes_per_class = Tensor::<TestBackend, 1, Int>::from_ints([2], &device);
        let iou_threshold = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let score_threshold = Tensor::<TestBackend, 1>::from_floats([0.0], &device);

        let output = model.forward(
            boxes,
            scores,
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        );

        // Class 0: box 0 (0.9) kept, box 1 suppressed (overlaps 0), box 2 (0.7) kept
        // Class 1: box 2 (0.9) kept, box 3 suppressed (overlaps 2), box 1 (0.6) kept
        let expected = TensorData::from([[0i64, 0, 0], [0, 0, 2], [0, 1, 2], [0, 1, 1]]);
        output.to_data().assert_eq(&expected, true);
    }

    #[test]
    fn non_max_suppression_minimal_inputs_returns_empty() {
        let device = Default::default();
        let model: non_max_suppression_minimal::Model<TestBackend> =
            non_max_suppression_minimal::Model::new(&device);

        let output = model.forward(two_corner_boxes(&device), overlapping_scores(&device));

        // max_output_boxes_per_class defaults to 0, so output should be empty [0, 3].
        assert_eq!(output.shape().dims(), [0, 3]);
    }

    #[test]
    fn non_max_suppression_corner_accepts_reversed_diagonal_pairs() {
        let device = Default::default();
        let model: non_max_suppression::Model<TestBackend> =
            non_max_suppression::Model::new(&device);

        let max_output_boxes_per_class = Tensor::<TestBackend, 1, Int>::from_ints([2], &device);
        let iou_threshold = Tensor::<TestBackend, 1>::from_floats([0.5], &device);
        let score_threshold = Tensor::<TestBackend, 1>::from_floats([0.0], &device);

        let output = model.forward(
            reversed_corner_boxes(&device),
            overlapping_scores(&device),
            max_output_boxes_per_class,
            iou_threshold,
            score_threshold,
        );

        let expected = TensorData::from([[0i64, 0, 0]]);
        output.to_data().assert_eq(&expected, true);
    }
}
