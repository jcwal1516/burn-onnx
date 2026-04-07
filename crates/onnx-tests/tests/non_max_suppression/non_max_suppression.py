#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy==2.2.6",
# ]
# ///

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper
from onnx.reference import ReferenceEvaluator


CORNER_BOXES = np.array(
    [
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, -0.1, 1.0, 0.9],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
            [0.0, 100.0, 1.0, 101.0],
        ]
    ],
    dtype=np.float32,
)

CENTER_BOXES = np.array(
    [
        [
            [0.5, 0.5, 1.0, 1.0],
            [0.6, 0.5, 1.0, 1.0],
            [0.4, 0.5, 1.0, 1.0],
            [10.5, 0.5, 1.0, 1.0],
            [10.6, 0.5, 1.0, 1.0],
            [100.5, 0.5, 1.0, 1.0],
        ]
    ],
    dtype=np.float32,
)

SCORES = np.array([[[0.9, 0.75, 0.6, 0.95, 0.5, 0.3]]], dtype=np.float32)
MAX_OUTPUT = np.array([3], dtype=np.int64)
IOU_THRESHOLD = np.array([0.5], dtype=np.float32)
SCORE_THRESHOLD = np.array([0.0], dtype=np.float32)
HIGH_SCORE_THRESHOLD = np.array([0.8], dtype=np.float32)
NEGATIVE_SCORES = np.array([[[-0.1, -0.2]]], dtype=np.float32)

# Multi-class test data: 1 batch, 4 boxes, 2 classes
MULTI_CLASS_BOXES = np.array(
    [
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.0, 0.1, 1.0, 1.1],
            [0.0, 10.0, 1.0, 11.0],
            [0.0, 10.1, 1.0, 11.1],
        ]
    ],
    dtype=np.float32,
)
MULTI_CLASS_SCORES = np.array(
    [
        [
            [0.9, 0.8, 0.7, 0.6],
            [0.5, 0.6, 0.9, 0.8],
        ]
    ],
    dtype=np.float32,
)


def make_value_info(name: str, elem_type: int, shape):
    return helper.make_tensor_value_info(name, elem_type=elem_type, shape=shape)


def save_model(path: Path, center_point_box: int, node_inputs: list[str], graph_inputs):
    node = helper.make_node(
        "NonMaxSuppression",
        inputs=node_inputs,
        outputs=["selected_indices"],
        name=f"/{path.stem}",
        center_point_box=center_point_box,
    )

    model = helper.make_model(
        graph=helper.make_graph(
            nodes=[node],
            name=path.stem,
            inputs=graph_inputs,
            outputs=[
                make_value_info(
                    "selected_indices",
                    TensorProto.INT64,
                    ("num_selected", 3),
                )
            ],
        ),
        opset_imports=[helper.make_operatorsetid("", 11)],
    )

    onnx.checker.check_model(model)
    onnx.save(model, path)
    return model


def print_reference(label: str, model, feeds):
    print(f"\n{label}:")
    for name, value in feeds.items():
        print(f"  input {name}: shape={value.shape}, data={value.tolist()}")
    selected = ReferenceEvaluator(model).run(None, feeds)[0]
    print(f"  output: {selected.tolist()}")


def main():
    root = Path(__file__).resolve().parent

    corner_inputs = [
        make_value_info("boxes", TensorProto.FLOAT, (1, 6, 4)),
        make_value_info("scores", TensorProto.FLOAT, (1, 1, 6)),
        make_value_info("max_output_boxes_per_class", TensorProto.INT64, (1,)),
        make_value_info("iou_threshold", TensorProto.FLOAT, (1,)),
        make_value_info("score_threshold", TensorProto.FLOAT, (1,)),
    ]
    corner_model = save_model(
        root / "non_max_suppression.onnx",
        center_point_box=0,
        node_inputs=[
            "boxes",
            "scores",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        graph_inputs=corner_inputs,
    )
    print_reference(
        "corner expected",
        corner_model,
        {
            "boxes": CORNER_BOXES,
            "scores": SCORES,
            "max_output_boxes_per_class": MAX_OUTPUT,
            "iou_threshold": IOU_THRESHOLD,
            "score_threshold": SCORE_THRESHOLD,
        },
    )

    center_model = save_model(
        root / "non_max_suppression_center.onnx",
        center_point_box=1,
        node_inputs=[
            "boxes",
            "scores",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        graph_inputs=corner_inputs,
    )
    print_reference(
        "center expected",
        center_model,
        {
            "boxes": CENTER_BOXES,
            "scores": SCORES,
            "max_output_boxes_per_class": MAX_OUTPUT,
            "iou_threshold": IOU_THRESHOLD,
            "score_threshold": SCORE_THRESHOLD,
        },
    )

    missing_middle_inputs = [
        make_value_info("boxes", TensorProto.FLOAT, (1, 6, 4)),
        make_value_info("scores", TensorProto.FLOAT, (1, 1, 6)),
        make_value_info("max_output_boxes_per_class", TensorProto.INT64, (1,)),
        make_value_info("score_threshold", TensorProto.FLOAT, (1,)),
    ]
    missing_middle_model = save_model(
        root / "non_max_suppression_missing_middle.onnx",
        center_point_box=0,
        node_inputs=[
            "boxes",
            "scores",
            "max_output_boxes_per_class",
            "",
            "score_threshold",
        ],
        graph_inputs=missing_middle_inputs,
    )

    # ReferenceEvaluator currently crashes on valid omitted-middle optional inputs.
    # Compute the expected result via an equivalent model that passes the ONNX
    # default iou_threshold explicitly.
    missing_middle_reference_model = save_model(
        root / "_non_max_suppression_missing_middle_reference.onnx",
        center_point_box=0,
        node_inputs=[
            "boxes",
            "scores",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        graph_inputs=corner_inputs,
    )
    print_reference(
        "missing-middle expected",
        missing_middle_reference_model,
        {
            "boxes": CORNER_BOXES,
            "scores": SCORES,
            "max_output_boxes_per_class": MAX_OUTPUT,
            "iou_threshold": np.array([0.0], dtype=np.float32),
            "score_threshold": HIGH_SCORE_THRESHOLD,
        },
    )

    missing_score_threshold_inputs = [
        make_value_info("boxes", TensorProto.FLOAT, (1, 6, 4)),
        make_value_info("scores", TensorProto.FLOAT, (1, 1, 6)),
        make_value_info("max_output_boxes_per_class", TensorProto.INT64, (1,)),
        make_value_info("iou_threshold", TensorProto.FLOAT, (1,)),
    ]
    missing_score_threshold_model = save_model(
        root / "non_max_suppression_missing_score_threshold.onnx",
        center_point_box=0,
        node_inputs=[
            "boxes",
            "scores",
            "max_output_boxes_per_class",
            "iou_threshold",
        ],
        graph_inputs=missing_score_threshold_inputs,
    )
    print_reference(
        "missing-score-threshold expected",
        missing_score_threshold_model,
        {
            "boxes": CORNER_BOXES[:, :2, :],
            "scores": NEGATIVE_SCORES,
            "max_output_boxes_per_class": np.array([1], dtype=np.int64),
            "iou_threshold": IOU_THRESHOLD,
        },
    )

    (root / "_non_max_suppression_missing_middle_reference.onnx").unlink()

    # Minimal model: only boxes and scores (max_output_boxes_per_class omitted).
    # ONNX default for max_output_boxes_per_class is 0, so output should be empty.
    minimal_inputs = [
        make_value_info("boxes", TensorProto.FLOAT, (1, 2, 4)),
        make_value_info("scores", TensorProto.FLOAT, (1, 1, 2)),
    ]
    save_model(
        root / "non_max_suppression_minimal.onnx",
        center_point_box=0,
        node_inputs=["boxes", "scores"],
        graph_inputs=minimal_inputs,
    )
    # No ReferenceEvaluator call — the expected output is a 0-row [0, 3] tensor.

    multi_class_inputs = [
        make_value_info("boxes", TensorProto.FLOAT, (1, 4, 4)),
        make_value_info("scores", TensorProto.FLOAT, (1, 2, 4)),
        make_value_info("max_output_boxes_per_class", TensorProto.INT64, (1,)),
        make_value_info("iou_threshold", TensorProto.FLOAT, (1,)),
        make_value_info("score_threshold", TensorProto.FLOAT, (1,)),
    ]
    multi_class_model = save_model(
        root / "non_max_suppression_multi_class.onnx",
        center_point_box=0,
        node_inputs=[
            "boxes",
            "scores",
            "max_output_boxes_per_class",
            "iou_threshold",
            "score_threshold",
        ],
        graph_inputs=multi_class_inputs,
    )
    print_reference(
        "multi-class expected",
        multi_class_model,
        {
            "boxes": MULTI_CLASS_BOXES,
            "scores": MULTI_CLASS_SCORES,
            "max_output_boxes_per_class": np.array([2], dtype=np.int64),
            "iou_threshold": IOU_THRESHOLD,
            "score_threshold": SCORE_THRESHOLD,
        },
    )


if __name__ == "__main__":
    main()
