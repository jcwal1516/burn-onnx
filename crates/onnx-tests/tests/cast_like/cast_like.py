#!/usr/bin/env -S uv run --script

# /// script
# dependencies = [
#   "onnx==1.19.0",
#   "numpy",
# ]
# ///

# used to generate model: cast_like.onnx

import numpy as np
import onnx
from onnx import helper, TensorProto

OPSET_VERSION = 15


def main():
    # CastLike: cast float tensor to match the type of an int tensor
    node0 = helper.make_node(
        "CastLike",
        inputs=["float_input", "int_target"],
        outputs=["float_to_int"],
    )
    # CastLike: cast int tensor to match the type of a float tensor
    node1 = helper.make_node(
        "CastLike",
        inputs=["int_input", "float_target"],
        outputs=["int_to_float"],
    )

    float_input = helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [2, 3])
    int_target = helper.make_tensor_value_info("int_target", TensorProto.INT64, [2, 3])
    int_input = helper.make_tensor_value_info("int_input", TensorProto.INT64, [2, 3])
    float_target = helper.make_tensor_value_info("float_target", TensorProto.FLOAT, [2, 3])

    float_to_int = helper.make_tensor_value_info("float_to_int", TensorProto.INT64, [2, 3])
    int_to_float = helper.make_tensor_value_info("int_to_float", TensorProto.FLOAT, [2, 3])

    graph = helper.make_graph(
        [node0, node1],
        "cast_like_graph",
        [float_input, int_target, int_input, float_target],
        [float_to_int, int_to_float],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_operatorsetid("", OPSET_VERSION)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, "cast_like.onnx")
    print("Finished exporting model to cast_like.onnx")


if __name__ == "__main__":
    main()
