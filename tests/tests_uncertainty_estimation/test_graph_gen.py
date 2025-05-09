import sys, os

sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + "/..")

import unittest
import numpy as np
import onnxruntime as ort
from onnx.checker import check_graph
from onnx.helper import make_model, make_opsetid
from uncertainty_estimation import create_uncertainty_onnx_graph


class TestGraphGen(unittest.TestCase):

    def test_1(self):

        print("Uncertainty: Testing ONNX")

        preds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        obsvt = np.array([1.1, 1.9, 3.2, 4.0, 4.9, 6.1, 7.0, 8.3, 9.1])
        deg_f = 4
        graph = create_uncertainty_onnx_graph(
            predictions=preds, observations=obsvt, deg_freedom=deg_f
        )
        self.assertIsNone(check_graph(graph))
        model = make_model(
            graph,
            opset_imports=[make_opsetid("", 19), make_opsetid("ai.onnx.ml", 2)],
            ir_version=9,
        )
        session = ort.InferenceSession(model.SerializeToString())
        feed = {"forecast_input": [[5.5]]}

        forecast = session.run(None, feed)[0]
        self.assertEqual(forecast[0][0], np.float32(4.9701014))


if __name__ == "__main__":
    unittest.main()
