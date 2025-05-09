import sys, os

sys.path.append(os.getcwd())

import unittest
import numpy as np
import onnxruntime as ort
from onnx.checker import check_graph
from onnx.helper import make_model, make_opsetid
from preprocessing import get


class TestStandard(unittest.TestCase):

    predictor_data = np.array(
        [
            [0.7856, 26.20420, 0.5580, 3.49560],
            [0.9650, 7.021900, 0.0858, 1.57572],
            [0.2727, 31.93610, 0.8096, 0.74916],
            [0.1532, 36.95420, 0.8730, 2.19240],
            [0.5327, 11.45520, 0.5815, 3.01320],
            [0.4724, 38.74300, 0.4202, 1.05372],
            [0.3160, 15.50580, 0.7954, 0.58608],
            [0.9585, 4.983700, 0.7940, 3.36240],
            [0.1251, 12.21200, 0.2178, 2.62080],
            [0.6020, 0.191909, 0.1892, 3.39120],
        ]
    )

    def test_fit_transform(self):
        print("Preprocessing: Testing Standard fit/transform")
        std = get("standard")()
        std.fit(X=self.predictor_data)
        new_data = np.array([[0.5355, 22.42513, 0.5225, 2.34411]])
        transformed_data = std.transform(X=new_data)
        self.assertEqual(transformed_data[0][0], np.float64(0.058473440142459414))

    def test_graph_gen(self):
        print("Preprocessing: Testing Standard ONNX generation")
        std = get("standard")()
        std.fit(X=self.predictor_data)
        graph = std.create_onnx_graph()
        self.assertIsNone(check_graph(graph))
        model = make_model(
            graph,
            opset_imports=[make_opsetid("", 19), make_opsetid("ai.onnx.ml", 2)],
            ir_version=9,
        )
        session = ort.InferenceSession(model.SerializeToString())
        feed = {
            "data_input_0": [[0.5355]],
            "data_input_1": [[22.42513]],
            "data_input_2": [[0.5225]],
            "data_input_3": [[2.34411]],
        }
        val = session.run(None, feed)[0]
        self.assertEqual(val[0][0], np.float32(0.058473323))


if __name__ == "__main__":
    unittest.main()
