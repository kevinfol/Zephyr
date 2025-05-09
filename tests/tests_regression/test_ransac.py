import sys, os

sys.path.append(os.getcwd())

import unittest
import numpy as np
from regression import RansacRegression
import onnxruntime as ort
from onnx.checker import check_graph
from onnx.helper import make_model, make_opsetid


class TestRansac(unittest.TestCase):

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

    target_data = np.array(
        [
            125.90570142,
            40.38500709,
            143.0962353,
            168.26103496,
            62.79050046,
            169.6674689,
            74.67777257,
            36.56475897,
            64.27187199,
            15.39291425,
        ]
    )

    def test_fit_predict(self):
        print("Regression: Testing RANSAC")
        rr = RansacRegression()
        rr.set_monotonic([True, True, True, True])
        rr.fit(self.predictor_data, self.target_data)
        self.assertAlmostEqual(rr.regr.estimator_.coef_[1], 4, 0)
        predictions = rr.predict(self.predictor_data)
        self.assertAlmostEqual(predictions[0], 125, 0)

    def test_graph_gen(self):
        print("Regression: Testing RANSAC ONNX")
        rr = RansacRegression()
        rr.set_monotonic([True, True, True, True])
        rr.fit(self.predictor_data, self.target_data)
        graph = rr.create_onnx_graph()
        self.assertIsNone(check_graph(graph))
        model = make_model(
            graph,
            opset_imports=[make_opsetid("", 19), make_opsetid("ai.onnx.ml", 2)],
            ir_version=9,
        )
        session = ort.InferenceSession(model.SerializeToString())
        feed = {"x": [[0.7856, 26.20420, 0.5580, 3.49560]]}
        val = session.run(None, feed)[0]
        self.assertAlmostEqual(val[0], np.float32(125), 0)


if __name__ == "__main__":
    unittest.main()
