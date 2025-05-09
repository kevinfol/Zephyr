import sys, os

sys.path.append(os.getcwd())

import unittest
import numpy as np
import torch
from regression import NeuralNetworkRegression
import onnxruntime as ort
from onnx.checker import check_model
from onnx.helper import make_model, make_opsetid


class TestNN(unittest.TestCase):

    predictors = np.array(
        [
            [9.71395150e-01, 9.05222926e-01, 5.70907828e-01, 5.41806951e-02],
            [5.97419132e-01, 9.11192837e-01, 7.45087736e-01, 3.11186945e-01],
            [6.81331104e-01, 3.63061876e-01, 7.75503610e-01, 5.78507748e-01],
            [8.26000695e-01, 9.59130842e-01, 4.87378581e-01, 6.05251298e-01],
            [7.41834587e-02, 4.06866748e-01, 9.38616525e-01, 5.44357354e-01],
            [7.48640212e-01, 9.02111384e-01, 1.12320379e-01, 6.00396872e-01],
            [1.14883125e-01, 9.38865573e-01, 8.24883230e-01, 1.56438568e-01],
            [4.12688625e-01, 6.66603237e-01, 2.42019762e-01, 1.36000812e-01],
            [8.75194898e-01, 4.69063386e-01, 8.80017014e-02, 9.63501193e-01],
            [3.02434309e-01, 1.31977153e-01, 6.40814331e-01, 1.20240797e-01],
            [5.31893637e-01, 8.51192989e-01, 4.92573983e-01, 8.24244818e-01],
            [6.02983678e-01, 3.92082000e-01, 7.97711679e-01, 5.29520440e-01],
            [7.91489357e-01, 4.33048043e-02, 5.22122602e-01, 9.90076700e-01],
            [6.89920597e-01, 4.38146115e-01, 6.10430566e-01, 7.26848895e-01],
            [1.08480563e-01, 7.18377475e-01, 8.70915576e-01, 8.16058030e-01],
            [2.80691508e-01, 5.75657143e-01, 1.26247212e-01, 3.97492115e-01],
            [1.66298670e-01, 5.41539807e-01, 6.92769101e-01, 9.29417357e-01],
            [3.38995895e-02, 2.11343566e-01, 8.49721523e-01, 5.60834873e-01],
            [8.42659959e-01, 4.61629268e-01, 8.83471321e-01, 4.72628835e-01],
            [7.10094747e-01, 7.54251893e-01, 4.49309488e-01, 7.39076970e-01],
            [6.17723543e-01, 4.09181595e-01, 8.05225126e-02, 3.00953900e-01],
            [3.68362524e-01, 4.65972176e-01, 8.91954022e-01, 3.80736545e-01],
            [8.73902771e-02, 9.68561281e-01, 2.68298236e-01, 8.26153777e-01],
            [1.01464322e-01, 7.52625179e-01, 6.30462454e-02, 3.84595123e-01],
            [4.33972269e-01, 5.77423843e-01, 3.33083117e-01, 5.93340528e-01],
            [7.12085594e-04, 7.21639491e-01, 8.35168490e-01, 3.35323735e-01],
            [1.84243577e-01, 8.51045961e-01, 5.02885405e-01, 2.99384215e-01],
        ]
    )

    target = np.array(
        [
            11.68251838,
            8.63136228,
            6.81455812,
            7.57506159,
            5.71315972,
            7.15599797,
            8.80159374,
            3.36802701,
            6.01552442,
            4.74853612,
            6.69714541,
            6.60041185,
            3.38678226,
            5.30477232,
            5.385417,
            0.86310239,
            2.47414454,
            5.57442742,
            9.08989404,
            6.73778715,
            2.37466756,
            5.77924631,
            1.47549849,
            0.99849294,
            1.69259627,
            6.07053189,
            3.44131379,
        ]
    )

    def test_fit_predict(self):

        regr = NeuralNetworkRegression()
        regr.set_monotonic([True, True, True, False])
        regr.set_params(**{"num_hidden_layers": 2})

        regr.fit(self.predictors, self.target)
        pred = regr.predict(self.predictors)

        return

    def test_onnx_graph_gen(self):

        regr = NeuralNetworkRegression()
        regr.set_monotonic([True, True, True, False])
        regr.set_params(**{"num_hidden_layers": 3})

        regr.fit(self.predictors, self.target)

        graph = regr.create_onnx_graph()
        model = make_model(
            graph,
            opset_imports=[make_opsetid("", 19), make_opsetid("ai.onnx.ml", 2)],
            ir_version=9,
        )
        self.assertIsNone(check_model(model))
        session = ort.InferenceSession(model.SerializeToString())
        feed = {"x": [[9.71395150e-01, 9.05222926e-01, 5.70907828e-01, 5.41806951e-02]]}
        val = session.run(None, feed)[0]
        self.assertAlmostEqual(np.log10(val[0]), 1, 0)
        print()


if __name__ == "__main__":
    unittest.main()
