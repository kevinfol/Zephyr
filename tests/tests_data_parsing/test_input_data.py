import sys, os

sys.path.append(os.getcwd())

import unittest
import json
from data_parsing import InputData
from io import StringIO
import onnxruntime as ort
from onnx.checker import check_graph
from onnx.helper import make_model, make_opsetid

DATA_FILE_IO = StringIO(
    """Year	Little_Laramie_R_nr_Filmore/06661000:WY:USGS/A	Albany/06H11:WY:SNOW/W;5-1;75447	Brooklyn_Lake/367:WY:SNTL/P;10-01/05-01;75441	Brooklyn_Lake/367:WY:SNTL/W;5-1;75449	Hairpin_Turn/06H02:WY:SNOW/W;5-1;75443	Libby_Lodge/06H03:WY:SNOW/W;5-1;75445	North_French_Creek/668:WY:SNTL/P;10-01/05-01;75451	North_French_Creek/668:WY:SNTL/W;5-1;75453	
1991	42997.68	9.6	21.8	20.3	11.2	5.7	28.8	32.1	
1992	26513.07	6.0	18.7	14.5	6.4	0.0	24.1	21.0	
1993	52986.45	16.0	26.8	30.3	17.3	10.3	33.2	38.4	"""
)
DATA_FILE_IO.name = "testfile.txt"


class TestInputData(unittest.TestCase):

    def test_parsing(self):
        print("Data Parsing: Testing input reading")
        DATA_FILE_IO.seek(0)
        data = InputData(DATA_FILE_IO)
        self.assertEqual(
            data.target_name, "Little_Laramie_R_nr_Filmore/06661000:WY:USGS/A"
        )
        self.assertEqual(data.feature_names[0], "Albany/06H11:WY:SNOW/W;5-1;75447")
        self.assertEqual(data.feature_data[0][0], 9.6)
        self.assertEqual(data.years[0], 1991.0)
        self.assertEqual(data.target_data[0], 42997.68)

    def test_json(self):
        print("Data Parsing: Testing JSON generation")
        DATA_FILE_IO.seek(0)
        data = InputData(DATA_FILE_IO)
        json_str = data.make_json_metadata()
        d = json.loads(json_str)
        self.assertEqual(d["target_info"]["min"], 26513.07)

    def test_graph_gen(self):
        print("Data Parsing: Testing ONNX generation")
        DATA_FILE_IO.seek(0)
        data = InputData(DATA_FILE_IO)
        graph = data.make_onnx_graph()
        self.assertIsNone(check_graph(graph))
        model = make_model(
            graph,
            opset_imports=[make_opsetid("", 19), make_opsetid("ai.onnx.ml", 2)],
            ir_version=9,
        )
        session = ort.InferenceSession(model.SerializeToString())
        feed = {
            "Albany/06H11:WY:SNOW/W;5-1;75447": [[1.0]],
            "Brooklyn_Lake/367:WY:SNTL/P;10-01/05-01;75441": [[2.0]],
            "Brooklyn_Lake/367:WY:SNTL/W;5-1;75449": [[3.0]],
            "Hairpin_Turn/06H02:WY:SNOW/W;5-1;75443": [[4.0]],
            "Libby_Lodge/06H03:WY:SNOW/W;5-1;75445": [[5.0]],
            "North_French_Creek/668:WY:SNTL/P;10-01/05-01;75451": [[6.0]],
            "North_French_Creek/668:WY:SNTL/W;5-1;75453": [[7.0]],
        }
        val = session.run(None, feed)[1]
        self.assertEqual(val[0][0], 2.0)


if __name__ == "__main__":
    unittest.main()
