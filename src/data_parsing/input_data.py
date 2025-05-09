import numpy as np
import json
from argparse import FileType
from onnx import TensorProto, GraphProto
from onnx.helper import make_graph, make_tensor_value_info


class InputData:
    """_summary_"""

    def __init__(self, file: FileType) -> None:
        """Class initializer. Reads the input data file that is passed as
        an argparse `FileType` object and separates the input data file into
        component parts (i.e. the target data, the years, the feature data,
        the feature names, the target name)

        Args:
            file (FileType): the FileType object containing the input data file.
        """

        # store filename
        self.filename = file.name

        # use genfromtxt to read the entire file as a field-ed array
        entire_file = np.genfromtxt(file, names=True, deletechars="")
        column_names = list(entire_file.dtype.names)

        # Get the names
        self.target_name = column_names[1]
        self.feature_names = column_names[2:]

        # Get the years, the target, and the features
        self.years = entire_file[column_names[0]]
        self.target_data = entire_file[self.target_name]
        self.feature_data = entire_file[self.feature_names]

        return

    def make_json_metadata(self) -> str:
        """Generates a JSON representation of the data, along with some metadata
        for use in external products and general documentation.

        example:
        >>> self.make_json_metadata()
        {
            "filename": "test_file.txt",
            "target_info": {
                "name": "Little_Laramie_R_nr_Filmore/06661000:WY:USGS/A",
                "data": [42997.68, 26513.07, 52986.45],
                "min": 26513.07,
                "max": 52986.45,
                "median": 42997.68,
                "average": 40832.40
            },
            "feature_info": [
                {
                    "name": "Albany/06H11:WY:SNOW/W;5-1;75447",
                    "data": [9.6, 6.0, 16.0],
                    "min": 6.0,
                    "max": 16.0,
                    "median": 9.6,
                    "average": 10.53
                }, ...
            ],
            "years": [1991, 1992, 1993]
        }

        Returns:
            str: JSON representation
        """

        return json.dumps(
            {
                "filename": self.filename,
                "target_info": {
                    "name": self.target_name,
                    "data": self.target_data.tolist(),
                    "min": np.nanmin(self.target_data),
                    "max": np.nanmax(self.target_data),
                    "median": np.nanmedian(self.target_data),
                    "average": np.nanmean(self.target_data),
                },
                "feature_data": [
                    {
                        "name": feature,
                        "data": self.feature_data[feature].tolist(),
                        "min": np.nanmin(self.feature_data[feature]),
                        "max": np.nanmax(self.feature_data[feature]),
                        "median": np.nanmedian(self.feature_data[feature]),
                        "average": np.nanmean(self.feature_data[feature]),
                    }
                    for i, feature in enumerate(self.feature_names)
                ],
                "years": self.years.tolist(),
            }
        )

    def make_onnx_graph(self) -> GraphProto:
        """Generates the input layer graph for the input_data

        Returns:
            GraphProto: A ONNX graph containing the predictor input layer.
                Each feature in the input dataset will have a node.
        """

        # Inputs
        inputs = [
            make_tensor_value_info(feature_name, TensorProto.FLOAT, [None, 1])
            for feature_name in self.feature_names
        ]

        # Outputs
        outputs = [
            make_tensor_value_info(feature_name, TensorProto.FLOAT, [None, 1])
            for feature_name in self.feature_names
        ]

        return make_graph([], "predictor_inputs", inputs, outputs)
