import numpy as np
from . import GenericRegressor
import torch
import tempfile
import os

from onnx.helper import (
    make_node,
    make_tensor_value_info,
    make_graph,
    make_opsetid,
    make_model,
    make_tensor,
)
from onnx.compose import merge_graphs
from onnx import numpy_helper, TensorProto, GraphProto, load


class NeuralNet(torch.nn.Module):
    def __init__(self, n_features: int, num_hidden_layers: int, activation_fn="ReLU"):
        super().__init__()
        self.n_features = n_features
        self.num_hidden_layers = num_hidden_layers
        self.monotone_constraint = None
        self.learning_rate = 0.1
        self.activation_fn = getattr(torch.nn, activation_fn)
        hidden_layer_size = int(np.ceil(n_features / 2))
        hidden_layer_size = max(hidden_layer_size, 2)

        layers = [
            torch.nn.Linear(n_features, hidden_layer_size),
            self.activation_fn(),
        ]
        # layers[0].weight.data.fill_(1)
        # layers[0].bias.data.fill_(0.01)

        for n in range(1, num_hidden_layers + 1):
            if n == num_hidden_layers:
                layers.extend(
                    [
                        torch.nn.Linear(hidden_layer_size, 1),
                    ]
                )
                # layers[-1].weight.data.fill_(1)
                # layers[-1].bias.data.fill_(0.01)
            else:
                layers.extend(
                    [
                        torch.nn.Linear(hidden_layer_size, hidden_layer_size),
                        self.activation_fn(),
                    ]
                )
                # layers[-2].weight.data.fill_(1)
                # layers[-2].bias.data.fill_(0.01)

        self.linear_stack = torch.nn.Sequential(*layers)

    def get_params(self, *args, **kwargs):

        return {
            "num_hidden_layers": self.num_hidden_layers,
            "learning_rate": self.learning_rate,
            "activation_fn": self.activation_fn.__name__,
        }

    def set_params(self, **kwargs):
        mc = self.monotone_constraint
        if "num_hidden_layers" in kwargs.keys():
            if "activation_fn" in kwargs.keys():
                self.__init__(
                    self.n_features,
                    kwargs["num_hidden_layers"],
                    kwargs["activation_fn"],
                )
            else:
                self.__init__(self.n_features, kwargs["num_hidden_layers"])

        if "learning_rate" in kwargs.keys():
            self.learning_rate = kwargs["learning_rate"]

        if mc:
            self.set_monotonic_constraints(mc)
        else:
            self.set_monotonic_constraints([False] * self.n_features)

    def forward(self, X):
        return self.linear_stack(X)

    def set_monotonic_constraints(self, monotone_constraint: list[bool]):
        self.monotone_constraint = monotone_constraint
        for n in range(len(self.linear_stack)):
            layer = self.linear_stack[n]
            if isinstance(layer, torch.nn.Linear):
                if n == 0:
                    self.linear_stack[n].weight.monotonic_cols = monotone_constraint
                else:
                    self.linear_stack[n].weight.monotonic_cols = [
                        True
                    ] * layer.weight.data.shape[1]

    def initialize_weights_and_biases(self):
        self.linear_stack.apply(self.init_module_weights)

    def init_module_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)


class NonNegClipper:

    def __call__(self, module):

        if hasattr(module, "weight"):
            if hasattr(module.weight, "monotonic_cols"):
                monotonic_cols = module.weight.monotonic_cols
                w = module.weight.data[:, monotonic_cols]
                w = w.clamp(min=0.0)
                module.weight.data[:, monotonic_cols] = w


class NeuralNetworkRegression(GenericRegressor):
    """Implements a multilayer perceptron neural network"""

    PARAM_GRID = {
        "num_hidden_layers": [1, 2],
        "learning_rate": [0.1],  # , 0.02],
        "activation_fn": ["Mish", "Hardsigmoid"],
    }
    USE_PARALLEL_PROCESSING = True

    def __init__(self, *args, **kwargs):
        """_summary_"""
        GenericRegressor.__init__(self, *args, **kwargs)
        self.regr = NeuralNet(1, 1)

        self.regr.learning_rate = 0.1
        self.loss_function = torch.nn.MSELoss()
        self.num_epochs = 5000
        self.non_neg_clipper = NonNegClipper()
        self.regr.apply(self.non_neg_clipper)
        self.early_stopping_pct = 0.025 / 100

        return

    def set_monotonic(self, monotone_constraints: list[bool]) -> None:
        """_summary_

        Args:
            monotone_constraints (list[bool]): a list with True/False elements,
                where each element indicates whether the corresponding predictor
                should have a monotonic relationship with the target.
        """
        self.regr = NeuralNet(len(monotone_constraints), self.regr.num_hidden_layers)
        self.regr.set_monotonic_constraints(monotone_constraints)
        self.regr.apply(self.non_neg_clipper)
        return

    def has_zero_importance_predictors(self) -> bool:
        """If any of the predictors have zero importance / effect on the output
        of the model (e.g. zero-coefficient), then this function returns True,
        otherwise it returns False. It is useful for filtering out models.

        Returns:
            bool: True if one or more predictors have no effect on the
                model output. Otherwise, False
        """
        for input_num in range(self.regr.n_features):
            if not any(self.regr.linear_stack[0].weight.data[:, input_num]):
                return True
        return False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Create the model parameters for the brute force model lookup table.

        Args:
            X (np.ndarray): feature data of shape NxM
            y (np.ndarray): target data (must be of shape: Nx1)
        """

        X = torch.from_numpy(X)  # torch.tensor(X, dtype=torch.float32)
        y = torch.from_numpy(y)  # torch.tensor(y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.regr.parameters(), lr=self.regr.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 500, 2 / 3)
        initial_state_dict = scheduler.state_dict()

        self.y_min = torch.min(y)
        self.y_max = torch.max(y)
        self.y_proc = (y - self.y_min) / (self.y_max - self.y_min)
        self.y_proc = self.y_proc.unsqueeze(-1)
        self.regr.train()
        last_loss = None

        # Expectation is that there are a decent amount of local
        # minima in the parameters search space. We'll need to
        # monitor the loss and keep track of minima
        best_loss = None
        best_params = self.regr.state_dict()
        patience_lim = self.num_epochs * 0.05
        current_patience = 0

        for epoch in range(self.num_epochs):

            predictions = self.regr(X)
            loss = self.loss_function(predictions, self.y_proc)
            loss_val = loss.item()

            loss.backward()
            optimizer.step()
            self.regr.apply(self.non_neg_clipper)
            scheduler.step()
            # optimizer.zero_grad()
            for param in self.regr.parameters():
                param.grad = None

            # Check for plateau, if so: reset weights to try to find other minima
            if last_loss != None and epoch > self.num_epochs * 0.2:
                pct = (last_loss - loss_val) / abs(loss_val)
                if pct < self.early_stopping_pct and current_patience > patience_lim:

                    # Store loss and params if the loss is best
                    if best_loss == None:
                        best_loss = loss_val
                        best_params = self.regr.state_dict()
                    elif loss_val < best_loss:
                        best_loss = loss_val
                        best_params = self.regr.state_dict()

                    # Reset model weights
                    current_patience = 0
                    scheduler.load_state_dict(initial_state_dict)
                    last_loss = None
                    self.regr.initialize_weights_and_biases()
                    continue

            last_loss = loss_val
            current_patience += 1

        # load the best params
        self.regr.load_state_dict(best_params)
        last_loss = best_loss

        return

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts a new response (or responses) using the model parameters
        and the given new predictor data

        Args:
            X (np.ndarray): an array of shape NxM that contains new observations
                of predictor data to create predictions from

        Returns:
            np.ndarray: predictions created from the new predictor values
        """
        X = torch.from_numpy(X)  # torch.tensor(X, dtype=torch.float32)
        return (
            ((self.regr(X) * (self.y_max - self.y_min)) + self.y_min)
            .detach()
            .numpy()
            .flatten()
        )

    def create_onnx_graph(self) -> GraphProto:
        """Creates the ONNX graph that transforms an input layer containing
        new predictor data into a set of new responses/predictions using this
        regressor's fitted parameters.

        Returns:
            GraphProto: resulting ONNX graph
        """

        tempdir = tempfile.mkdtemp()
        tempf = os.path.join(tempdir, "temp")
        onnx_export = torch.onnx.export(
            self.regr,
            (torch.randn(1, self.regr.n_features),),
            f=tempf,
            input_names=["x"],
            output_names=["y1"],
            dynamic_axes={"x": [0]},
        )

        model = load(tempf)
        postprocess = make_graph(
            nodes=[
                make_node("Mul", ["yIn", "yStd"], ["post_1"]),
                make_node("Add", ["post_1", "yMean"], ["post_2"]),
                make_node("Reshape", ["post_2", "newShape"], ["y"]),
            ],
            name="NeuralNetPostProcess",
            inputs=[make_tensor_value_info("yIn", TensorProto.FLOAT, [None, 1])],
            outputs=[make_tensor_value_info("y", TensorProto.FLOAT, [None])],
            initializer=[
                make_tensor("newShape", TensorProto.INT64, [1], [-1]),
                make_tensor("yMean", TensorProto.FLOAT, [1], [self.y_min]),
                make_tensor("yStd", TensorProto.FLOAT, [1], [self.y_max - self.y_min]),
            ],
        )
        merged_graph = merge_graphs(model.graph, postprocess, io_map=[("y1", "yIn")])

        return merged_graph
