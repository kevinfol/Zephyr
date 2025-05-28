#### regression_algorithm
The specific regression algorithm used to create forecast models. The options are:  

* Linear Models
    * `multiple_linear` -> Multiple Linear Regression using OLS
    * `ridge` -> Ridge Regression
    * `ransac` -> (RANdom SAmple Consensus) Regression
* Non-Linear Models
    * `k_nearest_neighbors` -> K-Nearest-Neighbors using 5 neighbors
    * `random_forest` -> Random Forest Regression
    * `neural_network` -> Artificial Neural Network with 1 hidden layer
    * `support_vector` -> Support Vector Machine