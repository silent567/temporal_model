# temporal_model
Neural network models operating on temporal signals of variable lengths implemented in Tensorflow. It supports input signals with different temporal length, and randomly sampling signals into fixed-length during training processes (like online data-argumentation). Functions are intergrated in the network module including train, test, and cross_val. 

All neural network models are based on the VirtualNN class in virtualNeuralNetwork.py . It defines functions such as train, test and cross_val. The networks are constructed in each specific file in build_model function.
For each network, there is a default hyper-parameter class and hyper-parameter range class. The first one is for testing the selected hyper-parameter, while the second one is utilized for random search within the selected range. Feel free to ignore it if you don't use random search for tuning hyper-parameters.

The repo requires three more repos in the same directory: [cuda_utils](https://github.com/silent567/cuda_utils), [nn_parts](https://github.com/silent567/nn_parts) and [liblinear](https://github.com/cjlin1/liblinear)
