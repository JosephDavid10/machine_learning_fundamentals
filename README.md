# machine_learning_fundamentals
Exploring machine learning techniques and applications.

## pytorch_cnn_cifar10.py
|version|Achitecture|batch|optimizer|filters|fc_neurons|accuracy|observations|
|---|---|---|---|---|---|---|---|
|v1.0|CNN|16|SGD|6,16|120,84|61.70%||
|v1.1|CNN|16|SGD|32,64|128,84||| expanding with more filters
|v1.2|CNN|32|SGD|32,64|128,84||| batch_size = 32
|v1.3|CNN|16|ADAM|32,64|128,84||| testing adam optimizer
|v1.4|CNN|16|SGD|32,64|128,64,32||| testing with a new hidden layer
