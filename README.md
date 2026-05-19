# machine_learning_fundamentals
Exploring machine learning techniques and applications.

## pytorch_cnn_cifar10.py
|version|Architecture|batch|optimizer|filters|hidden_fc_neurons|accuracy|observations|class_results|
|---|---|---|---|---|---|---|---|---|
|v1.0|CNN|16|SGD|6, 16|120, 84|61.70%|||
|v1.1|CNN|16|SGD|32, 64|128, 84|71.11%| more filters| <details><summary> Classes</summary> plane: 82.3%<br> car: 82.4%<br> bird: 61.9%<br> cat: 48.2%<br> deer: 65.5%<br> dog: 66.9%<br> frog: 73.9%<br> horse: 79.1%<br> ship: 87.6%<br> truck: 63.3%</details> |
|v1.2|CNN|32|SGD|32, 64|128, 84|||| 
|v1.3|CNN|16|ADAM|32, 64|128, 84|||| 
|v1.4|CNN|16|SGD|32, 64|128, 64, 32||new hidden layer||
