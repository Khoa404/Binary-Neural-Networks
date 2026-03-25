# BNN (Binary Neural Network)

From the article: `"Efficient Hardware Implementation of Robust Binary Neural Networks Using End-to-End Unipolar Representation"` by Babak GOLBABAEI, ​​Yirong KAN, Renyuan ZHANG, Yasuhiko NAKASHIMA [Link](https://globals.ieice.org/en_transactions/fundamentals/10.1587/transfun.2025VLP0002/_f)

## BNN.py

The **BNN.py** file implements a Stochastic Binary Neural Network optimized for MNIST image classification tasks.

### Key Components:

- **UnipolarSte**: Straight-Through Estimator function for binary weight quantization in the backward pass
- **MPTSInputEncoding**: Input encoding using stochastic voting with multiple SGNs (Stochastic Number Generators)
- **TrainableThresholdActivation**: Learnable threshold activation function with sigmoid annealing for gradual network binarization
- **SBNLinear**: Binary linear layer with step-wise binarized weights
- **SBN_SF_Net_Optimized**: Complete neural network model with architecture 784 → 256 → 10 optimized for MNIST performance

### Features:

- Data-Aware Initialization technique for improved convergence
- Efficient computation through binary weight operations
- Training on MNIST dataset with optional GPU support
- Configurable batch size, learning rate, and training epochs

### Training:

- Epoch: 365 | Loss: 0.0890 | Steepness: 9.0
>>> Test Accuracy: 95.05%

<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/fc52308c-b3c1-4cab-b80f-ea71aed263cc" />
