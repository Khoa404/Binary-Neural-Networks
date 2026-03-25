# BNN Research Reproduction (MNIST)

This repository contains my experimental implementation of a **Stochastic/Binary Neural Network (BNN)** for MNIST classification, developed for learning and research purposes.

## Academic Attribution (Important)

This project is **based on concepts and methodology** from the following paper:

> Babak Golbabaei, Yirong Kan, Renyuan Zhang, Yasuhiko Nakashima,  
> **"Efficient Hardware Implementation of Robust Binary Neural Networks Using End-to-End Unipolar Representation"**  
> IEICE Transactions on Fundamentals  
> Link: https://globals.ieice.org/en_transactions/fundamentals/10.1587/transfun.2025VLP0002/_f

### Disclaimer

- This is an **independent re-implementation**, not the authors’ official source code.
- Some details (training setup, hyperparameters, architecture choices) may differ from the original publication.
- All credit for the original idea and method belongs to the paper authors.

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

## Citation

If you use this repository in academic work, please cite the original paper first.

```text
Golbabaei, B., Kan, Y., Zhang, R., & Nakashima, Y.
Efficient Hardware Implementation of Robust Binary Neural Networks Using End-to-End Unipolar Representation.
IEICE Transactions on Fundamentals.
```
