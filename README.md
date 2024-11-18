# Computational Intelligence Laboratory

Welcome to the **Computational Intelligence Laboratory** repository! This collection contains implementations and exercises from various labs exploring the fundamentals of computational intelligence, including neural networks, clustering algorithms, and reinforcement learning techniques.

## Table of Contents

- [Lab 1: Introduction to Python and MATLAB](#lab-1-introduction-to-python-and-matlab)
- [Lab 2: Perceptron Implementation](#lab-2-perceptron-implementation)
- [Lab 3: Multi-Layer Neural Network](#lab-3-multi-layer-neural-network)
- [Lab 4: k-Means Clustering](#lab-4-k-means-clustering)
- [Lab 5: Radial Basis Function (RBF) Networks](#lab-5-radial-basis-function-rbf-networks)
- [Lab 6: Hopfield Networks](#lab-6-hopfield-networks)
- [Lab 7: Convolutional Neural Networks (CNNs)](#lab-7-convolutional-neural-networks-cnns)
- [Lab 8: Recurrent Neural Networks (RNNs)](#lab-8-recurrent-neural-networks-rnns)
- [Lab 9: Neural Networks for Static and Continuous Identification](#lab-9-neural-networks-for-static-and-continuous-identification)

---

## Lab 1: Introduction to Python and MATLAB

This lab introduces Python and MATLAB programming, focusing on:
- Defining and using functions.
- Handling arrays and matrices.
- Plotting 2D and 3D graphs.
- Understanding similarities and differences between Python and MATLAB.

**Key Task:** Implement a function in both Python and MATLAB to compute the sigmoid and its derivative.

---

## Lab 2: Perceptron Implementation

Explore the basics of perceptrons:
- Understand their structure and relation to biological neurons.
- Implement perceptron learning algorithms in Python.
- Test the perceptron on real datasets (e.g., breast cancer dataset).

### Additional Tasks:
1. **Draw Computational Graph**:  
   - Represent the perceptron model as a computational graph.  
   - Manually calculate the gradients for the weights and biases.

2. **Data Augmentation for XOR Data**:  
   - Use data augmentation techniques to transform the XOR dataset.  
   - Analyze how these transformations affect the perceptron’s performance.

3. **Examine Limitations of Perceptron**:  
   - Investigate scenarios where the perceptron fails (e.g., solving non-linearly separable problems like XOR).  
   - Document the reasons for its limitations and propose potential solutions.

**Key Task:** Develop and test a perceptron model for binary classification problems.


---

## Lab 3: Multi-Layer Neural Network

Dive into multi-layer neural networks:
- Understand the limitations of perceptrons and the need for multi-layer models.
- Implement a 2-layer neural network in Python.
- Explore the backpropagation algorithm.

### Additional Tasks:
1. **Extend the Computational Graph**:  
   - Expand the computational graph to represent a multi-layer perceptron (MLP).  
   - Illustrate how forward propagation and backpropagation flow through multiple layers.

2. **General MLP Class Implementation**:  
   - Create a general-purpose MLP class from scratch, without using high-level frameworks.  
   - Implement layers, activation functions, forward propagation, and backpropagation manually.

3. **Train, Test, and Validate**:  
   - Write a clear workflow for training, testing, and validating the MLP.  
   - Include hyperparameter tuning, such as learning rate and number of epochs.  
   - Evaluate the model’s performance on different datasets (e.g., XOR, synthetic datasets).

**Key Task:** Create a 2-layer network and evaluate its performance on the XOR problem.


---

## Lab 4: k-Means Clustering

Learn the fundamentals of unsupervised learning:
- Understand k-means clustering and its applications.
- Implement the algorithm in Python from scratch.
- Test the implementation on synthetic datasets.

### Additional Tasks:
1. **Compress Images Using k-Means**:  
   - Apply the k-means algorithm to reduce the color palette of an image.  
   - Compare the original and compressed images, and analyze the quality trade-offs with different numbers of clusters.

2. **CUDA Implementation for Speed-Up**:  
   - Implement a CUDA version of the k-means algorithm for parallel execution on GPUs.  
   - Write kernel code to optimize the centroid computation and point assignment steps.  
   - Measure and compare the performance of the CUDA implementation with the CPU version.

**Key Task:** Evaluate clustering performance and limitations by varying the number of clusters.
y Task:** Evaluate clustering performance and limitations by varying the number of clusters.

---

## Lab 5: Radial Basis Function (RBF) Networks

Explore the structure and applications of RBF networks:
- Understand how RBF differs from multi-layer neural networks.
- Use Gaussian functions as activation functions in the hidden layer.
- Apply RBF networks for function approximation.

**Key Task:** Build and test an RBF network for regression tasks.

---

## Lab 6: Hopfield Networks

Study Hopfield networks and their applications:
- Analyze the memory storage and retrieval mechanism of Hopfield networks.
- Implement the Hopfield model for solving small optimization problems.

### Additional Task:
1. **Reconstruct Alphabets and Digits**:  
   - Use a Hopfield network to store and reconstruct patterns of alphabets and digits.  
   - Test the network's ability to recover noisy or incomplete inputs.  
   - Analyze the reconstruction performance and identify the limitations of the network.

**Key Task:** Simulate a Hopfield network to solve simple associative memory tasks.


---

## Lab 7: Convolutional Neural Networks (CNNs)

Understand CNNs for image recognition tasks:
- Learn the architecture and working principles of CNNs.
- Implement CNNs in Python using TensorFlow or PyTorch.
- Test CNNs on simple image datasets.

**Key Task:** Train a CNN for image classification and evaluate its performance.

---

## Lab 8: Recurrent Neural Networks (RNNs)

Delve into RNNs for sequential data:
- Understand RNN architectures and their applications.
- Build and train an RNN for time-series prediction.
- Explore different RNN configurations.

### Additional Tasks:
1. **Experiment with RNN Variants**:  
   - Implement simple RNN, LSTM, and GRU architectures from scratch or using frameworks like TensorFlow or PyTorch.  
   - Compare their performance on various tasks.

2. **Work with Classification and Regression Data**:  
   - **Classification**: Use the [IMDB Sentiment Analysis dataset](https://github.com/lakshay-arora/IMDB-sentiment-analysis-master) to train a model for sentiment classification.  
   - **Regression**: Use the [Daily Minimum Temperatures in Melbourne dataset](https://www.kaggle.com/datasets/paulbrabban/daily-minimum-temperatures-in-melbourne) for time-series regression.  
   - Evaluate the models on metrics suitable for each task and compare their results.

**Key Task:** Train an RNN for a sequence prediction task and analyze its results.


---

## Lab 9: Neural Networks for Static and Continuous Identification

Explore hybrid neural network architectures:
- Understand static and continuous identification problems.
- Implement neural networks to address these tasks.
- Combine different models for improved performance.

### Additional Tasks:
1. **Work with the Python Control Library**:  
   - Use the [Python Control Systems Library](https://python-control.readthedocs.io/) to model and analyze control systems.  
   - Explore its features to simulate dynamic systems and control designs.

2. **Identify Transfer Function of a System**:  
   - Implement different methods to identify the transfer function of a system:  
     - **Static Identifier**: Use a neural network to identify the static behavior of the system.  
     - **Parallel Identifier**: Combine multiple identifiers for improved accuracy and robustness.  

3. **State-Space System Identification**:  
   - Use neural networks to estimate the state-space representation of a system.  
   - Validate the identified state-space model by comparing its response with the actual system dynamics.

**Key Task:** Build and evaluate a neural network model for static and continuous data identification.


---


