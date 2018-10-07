# Optimization with Neural Processes

An implementation of [Neural Processes](https://arxiv.org/abs/1807.01622) paper in Python/Tensorflow.

Gaussian Processes (GP) have been used for a lot of interesting problems since they can capture a distribution over functions and provide uncertainty estimations. However, inability of GPs to scale with data led to much more prevalent utilization of Deep Learning for large datasets. A recent paper by DeepMind provides a novel approach called Neural Processes (NP), which combines the probabilistic nature of GP with scalability of Deep Learning. In particular, at sampling time, NP requires only a single pass through a feedforward Neural Network (NN), scaling as O(n) with data, while GP scales as O(n^3). In this project I provide an implementation of the NP in Python/Tensorflow, and use it for several applications to study the behavior of the method with different types of data. 
