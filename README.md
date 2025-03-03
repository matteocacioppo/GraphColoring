# GraphColoring

# 🚀 Graph Neural Network for Graph Coloring Problems

## 📌 Introduction
This repository contains an implementation of a Graph Neural Network (GNN) designed to solve Graph Coloring Problems (GCPs). The model leverages the Deep Graph Library (DGL) to perform convolutional operations on graph structures.

## 🎨 Problem Statement
A graph coloring problem involves assigning different colors to directly connected nodes while minimizing the total number of colors used. This concept can be extended beyond colors to other feature constraints that must be satisfied in a similar way.

## ⚙️ Implementation Details
The model is designed to handle arbitrary graphs with a given number of nodes and edges. Each node is assigned a one-hot-encoded vector where each component represents the probability of the node being assigned a specific color. The number of components in the vector exceeds the minimum required number of colors.

### 🔢 Initialization
- The node color probability vectors are initialized using PyTorch’s `Embedding` layer.
- The weight matrix is drawn from a normal distribution with mean zero and variance one.
- These values are updated during backpropagation.

### 🏗️ Network Architecture
- The network consists of a single hidden layer.
- Node features are aggregated using the **mean aggregation** function:
  \[ h_v^k = \frac{\sum_{u \in N(v)} h_u^{k-1}}{|N(v)|} \]
- ReLU activation is applied after aggregation.
- A final aggregation step produces the output.
- A **softmax normalization** is applied to ensure the output values represent valid probabilities.

## 📉 Loss Function
The loss function is inspired by the Hamiltonian of the Potts model, a statistical physics model that generalizes the Ising model to multiple states (colors). The Hamiltonian is expressed as:

\[ H = \sum_{<i,j>} \mathbf{p}_i \cdot \mathbf{p}_j^T \]

where \( \mathbf{p}_i \) and \( \mathbf{p}_j \) are the probability vectors of nodes \( i \) and \( j \), and the sum runs over all adjacent nodes.

Minimizing this loss encourages neighboring nodes to have different dominant color components, effectively solving the coloring problem. The adjacency matrix is used to compute these interactions efficiently.

Additionally, an auxiliary loss function is used to track the number of incorrectly colored nodes. It counts how many adjacent nodes share the same assigned color but does not contribute to gradient updates.

## 🏋️ Training and Inference
- **Dropout** is applied to enhance generalization.
- Once training is complete, the final color assignment is determined using `argmax` on the output vectors, selecting the most probable color for each node.

## 📚 References
- **Original Paper**: [Graph Coloring with Graph Neural Networks](https://arxiv.org/pdf/2202.01606.pdf)
- **Original Code Repository**: [amazon-science/gcp-with-gnns-example](https://github.com/amazon-science/gcp-with-gnns-example)

This repository provides a foundation for research in applying GNNs to combinatorial optimization problems. Happy coding! 🚀
