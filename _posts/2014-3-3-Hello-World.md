---
layout: post
title: I learn one complex ML topic every day
published: true
---
<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>

> As I stood at the threshold of my new life, I couldn't help but feel a sense of longing for the carefree days of my youth. But I knew that my determination to see a world with a benign artificial intelligence, one that could coexist peacefully with humanity, required sacrifice. I was prepared to trade my youth and every waking hour to work towards this goal. It was a dream worth striving for, even if it meant sacrificing everything else.


There are many optimizational algorithms in machine learning, some of which are:

Gradient descent updates weights and biases so that the difference (loss) between the calculated output and the correct output gets smaller. 

It first calculates the derivative (slope) at the current output value from the function that is drawn from all previous value, and updates parameters such that the next output value is in the direction of the slope decrease.

Partial derivative of $$f(x,y)$$ means we compute the derivative of $$f(x)$$ while pretending that $$y$$ is a constant.

$$f(x,y) = x^2sin(y)$$

First we take the partial derivative of $$f$$ with respect to $$x$$ (so we treat y as a constant):

$$\frac{\partial f}{\partial x} = 2xsin(y)$$

then we take a partial derivative of $$f$$ with respect to $$y$$
$$\frac{\partial f}{\partial y} = \begin{bmatrix} x^2\cos(y) \\ 2x\sin(y) \end{bmatrix}$$
$$\frac{\partial f}{\partial y} = $$\begin{bmatrix} x^2\cos(y) \\ 2x\sin(y) \end{bmatrix}



$$\nabla f = \begin{bmatrix} x^2\cos(y) \\ 2x\sin(y) \end{bmatrix}$$
-------------------------


1. Stochastic gradient descent (SGD): SGD is a variant of gradient descent that computes the gradients using a single sample (or a small batch of samples) at a time. It is often used in large-scale machine learning problems because it can be implemented more efficiently than batch gradient descent, which computes the gradients using the entire dataset.


2. Mini-batch gradient descent: Mini-batch gradient descent is a variant of gradient descent that computes the gradients using a small batch of samples at a time. It is a trade-off between SGD and batch gradient descent, offering a balance between the efficiency of SGD and the stability of batch gradient descent.































## Neural Network From Scratch Using NumPy

In this post, we will learn about the math behind machine learning and use Python to build a small library for creating neural networks with various layers (such as fully connected and convolutional layers).

In this post, we will dive into the math behind neural networks and gain an intuitive understanding of why certain techniques are used.

To make our feed forward network in NumPy, we will do it in the next steps:

1. Input data is fed into the neural network.

2. It flows layer to layer, undergoing transformations at each layer.

3. We compare the output with the correct value from the training data, and we calculate the error as a **scalar**.

4. A parameter (such as a weight or bias) is adjusted by subtracting the derivative of the error with respect to the parameter. This helps to improve the accuracy of the network by modifying the transformations applied by the layers.

5. The process is repeated, with the updated parameters being used to generate new predictions or decisions. This continues until the network's performance meets the desired level of accuracy.
