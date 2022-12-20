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

# Intuitive Understand Of Gradient Descent As Optimization Algorithm

Gradient descent updates weights and biases so that the difference (loss) between the calculated output and the correct output gets smaller. 

It first calculates the derivative (slope) at the current output value from the function that is drawn from all previous value, and updates parameters such that the next output value is in the direction of the slope decrease.

For this we will need 2 functions.

![2 funcs]({{site.baseurl}}/_posts/1 Two functions2.jpg)



Now let's plot the differences between those 2 function on another graph - x represent the same input values, but y axis is the (absolute) difference between calculated output $$f(x)$$ and corrent output $$y$$.

![gradient descent]({{site.baseurl}}/_posts/2 gradient descent.jpg)

(something like this)

As we can see, lower the $$y_{diff}$$ means the difference between f(x)_{predicted} and f(x)_{correct} is smaller, that is we get a better approximation. Now there are other problems if they match too closely, but that is beyond this beginner tutorial.




Function $$y$$ will represent correct outputs for $$x$$ data as inputs, and function $$f(x)$$ will be the network's predicted outputs for $$x$$ data as input. We are trying to get function $$f(x)$$ to (as closely as possible) overlap the function $$y$$. Keep in mind that it might be problematic if they are too similar, but that is beyond this beginner's tutorial.


Next, we need to calculate how to minimize the difference (loss) function, which will bring previous f(x)_{predicted} and f(x)_{correct} closer together.

To do this we will simply calculate in which direction
The question I has was "Why don't you just optimize parameters so the less functoin goes straight down?, Why are we slowly decreasing it?"

We could optimize it at one points to the value of 0, but the optimization may not keep it at 0, as this function can start going up again as the difference between f(x)_{predicted} and f(x)_{correct} starts increasing.



This is because the loss function is already an existing function and if at one point we put it as 0, the next point it can start growing. We are trying to minimize it so no matter how randomly it starts going up and down, we need to fine tune the parameters so it always stays close.


The answer is simple, we don't know exactly how much down we should go, and we shouldn't overshoot to below zero. We can control how much down we go with **learning rate**.

To calculate how to move the next $$f(x)$$ point towards the next $$y$$ point we need calculate the closest distance, or the direction vector from $$f(x)$$ to $$y$$.

Partial derivative of $$f(x,y)$$ means we compute the derivative of $$f(x)$$ while pretending that $$y$$ is a constant.



First we take the partial derivative of $$f$$ with respect to $$x$$ (so we treat y as a constant):

$$\frac{\partial f}{\partial x} = 2xsin(y)$$

then we take a partial derivative of $$f$$ with respect to $$y$$
$$\frac{\partial f}{\partial y} = \begin{bmatrix} x^2\cos(y) \\ 2x\sin(y) \end{bmatrix}$$
$$\frac{\partial f}{\partial y} = $$\begin{bmatrix} x^2\cos(y) \\ 2x\sin(y) \end{bmatrix}



$$\nabla f = \frac{\partial f}{\partial x} = \begin{bmatrix} x^2\cos(y) \\ 2x\sin(y) \end{bmatrix}$$
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
