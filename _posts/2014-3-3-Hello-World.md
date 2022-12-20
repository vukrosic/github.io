---
layout: post
title: Gradient Descent Optimization - An Intuitive Explanation
published: true
---
<!-- MathJax -->

<script type="text/javascript"

  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.3/MathJax.js?config=TeX-AMS-MML_HTMLorMML">

</script>

> As I stood at the threshold of my new life, I couldn't help but feel a sense of longing for the carefree days of my youth. But I knew that my determination to see a world with a benign artificial intelligence, one that could coexist peacefully with humanity, required sacrifice. I was prepared to trade my youth and every waking hour to work towards this goal. It was a dream worth striving for, even if it meant sacrificing everything else.

# Gradient Descent Optimization - An Intuitive Explanation



In this image you can see the functions of our model's predicted outputs and the correct outputs.

![1 Two functions22.jpg]({{site.baseurl}}/_posts/1 Two functions22.jpg)


f(x)_predicted are the outputs of our model given x as input data. f(x)_correct are the correct outputs given x as input data.


To make things simple. Let’s make these two functions linear.

![2 gradient descentwe.jpg]({{site.baseurl}}/_posts/2 gradient descentwe.jpg)
![3 Linear functions.jpg]({{site.baseurl}}/_posts/3 Linear functions.jpg)


Both of the have the format $$f(x)=weight*x+bias$$.


During the training, we know the values of x, the input data, and f(x), the output, and we need to find weight and bias (slope and offset) to overlap these 2 functions.

Now you might spot the problem - we have 1 equation and 2 unknowns, weight and bias. In this situation, we need to use big computers to guess as many weight and bias values as possible and find the ones that match the correct function as best as possible.

To measure the difference between f(x)_predicted and f(x)_correct, we will just subtract them and get the absolute value. This is a loss function called Mean Absolute Error (MAE).

For the next graph, we will choose a random input and output value and keep them constant, and we will only change the weight of the model. We can see how the MAE is changing depending on the weight. For a certain value of the weight, MAE is 0, meaning that at that point our model correctly predicts the output.

![4 calculating W.jpg]({{site.baseurl}}/_posts/4 calculating W.jpg)


To find this weight value, we will start at a random point on the function and calculate the derivative (slope). Then we will shift to the left or right, in the direction of the slope’s decrease. We will step in that direction for a certain distance and measure again. This distance is called learning rate. A small learning rate (step) will take too long to find the minimum, but a large one can overshoot the minimum. To combat this, we can make the learning rate proportional to the slope. We do this until the slope is 0.


In this simple function, this will get us to the global minimum, but in a more complex function we might end up in a local minimum, which is a much harder problem to solve.

What if the function has more inputs, for example x and y, and z as an output? To calculate the step direction in this case we need to use a gradient. A gradient of a function gives us the direction of the steepest ascent, and if we take the negative, we will get the steepest descent. This optimization algorithm is called gradient descent. Moreover, the length of this direction vector is proportional to the steepness at that point.

Now we will calculate the gradient:
Let’s say we have a 3-dimensional space where the z coordinate of the point is given as a function of x and y coordinates.

z = f(x,y) = x^2*sin(y)

To calculate the gradient at this point, we need to find two partial derivatives.
First the partial derivative of f with respect to x. This only means that we find the derivative of this function while pretending that y is constant.

$$\frac{\partial f}{\partial x} = 2xsin(y)$$

then the partial derivative of f with respect to y

$$\frac{\partial f}{\partial x} = x^2cos(y)$$

the gradient of f(x,y) is simply a vector of these 2 values


The final formula:

\nabla f(x,y) = \begin{bmatrix}\frac{\partial f}{\partial x} \\\\ \frac{\partial f}{\partial y} \end{bmatrix}= \begin{bmatrix} x^2\cos(y) \\\\ 2x\sin(y) \end{bmatrix}

This is the gradient of the function f at the coordinates x and y. The process is the same for any number of dimensions.

Hopefully, this helped you understand the math behind the gradient descent. 

Vuk Rosić,
vukrosic1@gmail.com
















Gradient is a function that takes in a point in a two-dimentional space and outputs an output.


![gradient descent]({{site.baseurl}}/_posts/2 gradient descent.jpg)


The first question I had was "Why don't we just optimize parameters so the less functoin goes straight down to 0? Why are we slowly decreasing it?"


In theory, it is possible for the loss to be zero, but this would occur only if the predicted outputs of the model are exactly equal to the true outputs for every input in the dataset.

In practice, this is extremely hard for big models, even on the training data, because there will always be some level of noise in the data, or incomplete or inaccurate information. However, it is possible to train a model to have a very low loss. The ultimate goal of training a model is to find the set of parameters that produces the lowest possible loss, which will likely result in a model that performs well on unseen data.




Function $$y$$ will represent correct outputs for $$x$$ data as inputs, and function $$f(x)$$ will be the network's predicted outputs for $$x$$ data as input. We are trying to get function $$f(x)$$ to (as closely as possible) overlap the function $$y$$. Keep in mind that it might be problematic if they are too similar, but that is beyond this beginner's tutorial.


Next, we need to calculate how to minimize the difference (loss) function, which will bring previous $$f(x)_{predicted} and f(x)_{correct}$$ closer together.

To do this we will simply calculate in which direction
The question I has was ""

We could optimize it at one points to the value of 0, but the optimization may not keep it at 0, as this function can start going up again as the difference between $$f(x)_{predicted} and f(x)_{correct}$$ starts increasing.





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
