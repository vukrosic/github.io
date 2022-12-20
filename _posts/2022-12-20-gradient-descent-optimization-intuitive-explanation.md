---
published: true
---
In this image you can see the functions of our model's predicted outputs and the correct outputs.

![1 Two functions22.jpg](https://raw.githubusercontent.com/vukrosic/vukrosic.github.io/master/BlogImages/Gradient%20Descent/1%20Two%20functions22.jpg)


$$f(x)_{predicted}$$ are the outputs of our model given $$x$$ as input data.
$$f(x)_{correct}$$ are the correct outputs given $$x$$ as input data.

# Linear Functions

To make things simple, let’s make these two functions linear.

![3 Linear functions.jpg](https://github.com/vukrosic/vukrosic.github.io/blob/master/BlogImages/Gradient%20Descent/3%20Linear%20functions.jpg?raw=true)


Both of the have the formula:

## $$f(x)=weight*x+bias$$.

Just $$weight$$ and $$bias$$ values are different.

During the training, we know the values of $$x$$, the input data, and $$f(x)$$, the output, and we need to find $$weight$$ and $$bias$$ (slope and offset) to overlap these 2 functions.

Now you might spot the problem - we have 1 equation and 2 unknowns, $$weight$$ and $$bias$$. In this situation, we need to use a big computer to guess as many weight and bias values as possible and find the ones that match the correct values as best as possible.

To measure the difference between **$$f(x)_{predicted}$$**  and  **$$f(x)_{correct}$$** , we will just **subtract** them and get the **absolute value**. This is a loss function called Mean **Absolute Error (_MAE_)**.

# Mean Absolute Error

For the next graph, we will choose a constant input and output value, and we will only change the weight of the model. We can see how the **MAE** is changing depending on the $$weight$$.

For a certain value of the $$weight$$ **MAE** is 0, meaning that at that point our model correctly predicts the output.

![5 calculating Weight.jpg](https://github.com/vukrosic/vukrosic.github.io/blob/master/BlogImages/Gradient%20Descent/6%20MAE2.jpg?raw=true)


To find this weight value, we will start at a random point on the function and calculate the derivative (slope). Then we will step to the left or right for a certain distance in the direction of the slope’s decrease and repeat the process. This stepping distance is called **learning rate**. 

# Learning Rate

A small **learning rate** (step) will take too long to find the minimum, but a large one can overshoot the minimum. To combat this, we can make the **learning rate** proportional to the slope. We do this until the slope is 0.


In this simple function, this will get us to the global minimum, but in a more complex function we might end up in a local minimum, which is a much harder problem to solve.

# More Dimensions

What if the function has more inputs, for example **$$x$$** and **$$y$$**, and **$$z$$** as the output?

To calculate the step direction in this case we need to use a **gradient**. A gradient of a function gives us the direction of the steepest ascent, and if we take the negative, we will get the **steepest descent**. This optimization algorithm is called > gradient descent. Moreover, the length of this direction vector is proportional to the steepness at that point.

# Gradient Descent

Now we will calculate the gradient:
Let’s say we have a 3-dimensional space where the $$z$$ coordinate of the point is given as a function of $$x$$ and $$y$$ coordinates.

$$z = f(x,y) = x^2\sin(y)$$

To calculate the gradient at this point, we need to find two partial derivatives.
First the partial derivative of $$f$$ with respect to $$x$$. This only means that we find the derivative of this function while pretending that $$y$$ is constant.

$$\frac{\partial f}{\partial x} = 2x\sin(y)$$

then the partial derivative of $$f$$ with respect to $$y$$

$$\frac{\partial f}{\partial x} = x^2\cos(y)$$ 


The **gradient** of $$f(x,y)$$ is simply a vector of these 2 values: 


$$\nabla f(x,y) = \begin{bmatrix}\frac{\partial f}{\partial x} \\\\ \frac{\partial f}{\partial y}
\end{bmatrix} = \begin{bmatrix} x^2\cos(y) \\\\ 2x\sin(y) \end{bmatrix}$$

Then simply calculate the next weight:

### $$weight = weight - learning_rate * gradient$$ 


This is the gradient of the function $$f$$ at the coordinates $$x$$ and $$y$$. The process is the same for any number of dimensions.

Hopefully, this helped you understand the math behind the gradient descent. 

Vuk Rosić,
vukrosic1@gmail.com

-------------------------
