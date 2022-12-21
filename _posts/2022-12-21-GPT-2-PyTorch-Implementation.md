---
published: true
---
OpenAI released GPT-2 Tensorflow implementation, so we will implement it in PyTorch.


## Start with hyperparameters


def default_hparams():
    return HParams(
        n_vocab=0,  # total number of tokens
        n_ctx=1024,  # length of input sequence
        n_embd=768,  # length of embedding vector
        n_head=12,  # number of attention heads
        n_layer=12,  # the total number of layers in the model
    )



Dynamic tensors are tensors whose size or shape is not fixed when they are created.



## Deal with dynamic vectors cleanly.
    
{% highlight python %}

def shape_list(x):
    # get the static shape of the tensor as a list
    static = x.shape.tolist()
    # get the dynamic shape
    dynamic = x.shape
    # return a new list where the i-th element is the size of the i-th dimension of x
    # if the size of the i-th dimension is not known (i.e., it is None in the static shape),
    # use the size from the dynamic shape instead
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]
          
{% endhighlight %}


## Define softmax


{% highlight python %}

def softmax(x, axis=-1):
    # subtract the maximum value along the specified axis,
    # to ensure numerical stability and prevent overflow
    x = x - x.max(axis=axis, keepdim=True)[0]
    
    # calculate the exponentiated values of the tensor
    ex = torch.exp(x)
    
    # normalize the exponentiated values by dividing by the sum
    # along the specified axis, with the keepdim argument
    # keeping the original dimensions of the tensor
    return ex / ex.sum(axis=axis, keepdim=True)


{% endhighlight %}


## Define gelu

The gelu function calculates the GELU (Gaussian Error Linear Unit) activation function, which is a smooth approximation of the ReLU activation function. \
The function first calculates the square root of 2/pi and stores it in a variable called sqrt_2_over_pi. 
Then it calculates the intermediate term x + 0.044715 * x^3 and applies the tanh function to it. 
Finally, it multiplies the intermediate result by 0.5 and adds 1, and then multiplies this by the input tensor x to obtain the final result.


{% highlight python %}


def gelu(x):
    # calculate the square root of 2/pi
    sqrt_2_over_pi = np.sqrt(2 / np.pi)
    
    # calculate the intermediate term (x + 0.044715*x^3)
    intermediate = x + 0.044715 * x**3
    
    # calculate the tanh of the intermediate term
    tanh_intermediate = torch.tanh(sqrt_2_over_pi * intermediate)
    
    # calculate and return the final result
    return 0.5 * x * (1 + tanh_intermediate)

{% endhighlight %}


## Normalize input data

The norm function in GPT-2 performs normalization and an affine transform on the input tensor, allowing the model to learn a linear transformation of the data specific to the task. It is often used in the Transformer architecture to improve the stability and performance of the model.


{% highlight python %}

import torch

def norm(x, scope, *, axis=-1, epsilon=1e-5):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    with torch.no_grad():  # PyTorch does not have variable scopes, so we use torch.no_grad() to disable gradient calculation
        # get the number of elements along the specified axis
        n_state = x.shape[-1]

        # create trainable variables g and b with the specified shape
        g = torch.nn.Parameter(torch.ones(n_state))
        b = torch.nn.Parameter(torch.zeros(n_state))

        # calculate the mean and standard deviation along the specified axis
        u = x.mean(dim=axis, keepdim=True)
        s = ((x - u) ** 2).mean(dim=axis, keepdim=True)

        # normalize the input tensor and apply the affine transform
        x = (x - u) * torch.rsqrt(s + epsilon)
        x = x * g + b

        return x

{% endhighlight %}

The norm function first creates trainable variables g and b with the specified shape using the Parameter class from torch.nn. These variables will be updated during training. Then, it calculates the mean and standard deviation of the input tensor along the specified axis using the mean function. Finally, it normalizes the input tensor by subtracting the mean and dividing by the standard deviation, and applies the affine transform using the trainable variables g and b.

## Reshape the last dimension of x into [n, x.shape[-1]/n].

{% highlight python %}

def split_states(x, n):
    # Get the shape of the input tensor as a tuple of integers
    start = list(x.shape[:-1])
    # Calculate the size of the last dimension after reshaping
    m = x.shape[-1] // n
    # Reshape the tensor using the `view` method
    return x.view(*start, n, m)

{% endhighlight %}

The * operator expands a list into separate arguments, which is useful when passing a list to a function that expects separate arguments. The split_states function reshapes the last dimension of a tensor x into a shape of [n, x.shape[-1]/n], and uses the * operator to expand the list stored in start to separate arguments for the view method.