---
published: true
---
{% highlight javascript %}
document.write("JavaScript is a simple language for javatpoint learners");
{% endhighlight %}

## A New Post

Enter text in [Markdown](http://daringfireball.net/projects/markdown/). Use the toolbar above, or click the **?** button for formatting help.

def shape_list(x):
    """Deal with dynamic vectors cleanly."""
    # Get the static shape of the tensor as a list
    static = x.shape.tolist()
    # Get the dynamic shape of the tensor
    dynamic = x.shape
    # Return a new list where the i-th element is the size of the i-th dimension of x
    # If the size of the i-th dimension is not known (i.e., it is None in the static shape),
    # use the size from the dynamic shape instead
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]
    
    


      def shape_list(x):
          """Deal with dynamic vectors cleanly."""
          # Get the static shape of the tensor as a list
          static = x.shape.tolist()
          # Get the dynamic shape of the tensor
          dynamic = x.shape
          # Return a new list where the i-th element is the size of the i-th dimension of x
          # If the size of the i-th dimension is not known (i.e., it is None in the static shape),
          # use the size from the dynamic shape instead
          return [dynamic[i] if s is None else s for i, s in enumerate(static)]
