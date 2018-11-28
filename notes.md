# Notes from Goodfellow textbook

[Deep Learning](https://www.deeplearningbook.org/)

## Linear alebgra

Scalar - a single number

Vector - array of numbers, thing of them as identifying points in space, each giving a coordinate along a different axis

Matrix (matrices) - A 2-d array where each element is identified by two indices. $A_{i,:}$ is the i-th row. $A_{:,i}$ is the i-th column

Tensor - an array with more than two axis. In general case, an array of 

numbers on a regular grid with a variable number of axes is a tensor. $A_{i,j,k}$

Transpose operation of matrix:  $(A^T)_{i,j} = A_{j,k}$

A vector can be thought of a matrix with only one column. You can denote a vector as a transpose of a row matrix:

$ x = [x_1, x_2, x_3 ] ^T$

A scalar can be thought of a matrix with only one entry whose tranpose is itself: $a=a^T$

Add two matrices together is possible as long as they have the same shape: $ C = A + B$, where $C_{i,j} = A_{i,j} + B_{i,j}$

Multiply a matrix by a scalar or add a scalar to a matrix: $ D = a \times B + c $ where $D_{i,j} = a \times B_{i,j} + c$

In addition, adding a vector to a matrix is *broadcasting*: $ C= A +b $ where $C_{i,j} = A_{i,j} + b$



## Gradient based optimization

A derivative of a function tells how much a change in x  results in a change in y. 

$F(x) = \frac{1}{2}x^2$

$F^\prime(x) = x$

When the derivative is equal to zero is called a critical point. A global minimum (maximum) is where small changes in x do not result in decrease (increase) of the funtion. 

In deep learning we generally want to minimize functions that have multiple inputs: $f: \mathbb{R}^n  \mathbb{R}$
We still need such minmizations to map to a single, scalar output. 

The [gradient](https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/gradient-and-directional-derivatives/v/gradient) is a vector of partial derivatives denoted as $\nabla_xf(x)$

The gradient points you in the direction of steepest ascent. The length of the gradient vector (the vector field representing the gradient) tells you the steepness of the direction of steepest ascent. 


## Gradient based learning 

A machine learning algorithm has 3 parts (Section 5.10): 

1. optimization procedure ( stochastic gradient descent, [RMSProp](https://towardsdatascience.com/a-look-at-gradient-descent-and-rmsprop-optimizers-f77d483ef08b)  )

2. Cost function ( for example the [negative log-likelihood](https://ljvmiranda921.github.io/notebook/2017/08/13/softmax-and-the-negative-log-likelihood/#nll) also known as cross-entropy

3. A model family (linear ,non-linear?)


## Deep Feedforward Networks

These are also called multilayer perceptrons (MLPs). We have a function that maps input $x$ to label/category $y$
The network learns the parameters $\theta$  to define the mapping $y=f(x,\theta)$  that result in the best function approximation. 

The feedforward network is associated with a DAG describing how functions are composed together into layers. For example a 3 layer network is in the form $f(x)=f^{(3)}(f^{(2)}(f^{(1)})x)))$ The final layer is the output layer. The training data specifies what the output layer should produce; it doesn't indicate what the other layers should produce. Hence, these layers are *hidden*. These hidden layers must have activation functions that will be used to compute hidden layer values.  
