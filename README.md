# PEGASOS-algorithm-on-RCV1-Datatset
Use PEGASOS to train an SVM which separates the training data into it’s respective categories

1. Initialize all weights(w) to zero. Size of the weight parameter is 47236x1, i.e. equal to the number of features for each article.
2. Update ‘w’ upto ‘i’ iterations w.r.t to the computed gradient
3. Update value determination:
    a. Choose a subset of B data points form the training set
    b. Compute the predicted value of the selected data points with the current w. Select the points, labels with values              (label<predicted value>) < 1
    c. Compute the gradient on the selected false prediction points. Update the weights based on the computed gradient and the update value(neta)
    d. Calculate projected value of the new calculated ‘w’ and update according to the projection Repeat for ‘i’ iterations

Initialization:
Number of iterations = 100
Batch size = 1000
Regularization parameter ‘neta’ = 0.0000010

## Results:

<img width="514" alt="PEGASOS_REGULARIZATION" src="https://user-images.githubusercontent.com/43916672/63936274-0ed3e000-ca7d-11e9-98b9-e164423ba344.png">

The above plot describes the training error vs no. of iterations for different regularization parameters.
From the plot it can be observed that very high values of the regularization parameter results in almost zero update in the error. This is since the loss converges to the minimum at a very low pace such that the convergence is almost zero.
Very low values of the regularization parameter such as 1e-08 will increase the step size of the update to a very high value. This will result in missing the global minimum of the loss function which results in the oscillation of the loss.
From the graph we can observe that regularization value of 1e-06 is optimal for our solution since it converges quickly to the global minimum and gives the least error while compared to the other values.

<img width="721" alt="PEGASOS_Batch_Sizes" src="https://user-images.githubusercontent.com/43916672/63936331-30cd6280-ca7d-11e9-8226-b360f225ba0b.png">

The above second plot details the training error vs no. of iterations for different batch sizes. The batch size approximates all the points in the data set and gives an expectation of the loss is comparable to the loss acquired by gradient descent.
While a batch size of 3000 gives low error an converges quickly, we can also observe that a batch size of 1000 gives equally good results. Taking the batch size to be as low as possible will ensure that the process of updating our weights is not computationally expensive. Since the batch size of 3000 and 1000 both give better results than other sizes, it is advisable to use a batch size of 1000.
