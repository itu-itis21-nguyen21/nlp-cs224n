import numpy as np

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad

def softmaxCostAndGradient(predicted, target, outputVectors):
    """ Softmax cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, assuming the softmax prediction function and cross
    entropy loss.

    Arguments:
    predicted -- numpy ndarray, predicted word vector (\hat{v} in
                 the written component)
    target -- integer, the index of the target word
    outputVectors -- "output" vectors (as rows) for all tokens
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    gradPred -- the gradient with respect to the predicted word
           vector
    grad -- the gradient with respect to all the other word
           vectors

    We will not provide starter code for this function, but feel
    free to reference the code you previously wrote for this
    assignment!
    """

    ### YOUR CODE HERE
    predicted = predicted[:, np.newaxis]

    # Make one-hot vector y
    y = np.zeros((outputVectors.shape[0], 1))
    y[target, :] = 1

    # Forward propagation - calculate cost
    y_hat = softmax(np.dot(outputVectors, predicted))
    cost = - np.multiply(y, np.log(y_hat))

    # Backward propagation - caculate gradients
    gradPred = np.dot(outputVectors.T, y_hat - y)
    grad = np.dot(predicted, (y_hat - y).T)

    ### END YOUR CODE

    return cost, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors,
                               K=10):
    """ Negative sampling cost function for word2vec models

    Implement the cost and gradients for one predicted word vector
    and one target word vector as a building block for word2vec
    models, using the negative sampling technique. K is the sample
    size.

    Note: See test_word2vec below for dataset's initialization.

    Arguments/Return Specifications: same as softmaxCostAndGradient
    """

    indices = [0, 1, 3]

     ### YOUR CODE HERE
    predicted = predicted[:, np.newaxis]

    # Filter the negative samples from outputVectors
    pos_word_vec = outputVectors[indices[0], :][np.newaxis, :]
    neg_word_vec = outputVectors[indices[1:], :]

    # Compute useful quantities
    pos_word_vec_sig = sigmoid(np.dot(pos_word_vec, predicted))
    neg_word_vec_sig = sigmoid(np.dot(-neg_word_vec, predicted))

    # Forward propagation - compute cost
    pos_word_vec_cost = np.log(pos_word_vec_sig)
    neg_word_vec_cost = np.sum(np.log(neg_word_vec_sig))
    cost = - pos_word_vec_cost - neg_word_vec_cost

    # Backward propagation - compute gradients
    print(pos_word_vec_sig.shape)
    print(pos_word_vec.T.shape)
    gradPred = np.dot((pos_word_vec_sig - 1), pos_word_vec) - np.dot((neg_word_vec_sig - 1).T, neg_word_vec)

    grad = np.zeros(outputVectors.shape)
    grad[indices[0], :] = np.dot((pos_word_vec_sig - 1), predicted.T)
    grad[indices[1:], :] = - np.dot((neg_word_vec_sig - 1), predicted.T)
    ### END YOUR CODE

    return cost, gradPred, grad

predicted = np.array([-0.018, 0.404, -0.317])
outputVectors = np.array([(0.204, -0.007, -0.733), (-0.706, 0.745, 0.002), (-0.492, -0.709, 0.133), (-0.628, -0.073, 0.502)])
target = 1

predicted = predicted[:, np.newaxis]
predicted = predicted.reshape(3,)
print(predicted.shape)