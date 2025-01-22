from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #1. 점수계산 -> 2. 각 인스턴스의 점수를 모두 지수화 -> 3. 정답항만 normalize
    # -> 4. 정답항만 -log취해주기.
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train) :
      scores = np.dot(X[i],W)
      scores = np.exp(scores)
      scores = scores / np.sum(scores)
      loss += -np.log(scores[y[i]])

      #Gradient 계산
      for j in range(num_classes) :
        if j == y[i] : # 정답 클래스
          dW[:, j] += (scores[j] - 1) * X[i]
        else:  # 비정답 클래스
          dW[:, j] += scores[j] * X[i]
    #평균 손실&그라디언트 계산
    loss /= num_train
    dW /= num_train

    #정규화 추가
    loss += reg * np.sum(W * W)
    dW += 2*reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # Number of training examples
    num_train = X.shape[0]

    # 1. 점수 계산 및 숫자 안정성 확보
    scores = np.dot(X, W)

    # 2. Softmax 확률 계산
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 3. 손실 계산
    loss = -np.sum(np.log(probs[np.arange(num_train), y]))
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)  # 정규화 추가

    # 4. 그라디언트 계산
    _tmp = probs.copy()  # probs를 복사하여 사용
    _tmp[np.arange(num_train), y] -= 1
    dW = np.dot(X.T, _tmp) / num_train
    dW += reg * W  # 정규화 그라디언트 추가

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
