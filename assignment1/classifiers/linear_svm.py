from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += reg * 2 * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #1) 데이터의 클래스 개수정의하기
    num_classes = W.shape[1]
    #2)데이터의 인스턴스 개수 정의하기
    num_train = X.shape[0]
    #scores는 X*W -> X는 (N,D) W는 (D,C)
    scores = X.dot(W)
    #train을 순차로 나열하고, scores행렬에서 정답 클래스의 값을 추출
    correct_class_score = scores[np.arange(num_train), y]
    #로스(마진)값을 계산
    margins = np.maximum(0, scores - correct_class_score[:, np.newaxis] + 1)
    #정답클래스의 마진(로스)은 0
    margins[np.arange(num_train), y] = 0
    loss = np.sum(margins)
    loss /= num_train
    loss += reg * np.sum(W * W)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #margins의 shape을 카피해 놓고 -> 나중에 그라디언트 계산할 때 활용
    #손실이 있는..(잘못 예측한) 클래스에는 1을 할당하여 정답 예측한 부분과 분리
    X_mask = np.zeros(margins.shape)
    X_mask[margins > 0] = 1
    #잘못 예측했으면 열의 값이 1일 것이다. 이 1들을 모두 sum하여 개수를 산정하면 -> 잘못분류한 데이터의 개수이다.
    incorrect_counts = np.sum(X_mask, axis=1)
    # 정답클래스에 잘못 분류한 개수만큼 -를 줌으로서 zero-sum으로 만들어줌 -> 학습할 때..정답 클래스 값이 더 커지니깐..오류없이 학습하도록 유도가 가능함
    X_mask[np.arange(num_train), y] = -incorrect_counts
    #X를 transpose -> (D,N)으로 바꿔줌 X.mask (N,C)의 형태이므로 (D,C)=가중치와 같은 형태로 만들어줌 = dw의 역할
    dW = X.T.dot(X_mask)
    dW /= num_train
    dW += reg * 2 * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW