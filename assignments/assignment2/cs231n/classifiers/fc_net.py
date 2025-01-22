from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        # 1. 입력 크기 설정
        layer_input_dim = input_dim  # 입력 계층의 크기

        # 2. 은닉 계층 및 가중치 초기화
        for i, hidden_dim in enumerate(hidden_dims):
            # 가중치 및 바이어스 초기화
            self.params[f"W{i+1}"] = np.random.normal(0, weight_scale, (layer_input_dim, hidden_dim))
            self.params[f"b{i+1}"] = np.zeros(hidden_dim)

            # 배치 정규화 매개변수 초기화
            if self.normalization:
                self.params[f"gamma{i+1}"] = np.ones(hidden_dim)
                self.params[f"beta{i+1}"] = np.zeros(hidden_dim)

            # 다음 계층의 입력 크기 업데이트
            layer_input_dim = hidden_dim

        # 3. 출력 계층 (softmax 계층) 초기화
        self.params[f"W{self.num_layers}"] = np.random.normal(0, weight_scale, (layer_input_dim, num_classes))
        self.params[f"b{self.num_layers}"] = np.zeros(num_classes)

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        caches = []  # To store intermediate caches for backpropagation
        dropout_caches = []  # To store dropout caches
        out = X  # Input to the first layer

        # Forward pass through hidden layers
        for i in range(1, self.num_layers):
            W, b = self.params[f"W{i}"], self.params[f"b{i}"]

            # Affine forward
            out, cache_affine = affine_forward(out, W, b)

            # Batch or layer normalization
            if self.normalization == "batchnorm":
                gamma, beta = self.params[f"gamma{i}"], self.params[f"beta{i}"]
                out, cache_bn = batchnorm_forward(out, gamma, beta, self.bn_params[i-1])
                cache_affine = (cache_affine, cache_bn)
            elif self.normalization == "layernorm":
                gamma, beta = self.params[f"gamma{i}"], self.params[f"beta{i}"]
                out, cache_ln = layernorm_forward(out, gamma, beta, self.bn_params[i-1])
                cache_affine = (cache_affine, cache_ln)

            # ReLU activation
            out, cache_relu = relu_forward(out)

            # Apply dropout if enabled
            if self.use_dropout:
                out, cache_dropout = dropout_forward(out, self.dropout_param)
                dropout_caches.append(cache_dropout)

            caches.append((cache_affine, cache_relu))

        # Forward pass through the final layer
        W_final, b_final = self.params[f"W{self.num_layers}"], self.params[f"b{self.num_layers}"]
        scores, cache_final = affine_forward(out, W_final, b_final)
        caches.append(cache_final)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        loss, grads = 0.0, {}

        # Compute softmax loss and gradients
        loss, dscores = softmax_loss(scores, y)

        # Add L2 regularization to the loss
        for i in range(1, self.num_layers + 1):
            W = self.params[f"W{i}"]
            loss += 0.5 * self.reg * np.sum(W ** 2)

        # Backward pass through the final layer
        dout, dW, db = affine_backward(dscores, caches[-1])
        grads[f"W{self.num_layers}"] = dW + self.reg * self.params[f"W{self.num_layers}"]
        grads[f"b{self.num_layers}"] = db

        # Backward pass through hidden layers
        for i in range(self.num_layers - 1, 0, -1):
            cache_affine, cache_relu = caches[i-1]

            # Dropout backward
            if self.use_dropout:
                dout = dropout_backward(dout, dropout_caches[i-1])

            # ReLU backward
            dout = relu_backward(dout, cache_relu)

            # Batch or layer normalization backward
            if self.normalization == "batchnorm":
                cache_affine, cache_bn = cache_affine
                dout, dgamma, dbeta = batchnorm_backward(dout, cache_bn)
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta
            elif self.normalization == "layernorm":
                cache_affine, cache_ln = cache_affine
                dout, dgamma, dbeta = layernorm_backward(dout, cache_ln)
                grads[f"gamma{i}"] = dgamma
                grads[f"beta{i}"] = dbeta

            # Affine backward
            dout, dW, db = affine_backward(dout, cache_affine)
            grads[f"W{i}"] = dW + self.reg * self.params[f"W{i}"]
            grads[f"b{i}"] = db
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads