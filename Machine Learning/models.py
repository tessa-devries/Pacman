import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        return nn.DotProduct(x, self.get_weights())

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        if nn.as_scalar(self.run(x)) >= 0:
            return 1
        else:
            return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        
        while True:
            trained = True
            for x,y in dataset.iterate_once(1):
                if (self.get_prediction(x) != nn.as_scalar(y)):
                    trained = False
                    nn.Parameter.update(self.get_weights(), x, nn.as_scalar(y))
            if trained:
                return
                
                
                    
                
                
            

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        self.w1 = nn.Parameter(1, 100)
        self.w2 = nn.Parameter(100, 1)
        self.b1 = nn.Parameter(1, 100)
        self.b2 = nn.Parameter(1, 1)
        self.batch_size = 1
        

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        print(x)
        x_w1 = nn.Linear(x, self.w1)
        relu_input = nn.AddBias(x_w1, self.b1)
        relu = nn.ReLU(relu_input)
        relu_w2 = nn.Linear(relu, self.w2)
        with_b2 = nn.AddBias(relu_w2, self.b2)
        return with_b2

    
    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        return nn.SquareLoss(self.run(x), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        
        while True:
            for x,y in dataset.iterate_once(1):
                loss = self.get_loss(x, y)
                
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(grad_wrt_w1, -0.006)
                self.b1.update(grad_wrt_b1, -0.006)
                self.w2.update(grad_wrt_w2, -0.006)
                self.b2.update(grad_wrt_b2, -0.006)
                
            if nn.as_scalar(loss) < 0.02:
                    return
                

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        self.w1 = nn.Parameter(784, 100)
        # x is a (10, 784) matrix, so x*w1 = (10, 100)
        self.w2 = nn.Parameter(100, 10)
        #(x*w1 + b1) * w2 is (10, 100) * (100, 10) = (10, 10)
        self.b1 = nn.Parameter(1, 100)
        #x*w1 + b1 = (10, 100)
        self.b2 = nn.Parameter(1, 10)
        # total is (10, 10) + (1, 10) = (10, 10) 
        self.batch_size = 1

      
    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        x_w1 = nn.Linear(x, self.w1)
        relu_input = nn.AddBias(x_w1, self.b1)
        relu = nn.ReLU(relu_input)
        relu_w2 = nn.Linear(relu, self.w2)
        with_b2 = nn.AddBias(relu_w2, self.b2)
        return with_b2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        return nn.SoftmaxLoss(self.run(x), y)
        
    def train(self, dataset):
        """
        Trains the model.
        """
        while True:
            for x,y in dataset.iterate_once(1):
                loss = self.get_loss(x, y)
                
                grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2])
                self.w1.update(grad_wrt_w1, -0.005)
                self.b1.update(grad_wrt_b1, -0.005)
                self.w2.update(grad_wrt_w2, -0.005)
                self.b2.update(grad_wrt_b2, -0.005)
                
            if dataset.get_validation_accuracy() >= 0.98:
                return
