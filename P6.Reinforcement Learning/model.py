import nn

class DeepQNetwork():
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = -0.67
        self.w1 = nn.Parameter(self.state_size, 100)
        # x is a (10, self.state_size) matrix, so x*w1 = (10, 100)
        #add another layer in between 784 and 100
        self.w2 = nn.Parameter(100, self.num_actions)
        #(x*w1 + b1) * w2 is (10, 100) * (100, 10) = (self.num_actions, self.num_actions)
        self.b1 = nn.Parameter(1, 100)
        #x*w1 + b1 = (10, 100)
        self.b2 = nn.Parameter(1, self.num_actions)
        #x*w1 + b1 + w2 is (self.num_actions, self.num_actions)
        self.w3 = nn.Parameter(self.num_actions, self.num_actions)
        self.b3 = nn.Parameter(1, self.num_actions)
        self.parameters = [self.w1, self.w2, self.b1, self.b2, self.w3, self.b3]
        self.batch_size = 300
        self.numTrainingGames = 3000

    def set_weights(self, layers):
        self.parameters = []
        for i in range(len(layers)):
            self.parameters.append(layers[i])

    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted
        by the network, and Q_target.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(states), Q_target)


    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a (batch_size x state_dim) numpy array
            (IGNORE) Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            result: (batch_size x num_actions) numpy array of Q-value
                scores, for each of the actions
        """
        "*** YOUR CODE HERE ***"

        x_w1 = nn.Linear(states, self.w1)
        relu_input = nn.AddBias(x_w1, self.b1)
        relu = nn.ReLU(relu_input)
        relu_w2 = nn.Linear(relu, self.w2)
        with_b2 = nn.AddBias(relu_w2, self.b2)
        relu_again = nn.ReLU(with_b2)
        relu_w3 = nn.Linear(relu_again, self.w3)
        with_b3 = nn.AddBias(relu_w3, self.b3)
        return with_b3


    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a (batch_size x state_dim) numpy array
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        learning_rate = self.learning_rate
        loss = self.get_loss(states, Q_target)
        grad_wrt_w1, grad_wrt_b1, grad_wrt_w2, grad_wrt_b2, grad_wrt_w3, grad_wrt_b3 = nn.gradients(loss, [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3])
        #learning rate was -0.008
        self.w1.update(grad_wrt_w1, learning_rate)
        self.b1.update(grad_wrt_b1, learning_rate)
        self.w2.update(grad_wrt_w2, learning_rate)
        self.b2.update(grad_wrt_b2, learning_rate)
        self.w3.update(grad_wrt_w3, learning_rate)
        self.b3.update(grad_wrt_b3, learning_rate)
