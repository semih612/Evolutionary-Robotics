import numpy as np

from evorob.world.robot.controllers.base import Controller


class NeuralNetworkController(Controller):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 16):
        self.n_input = input_size
        self.n_output = output_size
        self.n_hidden = hidden_size

        self.input_to_hidden = np.random.uniform(-1, 1, (hidden_size, input_size))
        self.hidden_to_output = np.random.uniform(-1, 1, (output_size, hidden_size))
        self.hidden_bias = np.zeros(hidden_size)
        self.output_bias = np.zeros(output_size)

        self.n_params_i2h = hidden_size * input_size
        self.n_params_h2o = output_size * hidden_size
        self.n_params_b1 = hidden_size
        self.n_params_b2 = output_size

        self.n_params = self.get_num_params()

    def get_action(self, state):    
        state = np.asarray(state)
        hidden = np.tanh(state @ self.input_to_hidden.T + self.hidden_bias)
        output = np.tanh(hidden @ self.hidden_to_output.T + self.output_bias)
        return np.clip(output, -1.0, 1.0)

    def set_weights(self, encoding):
    
        encoding = np.asarray(encoding)
        if encoding.size != self.n_params:
            raise ValueError(f"Expected encoding size {self.n_params}, got {encoding.size}")

        idx = 0
        i2h_end = idx + self.n_params_i2h
        self.input_to_hidden = encoding[idx:i2h_end].reshape((self.n_hidden, self.n_input))
        idx = i2h_end

        h2o_end = idx + self.n_params_h2o
        self.hidden_to_output = encoding[idx:h2o_end].reshape((self.n_output, self.n_hidden))
        idx = h2o_end

        b1_end = idx + self.n_params_b1
        self.hidden_bias = encoding[idx:b1_end]
        idx = b1_end

        b2_end = idx + self.n_params_b2
        self.output_bias = encoding[idx:b2_end]

    def geno2pheno(self, genotype):
        self.set_weights(genotype)

    def get_num_params(self):
        return self.n_params_i2h + self.n_params_h2o + self.n_params_b1 + self.n_params_b2

    def reset_controller(self, batch_size=1) -> None:
        pass
