import numpy as np
import torch
import torchbnn as bnn
import torch.nn as nn
import torch.optim as optim
from torch.distributions.kl import kl_divergence

################################## Basic DNN ###################################################

class BasicMLSModel(nn.Module):
    """
    Class for original Machine Learning Scan (MLS) model (based on arXiv:1708.06615)
    """

    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        """
        Args:
            input_size: Size of MLP input layer (int)
            hidden_size: Fixed size of MLP hidden layer (int)
            num_hidden_layers: Number of MLP hidden layers (int)
            output_size: Size of MLP output layer (int)
        """
        super(BasicMLSModel, self).__init__()
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

################################## Basic DNN with dropout (MCDropout) ###################################################

class MCDropoutModel(nn.Module):
    """
    Class for Machine Learning Scan (MLS) model with dropout applied to all layers except the output layer
    """
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, dropout_prob):
        """
        Args:
            input_size: Size of MLP input layer (int)
            hidden_size: Fixed size of MLP hidden layer (int)
            num_hidden_layers: Number of MLP hidden layers (int)
            output_size: Size of MLP output layer (int)
            dropout_prob: Probability of excluding any node in the forward pass
        """
        super(MCDropoutModel, self).__init__()
        
        # Input layer with dropout
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Hidden layers with dropout
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            )
            for _ in range(num_hidden_layers)
        ])
        
        # Output layer
        self.output_layer = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x

################################## Bayesian neural network ###################################################

class BNNModel(nn.Module):
    """
    Class for Bayesian Neural Network (BNN) implementation of the Machine Learning Scan (MLS) model 
    """
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size, prior_mu = 0, prior_sigma = 0.1):
        """
        Args:
            input_size: Size of MLP input layer (int)
            hidden_size: Fixed size of MLP hidden layer (int)
            num_hidden_layers: Number of MLP hidden layers (int)
            output_size: Size of MLP output layer (int)
            prior_mu: Initial choice (prior) of the mean of the weights of the BNN 
            prior_sigma: Initial choice (prior) of the standard deviation of the weights of the BNN 
        """
        super(BNNModel, self).__init__()
        # Input layer
        self.input_layer = nn.Sequential(
                            bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=input_size, out_features=hidden_size),
                            nn.ReLU())
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(
                bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=hidden_size, out_features=hidden_size),
                nn.ReLU())
            for _ in range(num_hidden_layers)
        ])
        
        # Output layer
        self.output_layer = bnn.BayesLinear(prior_mu=prior_mu, prior_sigma=prior_sigma, in_features=hidden_size, out_features=output_size)
        
    def forward(self, x):
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.output_layer(x)
        return x



################################## Neural process (sub)-networks ###################################################

class DeterministicEncoder(nn.Module):
    """
    Class for Deterministic Encoder of Neural Process
    """
    def __init__(self, output_sizes):
        """NP deterministic encoder."""
        
        super(DeterministicEncoder, self).__init__()
        self.output_sizes = output_sizes

        # Define the encoding MLP
        layers = []
        for i in range(len(output_sizes) - 2):
            layers.extend([nn.Linear(output_sizes[i], output_sizes[i+1]), nn.ReLU()])
        layers.extend([nn.Linear(output_sizes[-2], output_sizes[-1])])
        self.encoding_mlp = nn.Sequential(*layers)

    def forward(self, context_x, context_y):
        """Encodes the inputs into one representation."""
        # Concatenate x and y along the last axis
        encoder_input = torch.cat([context_x, context_y], dim=-1)

        # Pass through the encoding MLP
        hidden = self.encoding_mlp(encoder_input)
        
        # Aggregator: take the mean over all points
        hidden = torch.mean(hidden, dim=-2)

        return hidden

class LatentEncoder(nn.Module):
    """
    Class for Latent Encoder of Neural Process
    """
    def __init__(self, output_sizes, num_latents):
        """NP latent encoder."""
        super(LatentEncoder, self).__init__()
        self.output_sizes = output_sizes
        self.num_latents = num_latents

        # Define the encoding MLP
        layers = []
        for i in range(len(output_sizes) - 2):
            layers.extend([nn.Linear(output_sizes[i], output_sizes[i+1]), nn.ReLU()])
        layers.extend([nn.Linear(output_sizes[-2], output_sizes[-1])])
        self.encoding_mlp = nn.Sequential(*layers)
        
        # Intermediate layer for mean and log_sigma
        self.penultimate_layer = nn.Linear(output_sizes[-1], (output_sizes[-1] + num_latents) // 2)

        # Mean and log_sigma layers
        self.mean_layer = nn.Linear((output_sizes[-1] + num_latents) // 2, num_latents)
        self.std_layer = nn.Linear((output_sizes[-1] + num_latents) // 2, num_latents)

    def forward(self, x, y):
        """Encodes the inputs into one representation."""
        # Concatenate x and y along the last axis
        encoder_input = torch.cat([x, y], dim=-1)

        # Pass through the encoding MLP
        hidden = self.encoding_mlp(encoder_input)

        # Aggregator: take the mean over all points
        hidden = torch.mean(hidden, dim=-2)
        
        # Apply intermediate ReLU layer
        hidden = F.relu(self.penultimate_layer(hidden))

        # Output mean and log_sigma
        mu = self.mean_layer(hidden)
        log_sigma = self.std_layer(hidden)

        # Compute sigma
        sigma = 0.1 + 0.9 * torch.sigmoid(log_sigma)

        # Create a normal distribution
        normal_dist = torch.distributions.Normal(loc=mu, scale=sigma)

        return normal_dist

class Decoder(nn.Module):
    """
    Class for Decoder of Neural Process
    """
    def __init__(self, output_sizes):
        """NP decoder."""
        super(Decoder, self).__init__()
        self.output_sizes = output_sizes

        # Define the decoder MLP
        layers = []
        for i in range(len(output_sizes) - 2):
            layers.extend([nn.Linear(output_sizes[i], output_sizes[i+1]), nn.ReLU()])
        layers.extend([nn.Linear(output_sizes[-2], output_sizes[-1])])
        self.decoder_mlp = nn.Sequential(*layers)

    def forward(self, representation, target_x):
        """Decodes the individual targets."""
        # Concatenate target_x and representation
        hidden = torch.cat([representation, target_x], dim=-1)
        
        # Pass through the decoder MLP
        hidden = self.decoder_mlp(hidden)

        # Split the hidden representation into mean (mu) and log standard deviation (log_sigma)
        mu, log_sigma = torch.split(hidden, hidden.size(-1) // 2, dim=-1)

        # Bound the standard deviation (sigma)
        sigma = 0.1 + 0.9 * F.softplus(log_sigma)

        # Create a multivariate Gaussian distribution
        dist = torch.distributions.Normal(loc=mu, scale=sigma)

        return dist, mu, sigma

class NPModel(nn.Module):
    """The NP model."""
    # This is the full NP model that uses the encoder and decoder networks defined above
    def __init__(self, latent_encoder_output_sizes, num_latents,
                 decoder_output_sizes, use_deterministic_path=True,
                 deterministic_encoder_output_sizes=None):
        super(NPModel, self).__init__()
        self.lat_encoder = LatentEncoder(latent_encoder_output_sizes, num_latents)
        self.decoder = Decoder(decoder_output_sizes)
        self._use_deterministic_path = use_deterministic_path
        if use_deterministic_path:
            self.det_encoder = DeterministicEncoder(
                deterministic_encoder_output_sizes)


    def forward(self, context_x, context_y, target_x, num_targets, target_y=None):
        # Pass query through the encoder and the decoder
        prior = self.lat_encoder(context_x, context_y)

        # For training, when target_y is available, use targets for latent encoder.
        # Note that targets contain contexts by design.
        if target_y is None:
            lat_rep = prior.sample()
        # For testing, when target_y unavailable, use contexts for latent encoder.
        else:
            posterior = self.lat_encoder(target_x, target_y)
            lat_rep = posterior.sample()
        lat_rep = torch.tile(lat_rep,[num_targets,1])

        if self._use_deterministic_path:
            det_rep = self.det_encoder(context_x, context_y)
            det_rep = torch.tile(det_rep,[num_targets,1])
            full_rep = torch.cat([det_rep, lat_rep], axis=-1)
        else:
            full_rep = latent_rep

        dist, mu, sigma = self.decoder(full_rep, target_x)

        # If we want to calculate the log_prob for training we will make use of the
        # target_y. At test time the target_y is not available so we return None.
        if target_y is not None:
            log_p = dist.log_prob(target_y)
            posterior = self.lat_encoder(target_x, target_y)
            kl = kl_divergence(posterior, prior)
            kl = kl.mean(axis=-1)
            kl = torch.tile(kl, [1, num_targets])
            loss = -log_p.mean() + kl.mean()
        else:
            log_p = None
            kl = None
            loss = None

        return mu, sigma, log_p, kl, loss

############################### Include new networks below ##########################