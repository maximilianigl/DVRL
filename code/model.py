import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from policy import Categorical, DiagGaussian
from torch.nn.init import xavier_normal_, orthogonal_
import encoder_decoder
import namedlist
from operator import mul
from functools import reduce

# Container to return all required values from model
PolicyReturn = namedlist.namedlist('PolicyReturn', [
    ('latent_state', None),
    ('value_estimate', None),
    ('action', None),
    ('action_log_probs', None),
    ('dist_entropy', None),
    ('total_encoding_loss', None),
    ('encoding_losses', None),
    ('num_killed_particles', None),
    ('predicted_obs_img', None),
    ('particle_obs_img', None),
])


class Policy(nn.Module):
    """
    Parent class to both RNNPolicy and DVRLPolicy.
    """

    def __init__(self, action_space, encoding_dimension):
        super().__init__()

        # Value function V(latent_state)
        self.critic_linear = nn.Linear(encoding_dimension, 1)

        # Policy \pi(a|latent_state)
        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(encoding_dimension, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(encoding_dimension, num_outputs)
        else:
            raise NotImplementedError

        # BatchNorm layer (in case we use it, see algorithm.model in default.yaml for configuration)
        self.encoding_bn = nn.BatchNorm1d(encoding_dimension)

    def encode(self, observation, reward, actions, previous_latent_state, predicted_times):
        """
        To be provided by child class. Recurrently encodes the current observation and last action
        into a new latent state.
        Args:
            observation: current observation
            reward: current reward (not really needed atm, but we could learn to predict it)
            actions: last action on all n_s environments
            previous_latent_state: last latent state (one tensor for RNN or particle ensemble for DVRL)
            predicted_times: list of ints or None, indicating whether predictions should be returned.

        Returns:
            new_latent_state: Updated latent state
            total_encoding_loss: E.g. L^{ELBO} for DVRL
            encoding_losses: Split up into losses on latent and reconstruction
            avg_num_killed_particles: Average numer of killed particles in particle filter
            predicted_observations: Predicted observations (depending on timesteps specified in predicted_times)
            predicted_particles: List of Nones
        """
        raise NotImplementedError("Should be provided by child class, e.g. RNNPolicy or DVRLPolicy.")

    def new_latent_state(self):
        """
        To be provided by child class. Creates either n_s latent tensors for RNN or n_s particle
        ensembles for DVRL.
        """
        raise NotImplementedError("Should be provided by child class, e.g. RNNPolicy or DVRLPolicy.")

    def vec_conditional_new_latent_state(self, latent_state, mask):
        """
        To be provided by child class. Creates a new latent state for each environment in which the episode ended.
        """

    def forward(self, current_memory, deterministic=False, predicted_times=None):
        """
        Run the model forward and compute all the stuff we need (see PolicyReturn namedTuple).

        Args:
            current_memory: Contains
              - 'current_obs': Current observation
              - 'oneHotActions': Actions, either oneHot (if discrete) or just the action (if continuous)
              - 'states': previous latent state of model
            deterministic: Take the action with highest probability?
            predicted_times (list of ints): Ask the model to reconstruct/predict observations.
                                            (Only makes sense if observation_coef is not 0, i.e. if we learn a model)
                                            0: Current observation, 1: One step into the future, etc...

        Returns:
            policy_return (namedTuple): Contains:
              - 'latent_state': Updated latent state of model
              - 'value_estimate': V
              - 'action': a, sampled from policy
              - 'action_log_prob': Log probability of action under policy
              - 'dist_entropy': Entropy of policy
              - 'total_encoding_loss': L^{ELBO} for DVRL, 0 for RNN
              - 'encoding losses' (tuple): For DVRL: - transition_logpdf + proposal_logpdf, - emission_logpdf
              - 'num_killed_particles': For DVRL: Number of averaged killed particles
              - 'predicted_obs_img': Predicted images (when `predicted_times` is not an empty list)
              - 'particle_obs_img': Predicted images (per particle) 
        """

        policy_return = PolicyReturn()

        device = next(self.parameters()).device

        # Run the encode function. This
        latent_state, total_encoding_loss, encoding_losses, n_killed_p,\
            img, p_img = self.encode(
                observation=current_memory['current_obs'].to(device),
                reward=current_memory['rewards'].to(device),
                actions=current_memory['oneHotActions'].to(device).detach(),
                previous_latent_state=current_memory['states'].to(device),
                predicted_times=predicted_times,
            )

        # Detach latent state if we want to train encoder purely based on ELBO loss
        latent_state_for_encoding = latent_state.detach() if self.detach_encoder else latent_state

        # For DVRL encode all particles into one latent state
        encoded_state = (self.encode_particles(latent_state_for_encoding)
                         if type(self).__name__ == 'DVRLPolicy'
                         else latent_state_for_encoding)

        # Apply batch norm if so configured
        if self.policy_batch_norm:
            encoded_state = self.encoding_bn(encoded_state)

        # Fill up policy_return with return values
        policy_return.latent_state = latent_state
        policy_return.total_encoding_loss = total_encoding_loss
        policy_return.encoding_losses = encoding_losses
        policy_return.num_killed_particles = n_killed_p
        policy_return.predicted_obs_img = img
        policy_return.particle_obs_img = p_img

        policy_return.value_estimate = self.critic_linear(encoded_state)
        action = self.dist.sample(encoded_state, deterministic=deterministic)
        policy_return.action = action

        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(encoded_state, action.detach())
        policy_return.action_log_probs = action_log_probs
        policy_return.dist_entropy = dist_entropy

        return policy_return


class RNNPolicy(Policy):
    def __init__(self,
                 action_space,
                 nr_inputs,
                 observation_type,
                 action_encoding,
                 cnn_channels,
                 h_dim,
                 init_function,
                 encoder_batch_norm,
                 policy_batch_norm,
                 obs_loss_coef,
                 detach_encoder,
                 batch_size,
                 resample
                 ):

        super().__init__(action_space, encoding_dimension=h_dim)
        self.h_dim = h_dim
        self.init_function = init_function
        self.batch_size = batch_size
        self.obs_loss_coef = float(obs_loss_coef)
        self.encoder_batch_norm = encoder_batch_norm
        self.policy_batch_norm = policy_batch_norm
        self.observation_type = observation_type
        self.detach_encoder = detach_encoder
        self.resample = resample

        # All encoders and decoders are define centrally in one file
        self.encoder = encoder_decoder.get_encoder(
            observation_type,
            nr_inputs,
            cnn_channels,
            batch_norm=encoder_batch_norm
        )

        self.cnn_output_dimension = encoder_decoder.get_cnn_output_dimension(
            observation_type,
            cnn_channels
            )
        self.cnn_output_number = reduce(mul, self.cnn_output_dimension, 1)

        # Decoder takes latent_state + action_encoding
        # linear_obs_decoder is a fc network projecting the latent state onto the correct
        # dimensionality for a CNN decoder
        encoding_dimension = h_dim + action_encoding
        if encoder_batch_norm:
            self.linear_obs_decoder = nn.Sequential(
                nn.Linear(encoding_dimension, self.cnn_output_number),
                nn.BatchNorm1d(self.cnn_output_number),
                nn.ReLU()
            )
        else:
            self.linear_obs_decoder = nn.Sequential(
                nn.Linear(encoding_dimension, self.cnn_output_number),
                nn.ReLU()
            )

        self.decoder = encoder_decoder.get_decoder(
            observation_type,
            nr_inputs,
            cnn_channels,
            batch_norm=encoder_batch_norm)

        # Actions are encoded using one FC layer.
        if action_encoding > 0:
            if action_space.__class__.__name__ == "Discrete":
                action_shape = action_space.n
            else:
                action_shape = action_space.shape[0]
            if encoder_batch_norm:
                self.action_encoder = nn.Sequential(
                    nn.Linear(action_shape, action_encoding),
                    nn.BatchNorm1d(action_encoding),
                    nn.ReLU()
                )
            else:
                self.action_encoder = nn.Sequential(
                    nn.Linear(action_shape, action_encoding),
                    nn.ReLU())

        self.gru = nn.GRUCell(self.cnn_output_number + action_encoding, h_dim)
        # if self.encoder_batch_norm:
        #     self.gru_bn = nn.BatchNorm1d(h_dim)

        if observation_type == 'fc':
            self.obs_criterion = nn.MSELoss()
        else:
            self.obs_criterion = nn.BCEWithLogitsLoss()

        self.train()
        self.reset_parameters()

    def new_latent_state(self):
        """
        Return new latent state.
        self.batch_size is the number of parallel environments being used.
        """
        return torch.zeros(self.batch_size, self.h_dim)

    def vec_conditional_new_latent_state(self, latent_states, masks):
        """
        Set latent state to 0 when new episode beings.
        Masks and latent_states contain the values for each of the 16 environments.
        """
        return latent_states * masks

    def reset_parameters(self):
        def weights_init(gain):
            def fn(m):
                classname = m.__class__.__name__
                init_func = getattr(torch.nn.init, self.init_function)
                if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                    init_func(m.weight.data, gain=gain)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                if classname.find('GRUCell') != -1:
                    init_func(m.weight_ih.data)
                    init_func(m.weight_hh.data)
                    m.bias_ih.data.fill_(0)
                    m.bias_hh.data.fill_(0)

                #     #TODO: Debug/Remove:
                #     m.weight.data.fill_(0.001)
                # if classname.find('GRUCell') != -1:
                #     m.weight_ih.data.fill_(0.01)
                #     m.weight_hh.data.fill_(0.01)
                # print(m)

            return fn

        relu_gain = nn.init.calculate_gain('relu')
        self.apply(weights_init(relu_gain))
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def encode(self, observation, reward, actions, previous_latent_state, predicted_times):
        """
        Encode the observation and previous_latent_state into the new latent_state.
        Args:
            observation, reward: Last observation and reward recieved from all n_e environments
            actions: Action vector (oneHot for discrete actions)
            previous_latent_state: previous latent state, here a h_dim-dimensional vector
            predicted_times (list of ints): List of timesteps into the future for which predictions
                                            should be returned. Only makes sense if
                                            encoding_loss_coef != 0 and obs_loss_coef != 0

        Returns:
            latent_state: New latent state
            encoding_loss: Reconstruction loss when prediction current observation X obs_loss_coef
            predicted_obs_loss: Reconstruction loss when prediction current observation
            predicted_observations: Predicted observations (depending on timesteps specified in predicted_times)
            predicted_particles: List of Nones
        """
        x = self.encoder(observation)
        x = x.view(-1, self.cnn_output_number)

        encoded_actions = None
        if hasattr(self, 'action_encoder'):
            encoded_actions = self.action_encoder(actions)
            encoded_actions = F.relu(encoded_actions)
            x = torch.cat([x, encoded_actions], dim=1)

        # GRU
        if hasattr(self, 'gru'):
            latent_state = self.gru(x, previous_latent_state)

        # Compute observation losses
        device = previous_latent_state.device
        predicted_obs_loss = torch.zeros(1).to(device)

        # RNN: Predict observations/rewards based on previous(!) latent state
        predicted_observations = None
        if self.obs_loss_coef > 0:
            o_dec = self.linear_obs_decoder(
                torch.cat([
                    previous_latent_state,
                    encoded_actions
                ], dim=1)).view(-1, *self.cnn_output_dimension)
            obs_predicted = self.decoder(o_dec)
            predicted_obs_loss = self.obs_criterion(obs_predicted, observation)

            if predicted_times is not None:
                predicted_observations = self.predict_observations(
                    latent_state,
                    encoded_actions,
                    obs_predicted,
                    predicted_times)

        # This is only used for DVRL
        predicted_particles = (None if predicted_observations is None
                               else [None] * len(predicted_observations))

        encoding_loss = (self.obs_loss_coef * predicted_obs_loss)

        return latent_state, encoding_loss, \
            (torch.tensor(0), predicted_obs_loss), 0,\
            predicted_observations, predicted_particles

    def predict_observations(self, latent_state, encoded_actions,
                             obs_predicted, predicted_times):
        """
        Unroll the model into the future and predict observations.
        """
        max_distance = max(predicted_times)

        predicted_observations = []
        if 0 in predicted_times:
            predicted_observations.append(obs_predicted)

        for dt in range(max_distance):
            previous_latent_state = latent_state

            o_dec = self.linear_obs_decoder(
                torch.cat([
                    previous_latent_state,
                    encoded_actions
                ], dim=1)).view(-1, *self.cnn_output_dimension)
            # x = F.relu(x)
            obs_predicted = self.decoder(o_dec)

            x = self.encoder(obs_predicted)
            x = x.view(-1, self.cnn_output_number)
            x = torch.cat([x, encoded_actions], dim=1)

            latent_state = self.gru(x, previous_latent_state)

            if dt+1 in predicted_times:
                predicted_observations.append(obs_predicted)

        return predicted_observations







