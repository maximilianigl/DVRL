import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import aesmc.random_variable as rv

# import aesmc.autoencoder as ae
import aesmc.state as st
import aesmc.util as ae_util
import aesmc.statistics as stats
import aesmc.math as math
import aesmc.test_utils as tu
from aesmc.inference import sample_ancestral_index
import encoder_decoder
import numpy as np
import model
from operator import mul
from functools import reduce


class PF_State:
    def __init__(self, particle_state, particle_log_weights):
        self.particle_state = particle_state
        self.particle_log_weights = particle_log_weights

    def detach(self):
        return PF_State(
            self.particle_state.detach(), self.particle_log_weights.detach()
        )

    def cuda(self):
        return PF_State(self.particle_state.cuda(), self.particle_log_weights.cuda())


class DVRLPolicy(model.Policy):
    def __init__(
        self,
        action_space,
        nr_inputs,
        observation_type,
        action_encoding,
        # obs_encoding,
        cnn_channels,
        h_dim,
        init_function,
        encoder_batch_norm,
        policy_batch_norm,
        prior_loss_coef,
        obs_loss_coef,
        detach_encoder,
        batch_size,
        num_particles,
        particle_aggregation,
        z_dim,
        resample,
    ):
        super().__init__(action_space, encoding_dimension=h_dim)
        self.init_function = init_function
        self.num_particles = num_particles
        self.particle_aggregation = particle_aggregation
        self.batch_size = batch_size
        self.obs_loss_coef = float(obs_loss_coef)
        self.prior_loss_coef = float(prior_loss_coef)
        self.observation_type = observation_type
        self.encoder_batch_norm = encoder_batch_norm
        self.policy_batch_norm = policy_batch_norm
        self.detach_encoder = detach_encoder
        self.h_dim = h_dim
        self.z_dim = z_dim
        self.resample = resample

        # All encoder/decoders are defined in the encoder_decoder.py file
        self.cnn_output_dimension = encoder_decoder.get_cnn_output_dimension(
            observation_type, cnn_channels
        )
        self.cnn_output_number = reduce(mul, self.cnn_output_dimension, 1)

        # Naming conventions
        phi_x_dim = self.cnn_output_number

        if action_space.__class__.__name__ == "Discrete":
            action_shape = action_space.n
        else:
            action_shape = action_space.shape[0]

        ## Create all relevant networks

        # Encodes actions and observations into a latent state
        self.encoding_network = VRNN_encoding(
            phi_x_dim=phi_x_dim,
            nr_actions=action_shape,
            action_encoding=action_encoding,
            observation_type=observation_type,
            nr_inputs=nr_inputs,
            cnn_channels=cnn_channels,
            encoder_batch_norm=encoder_batch_norm,
        )

        # Computes p(z_t|h_{t-1}, a_{t-1})
        self.transition_network = VRNN_transition(
            h_dim=h_dim, z_dim=z_dim, action_encoding=action_encoding
        )

        # Computes h_t=f(h_{t-1}, z_t, a_{t-1}, o_t)
        self.deterministic_transition_network = VRNN_deterministic_transition(
            z_dim=z_dim,
            phi_x_dim=phi_x_dim,
            h_dim=h_dim,
            action_encoding=action_encoding,
        )

        # Computes p(o_t|h_t, z_t, a_{t-1})
        self.emission_network = VRNN_emission(
            h_dim=h_dim,
            action_encoding=action_encoding,
            observation_type=observation_type,
            nr_inputs=nr_inputs,
            cnn_channels=cnn_channels,
            encoder_batch_norm=encoder_batch_norm,
        )

        # Computes q(z_t|h_{t-1}, a_{t-1}, o_t)
        self.proposal_network = VRNN_proposal(
            z_dim=z_dim,
            h_dim=h_dim,
            phi_x_dim=phi_x_dim,
            action_encoding=action_encoding,
            encoder_batch_norm=encoder_batch_norm,
        )

        # dim is for z, h, w, where z & h both have h_dim and w is scalar
        dim = 2 * h_dim + 1
        if particle_aggregation == "rnn" and self.num_particles > 1:
            self.particle_gru = nn.GRU(dim, h_dim, batch_first=True)

        elif self.num_particles == 1:
            self.particle_gru = nn.Linear(dim, h_dim)

        self.reset_parameters()

    def new_latent_state(self):
        """
        Return new latent state.
        This is a function because the latent state is different for DVRL and RNN.
        """
        device = next(self.parameters()).device
        initial_state = st.State(
            h=torch.zeros(self.batch_size, self.num_particles, self.h_dim).to(device)
        )

        log_weight = torch.zeros(self.batch_size, self.num_particles).to(device)

        initial_state.log_weight = log_weight

        return initial_state

    def vec_conditional_new_latent_state(self, latent_state, mask):
        """
        Set latent state to 0-tensors when new episode begins.
        Args:
            latent_state (`State`): latent_state
            mask: binary tensor with 0 whenever a new episode begins.

        """
        # Multiply log_weight, h, z with mask
        return latent_state.multiply_each(mask, only=["log_weight", "h", "z"])

    def reset_parameters(self):
        def weights_init(gain):
            def fn(m):
                classname = m.__class__.__name__
                init_func = getattr(torch.nn.init, self.init_function)
                if classname.find("Conv") != -1 or classname.find("Linear") != -1:
                    init_func(m.weight.data, gain=gain)
                    if m.bias is not None:
                        m.bias.data.fill_(0)
                if classname.find("GRUCell") != -1:
                    init_func(m.weight_ih.data)
                    init_func(m.weight_hh.data)
                    m.bias_ih.data.fill_(0)
                    m.bias_hh.data.fill_(0)

            return fn

        relu_gain = nn.init.calculate_gain("relu")
        self.apply(weights_init(relu_gain))
        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def sample_from(self, state_random_variable):
        """
        Helper function, legazy code.
        """
        return state_random_variable.sample_reparameterized(
            self.batch_size, self.num_particles
        )

    def encode(
        self, observation, reward, actions, previous_latent_state, predicted_times
    ):
        """
        This is where the core of the DVRL algorithm is happening.

        Args:
            observation, reward: Last observation and reward recieved from all n_e environments
            actions: Action vector (oneHot for discrete actions)
            previous_latent_state: previous latent state of type state.State
            predicted_times (list of ints): List of timesteps into the future for which predictions
                                            should be returned. Only makes sense if
                                            encoding_loss_coef != 0 and obs_loss_coef != 0

        return latent_state, \
            - encoding_logli, \
            (- transition_logpdf + proposal_logpdf, - emission_logpdf),\
            avg_num_killed_particles,\
            predicted_observations, particle_observations
        Returns:
            latent_state: New latent state
            - encoding_logli = encoding_loss: Reconstruction loss when prediction current observation X obs_loss_coef
            - transition_logpdf + proposal_logpdf: KL divergence loss
            - emission_logpdf: Reconstruction loss
            avg_num_killed_particles: Average numer of killed particles in particle filter
            predicted_observations: Predicted observations (depending on timesteps specified in predicted_times)
            predicted_particles: List of Nones

        """
        batch_size, *rest = observation.size()

        # Total observation dim to normalise the likelihood
        # obs_dim = reduce(mul, rest, 1)

        # Needed for legacy AESMC code
        ae_util.init(observation.is_cuda)

        # Legacy code: We need to pass in a (time) sequence of observations
        # With dim=0 for time
        img_observation = observation.unsqueeze(0)
        actions = actions.unsqueeze(0)
        reward = reward.unsqueeze(0)

        # Legacy code: All values are wrapped in state.State (which can contain more than one value)
        observation_states = st.State(
            all_x=img_observation.contiguous(),
            all_a=actions.contiguous(),
            r=reward.contiguous(),
        )

        old_log_weight = previous_latent_state.log_weight

        # Encoding the actions and observations (nothing recurrent yet)
        observation_states = self.encoding_network(observation_states)

        # Expand the particle dimension
        observation_states.unsequeeze_and_expand_all_(dim=2, size=self.num_particles)

        ancestral_indices = sample_ancestral_index(old_log_weight)

        # How many particles were killed?
        # List over batch size
        num_killed_particles = list(
            tu.num_killed_particles(ancestral_indices.data.cpu())
        )
        if self.resample:
            previous_latent_state = previous_latent_state.resample(ancestral_indices)
        else:
            num_killed_particles = [0] * batch_size

        avg_num_killed_particles = sum(num_killed_particles) / len(num_killed_particles)

        # Legacy code: Select first (and only) time index
        current_observation = observation_states.index_elements(0)

        # Sample stochastic latent state z from proposal
        proposal_state_random_variable = self.proposal_network(
            previous_latent_state=previous_latent_state,
            observation_states=current_observation,
            time=0,
        )
        latent_state = self.sample_from(proposal_state_random_variable)

        # Compute deterministic state h and add to the latent state
        latent_state = self.deterministic_transition_network(
            previous_latent_state=previous_latent_state,
            latent_state=latent_state,
            observation_states=current_observation,
            time=0,
        )

        # Compute prior probability over z
        transition_state_random_variable = self.transition_network(
            previous_latent_state, current_observation
        )

        # Compute probability over observation
        emission_state_random_variable = self.emission_network(
            previous_latent_state,
            latent_state,
            current_observation
            # observation_states
        )

        emission_logpdf = emission_state_random_variable.logpdf(
            current_observation, batch_size, self.num_particles
        )

        proposal_logpdf = proposal_state_random_variable.logpdf(
            latent_state, batch_size, self.num_particles
        )
        transition_logpdf = transition_state_random_variable.logpdf(
            latent_state, batch_size, self.num_particles
        )

        assert self.prior_loss_coef == 1
        assert self.obs_loss_coef == 1
        new_log_weight = transition_logpdf - proposal_logpdf + emission_logpdf
        assert torch.sum(transition_logpdf != transition_logpdf) == 0
        assert torch.sum(proposal_logpdf != proposal_logpdf) == 0
        assert torch.sum(emission_logpdf != emission_logpdf) == 0
        # new_log_weight = (self.prior_loss_coef * (transition_logpdf - proposal_logpdf)
        #                   + self.obs_loss_coef * emission_logpdf)

        latent_state.log_weight = new_log_weight

        # Average (in log space) over particles
        encoding_logli = math.logsumexp(
            # torch.stack(log_weights, dim=0), dim=2
            new_log_weight,
            dim=1,
        ) - np.log(self.num_particles)

        # inference_result.latent_states = latent_states

        predicted_observations = None
        particle_observations = None
        if predicted_times is not None:
            predicted_observations, particle_observations = self.predict_observations(
                latent_state=latent_state,
                current_observation=current_observation,
                actions=actions,
                emission_state_random_variable=emission_state_random_variable,
                predicted_times=predicted_times,
            )

        ae_util.init(False)

        return (
            latent_state,
            -encoding_logli,
            (-transition_logpdf + proposal_logpdf, -emission_logpdf),
            avg_num_killed_particles,
            predicted_observations,
            particle_observations,
        )

    def predict_observations(
        self,
        latent_state,
        current_observation,
        actions,
        emission_state_random_variable,
        predicted_times,
    ):
        """
        Assumes that the current encoded action (saved in 'current_observation') is
        repeated into the future
        """

        max_distance = max(predicted_times)
        old_log_weight = latent_state.log_weight
        predicted_observations = []
        particle_observations = []

        if 0 in predicted_times:
            x = emission_state_random_variable.all_x._probability

            averaged_obs = stats.empirical_mean(x, old_log_weight)
            predicted_observations.append(averaged_obs)
            particle_observations.append(x)

        batch_size, num_particles, z_dim = latent_state.z.size()
        batch_size, num_particles, h_dim = latent_state.h.size()
        for dt in range(max_distance):
            old_observation = current_observation
            previous_latent_state = latent_state

            # Get next state
            transition_state_random_variable = self.transition_network(
                previous_latent_state, old_observation
            )
            latent_state = self.sample_from(transition_state_random_variable)

            # Hack. This is usually done in det_transition
            latent_state.phi_z = self.deterministic_transition_network.phi_z(
                latent_state.z.view(-1, z_dim)
            ).view(batch_size, num_particles, h_dim)

            # Draw observation
            emission_state_random_variable = self.emission_network(
                previous_latent_state,
                latent_state,
                old_observation
                # observation_states
            )
            x = emission_state_random_variable.all_x._probability
            averaged_obs = stats.empirical_mean(x, old_log_weight)

            # Encode observation
            # Unsqueeze time dimension
            current_observation = st.State(
                all_x=averaged_obs.unsqueeze(0), all_a=actions.contiguous()
            )
            current_observation = self.encoding_network(current_observation)
            current_observation.unsequeeze_and_expand_all_(
                dim=2, size=self.num_particles
            )
            current_observation = current_observation.index_elements(0)

            # Deterministic update
            latent_state = self.deterministic_transition_network(
                previous_latent_state=previous_latent_state,
                latent_state=latent_state,
                observation_states=current_observation,
                time=0,
            )

            if dt + 1 in predicted_times:
                predicted_observations.append(averaged_obs)
                particle_observations.append(x)

        return predicted_observations, particle_observations

    def encode_particles(self, latent_state):
        """
        RNN that encodes the set of particles into one latent vector that can be passed to policy.
        """
        batch_size, num_particles, h_dim = latent_state.h.size()
        state = torch.cat([latent_state.h, latent_state.phi_z], dim=2)

        # latent_state.h [batch, particles, h_dim?]
        normalized_log_weights = math.lognormexp(
            # inference_result.log_weights[-1],
            latent_state.log_weight,
            dim=1,
        )

        particle_state = torch.cat(
            [state, torch.exp(normalized_log_weights).unsqueeze(-1)], dim=2
        )

        if self.num_particles == 1:
            # Get rid of particle dimension, particle_gru is just a nn.Linear
            particle_state = particle_state.squeeze(1)
            encoded_particles = self.particle_gru(particle_state)
            # encoded_particles = self.particle_gru_bn(encoded_particles)
            return encoded_particles
        else:
            _, encoded_particles = self.particle_gru(particle_state)
            # encoded_particles [num_layers * num_directions, batch, h_dim]
            # First dimension: num_layers * num_directions
            # Dimension of Output?
            return encoded_particles[0]


class VRNN_encoding(nn.Module):
    def __init__(
        self,
        phi_x_dim,
        nr_actions,
        action_encoding,
        observation_type,
        nr_inputs,
        cnn_channels,
        encoder_batch_norm,
    ):
        super().__init__()
        self.action_encoding = action_encoding
        self.phi_x_dim = phi_x_dim
        assert action_encoding > 0

        self.phi_x = encoder_decoder.get_encoder(
            observation_type, nr_inputs, cnn_channels, batch_norm=encoder_batch_norm
        )

        self.cnn_output_dimension = encoder_decoder.get_cnn_output_dimension(
            observation_type, cnn_channels
        )
        self.cnn_output_number = reduce(mul, self.cnn_output_dimension, 1)

        if encoder_batch_norm:
            self.action_encoder = nn.Sequential(
                nn.Linear(nr_actions, action_encoding),
                nn.BatchNorm1d(action_encoding),
                nn.ReLU(),
            )
        else:
            self.action_encoder = nn.Sequential(
                nn.Linear(nr_actions, action_encoding), nn.ReLU()
            )
        self.nr_actions = nr_actions

    def forward(self, observation_states):
        #
        """Compute the encoding for all x

        Input:
        - Observations_states containing `all_x`    [seq_len, batch_size, channels, width, height]

        Output:
        - Initial state containing `h`
        - Observations_states with additional entry `all_phi_x`
          [seq_len, batch_size, num_particles, encoding_dim]
        """
        seq_len, batch_size, *obs_dim = observation_states.all_x.size()

        # Encode the observations and expand
        all_phi_x = self.phi_x(
            observation_states.all_x.view(-1, *obs_dim)  # Collapse particles
        ).view(
            -1, self.cnn_output_number
        )  # Flatten CNN output
        # all_phi_x = self.linear_obs_encoder(all_phi_x).view(seq_len, batch_size, -1)
        # all_phi_x = F.relu(all_phi_x)
        all_phi_x = all_phi_x.view(seq_len, batch_size, -1)
        observation_states.all_phi_x = all_phi_x

        if self.action_encoding > 0:
            encoded_action = self.action_encoder(
                observation_states.all_a.view(-1, self.nr_actions)
            ).view(seq_len, batch_size, -1)
            observation_states.encoded_action = encoded_action

        return observation_states


class VRNN_transition(nn.Module):
    def __init__(self, h_dim, z_dim, action_encoding):
        super().__init__()
        self.prior = nn.Sequential(nn.Linear(h_dim + action_encoding, h_dim), nn.ReLU())
        self.prior_mean = nn.Linear(h_dim, z_dim)
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())
        self.action_encoding = action_encoding

    def forward(self, previous_latent_state, observation_states):
        """Outputs the prior probability of z_t.

        Inputs:
            - previous_latent_state containing at least
                `h`     [batch, particles, h_dim]
        """

        batch_size, num_particles, h_dim = previous_latent_state.h.size()

        if self.action_encoding > 0:
            input = torch.cat(
                [previous_latent_state.h, observation_states.encoded_action], 2
            ).view(-1, h_dim + self.action_encoding)
        else:
            input = previous_latent_state.h.view(-1, h_dim)

        prior_t = self.prior(input)

        prior_mean_t = self.prior_mean(prior_t).view(batch_size, num_particles, -1)
        prior_std_t = self.prior_std(prior_t).view(batch_size, num_particles, -1)

        prior_dist = rv.StateRandomVariable(
            z=rv.MultivariateIndependentNormal(mean=prior_mean_t, variance=prior_std_t)
        )

        return prior_dist


class VRNN_deterministic_transition(nn.Module):
    def __init__(self, z_dim, phi_x_dim, h_dim, action_encoding):
        super().__init__()
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())
        # From phi_z and phi_x_dim
        self.rnn = nn.GRUCell(h_dim + phi_x_dim + action_encoding, h_dim)
        self.action_encoding = action_encoding

    def forward(self, previous_latent_state, latent_state, observation_states, time):
        batch_size, num_particles, z_dim = latent_state.z.size()
        batch_size, num_particles, phi_x_dim = observation_states.all_phi_x.size()
        batch_size, num_particles, h_dim = previous_latent_state.h.size()

        phi_x = observation_states.all_phi_x

        phi_z_t = self.phi_z(latent_state.z.view(-1, z_dim)).view(
            batch_size, num_particles, h_dim
        )

        if self.action_encoding > 0:
            input = torch.cat(
                [phi_x, phi_z_t, observation_states.encoded_action], 2
            ).view(-1, phi_x_dim + h_dim + self.action_encoding)
        else:
            input = torch.cat([phi_x, phi_z_t], 1).view(-1, phi_x_dim + h_dim)

        h = self.rnn(input, previous_latent_state.h.view(-1, h_dim))

        latent_state.phi_z = phi_z_t.view(batch_size, num_particles, -1)
        # We need [batch, particles, ...] for aesmc resampling!
        latent_state.h = h.view(batch_size, num_particles, h_dim)
        return latent_state


class VRNN_emission(nn.Module):
    def __init__(
        self,
        h_dim,
        action_encoding,
        observation_type,
        nr_inputs,
        cnn_channels,
        encoder_batch_norm,
    ):
        super().__init__()
        self.observation_type = observation_type
        self.action_encoding = action_encoding
        # From h and phi_z

        encoding_dimension = h_dim + h_dim + action_encoding

        # For observation
        self.dec, self.dec_mean, self.dec_std = encoder_decoder.get_decoder(
            observation_type, nr_inputs, cnn_channels, batch_norm=encoder_batch_norm
        )

        self.cnn_output_dimension = encoder_decoder.get_cnn_output_dimension(
            observation_type, cnn_channels
        )
        self.cnn_output_number = reduce(mul, self.cnn_output_dimension, 1)

        if encoder_batch_norm:
            self.linear_obs_decoder = nn.Sequential(
                nn.Linear(encoding_dimension, self.cnn_output_number),
                nn.BatchNorm1d(self.cnn_output_number),
                nn.ReLU(),
            )
        else:
            self.linear_obs_decoder = nn.Sequential(
                nn.Linear(encoding_dimension, self.cnn_output_number), nn.ReLU()
            )

    def forward(self, previous_latent_state, latent_state, observation_states):
        """
        Returns: emission_dist [batch-size, num_particles, channels, w, h]

        """
        batch_size, num_particles, phi_z_dim = latent_state.phi_z.size()
        batch_size, num_particles, h_dim = previous_latent_state.h.size()

        # Unsqueeze: Add spatial dimensions
        dec_t = self.linear_obs_decoder(
            torch.cat(
                [
                    latent_state.phi_z,
                    previous_latent_state.h,
                    observation_states.encoded_action,
                ],
                2,
            ).view(-1, phi_z_dim + h_dim + self.action_encoding)
        )

        dec_t = self.dec(dec_t.view(-1, *self.cnn_output_dimension))

        # dec_mean_t = self.location(dec_t).view(batch_size, num_particles,
        #                                        *obs_dim)
        dec_mean_t = self.dec_mean(dec_t)
        _, *obs_dim = dec_mean_t.size()
        dec_mean_t = dec_mean_t.view(batch_size, num_particles, *obs_dim)

        if self.observation_type == "fc":
            dec_std_t = self.dec_std(dec_t).view(batch_size, num_particles, *obs_dim)
            emission_dist = rv.StateRandomVariable(
                all_x=rv.MultivariateIndependentNormal(
                    mean=dec_mean_t, variance=dec_std_t
                )
            )
        else:
            emission_dist = rv.StateRandomVariable(
                all_x=rv.MultivariateIndependentPseudobernoulli(probability=dec_mean_t)
            )

        return emission_dist


class VRNN_proposal(nn.Module):
    def __init__(self, z_dim, h_dim, phi_x_dim, action_encoding, encoder_batch_norm):
        super().__init__()

        if encoder_batch_norm:
            self.enc = nn.Sequential(
                nn.Linear(h_dim + phi_x_dim + action_encoding, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            )
        else:
            self.enc = nn.Sequential(
                nn.Linear(h_dim + phi_x_dim + action_encoding, h_dim), nn.ReLU()
            )
        self.enc_mean = nn.Linear(h_dim, z_dim)
        self.enc_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())

        self.action_encoding = action_encoding

    def forward(self, previous_latent_state, observation_states, time):
        batch_size, num_particles, phi_x_dim = observation_states.all_phi_x.size()
        batch_size, num_particles, h_dim = previous_latent_state.h.size()

        if self.action_encoding > 0:
            input = torch.cat(
                [
                    observation_states.all_phi_x,
                    previous_latent_state.h,
                    observation_states.encoded_action,
                ],
                2,
            ).view(-1, phi_x_dim + h_dim + self.action_encoding)
        else:
            input = torch.cat(
                [observation_states.all_phi_x, previous_latent_state.h], 2
            ).view(-1, phi_x_dim + h_dim)

        enc_t = self.enc(input)

        enc_mean_t = self.enc_mean(enc_t).view(batch_size, num_particles, -1)
        enc_std_t = self.enc_std(enc_t).view(batch_size, num_particles, -1)

        proposed_state = rv.StateRandomVariable(
            z=rv.MultivariateIndependentNormal(mean=enc_mean_t, variance=enc_std_t)
        )
        return proposed_state
