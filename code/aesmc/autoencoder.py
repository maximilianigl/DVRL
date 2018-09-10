from . import inference
import torch
import torch.nn as nn
import aesmc.state as st


class AutoEncoder(nn.Module):
    def __init__(
        self,
            encoding_network,
            transition_network,
            deterministic_transition_network,
            emission_network,
            proposal_network
    ):

        super(AutoEncoder, self).__init__()

        # For separate gradient updates
        self.proposal_network = proposal_network
        self.transition_network = transition_network
        self.deterministic_transition_network = deterministic_transition_network
        self.emission_network = emission_network
        self.encoding_network = encoding_network

    def forward(self, observation_states, num_particles, resample,
                previous_latent_state, observation_coef, prior_coef):
        """Evaluate a computation graph whose gradient is an estimator for the
        gradient of the ELBO.

        input:
            observation_states: List of State objects. Usually contains:
                - all_x: All observations
                - (optional) all_a: All actions (for the RL setting)
            resample: bool whether to use AESMC or IWAE/VAE
            seq_lengts: Sequence lengths when padding is used
            previous_latent_state (optional): Will be used instead of
                drawing a sample from the initial_network

        output: InferenceResults namedList containing:
            - log_marginal_likelihood
            - vae_log_marginal_likelihood (only for resample=False)
            - latent_states: Resampled or not, depending on resample
                             (Currently not implemented with seq_lengths)
            - original_latent_states: Never resampled
            - log_weights
            - log_weight: (not implemented with seq_lengths)
                          This is the relevant weight for reweighting predictions
            - ancestral_indices
            - log_ancestral_indeices_proposal
            - num_killed_particles
        """

        if resample:
            algorithm = 'smc'
        else:
            algorithm = 'is'

        inference_result = inference.infer(
            algorithm=algorithm,
            observation_states=observation_states,
            encoding=self.encoding_network,
            transition=self.transition_network,
            deterministic_transition=self.deterministic_transition_network,
            emission=self.emission_network,
            proposal=self.proposal_network,
            num_particles=num_particles,
            reparameterized=True,
            previous_latent_state=previous_latent_state,
            observation_coef=observation_coef,
            prior_coef=prior_coef
        )

        return inference_result

    def reconstruct_predict(self,
                            observation_states,
                            num_particles,
                            resample,
                            reconstruction_length,
                            prediction_length,
                            summarize_function):
        """
        REWRITE!!
        input:
            observations: Variable [num_timesteps, batch_size, observation_dim]
            resample: bool. True: smc, False: is.
            num_particles: number. number of particles for posterior approximation.
            prediction_length: number. length of prediction.
            reconstruction_length: number. length of reconstruction_length
                                   Should be num_timesteps - prediction_length
            summarize_function: Function. Takes in an ensemble of states at a certain time
                                and corresponding weights. Outputs a single (combined) state.
        output:
            observations_reconstructed_predicted:
                Tensor [num_timesteps + prediction_length, batch_size, observation_dim]
        """

        # Check this
        num_timesteps, batch_size, nr_channels, w, h = observation_states.all_x.size()

        assert(num_timesteps == reconstruction_length + prediction_length)
        # I think we only predict one channel?

        # given_obs = observations[:reconstruction_length]
        # Only regress model on 'known' states
        given_obs = st.State(
            all_x=observation_states.all_x[:reconstruction_length]
        )

        inference_result = self.forward(
            observation_states=given_obs,
            num_particles=num_particles,
            resample=resample,
            return_inference_results=True
            )

        latent_states = inference_result.latent_states
        latent_state = latent_states[-1]

        for t in range(reconstruction_length, num_timesteps):
            previous_latent_state = latent_state

            # a) Prior: Draw latent_state.z
            transition_state_random_variable = self.transition_network(
                previous_latent_state
                )
            latent_state = transition_state_random_variable.sample(
                batch_size, num_particles
                )
            batch_size, num_particles, z_dim = latent_state.z.size()

            # TODO: This is highly specific for VRNN
            latent_state.phi_z = self.deterministic_transition_network.phi_z(
                latent_state.z.view(-1, z_dim)
                ).view(
                    batch_size, num_particles, -1
                    )

            latent_states.append(latent_state)

            # b) Generation: Draw x and compute phi_x
            emission_state_random_variable = self.emission_network(
                previous_latent_state, latent_state
                )
            x = emission_state_random_variable.sample(
                        batch_size, num_particles).all_x

            # This is usually done in the init-network
            # num_timesteps = 1 for one image
            # TODO: This is highly specific for VRNN
            phi_x = self.encoding_network.phi_x(
                x.view(-1, nr_channels, w, h)
                ).view(
                    1, batch_size, num_particles, -1
                    ).contiguous()

            current_observation = st.State(
                all_phi_x=phi_x,
                x=x  # Just in case, not used in VRNN
                )

            # Set time=0 because we have only the last observation
            if self.deterministic_transition_network is not None:
                latent_state = self.deterministic_transition_network(
                    previous_latent_state=previous_latent_state,
                    latent_state=latent_state,
                    observation_states=current_observation,
                    time=0
                )

        # Ok, at this point we have all the latent states
        # Compute reconstructed/predicted observations from given latents
        averaged_obs = torch.zeros(num_timesteps, batch_size, nr_channels, h, w)
        all_obs = torch.zeros(num_timesteps, batch_size, num_particles, nr_channels, h, w)
        observation_states.unsequeeze_and_expand_all_(dim=2, size=num_particles)
        initial_state_random_variable = self.initial_network(
            observation_states
        )

        initial_state = initial_state_random_variable.sample(
            batch_size, num_particles
            )

        for t in range(num_timesteps):
            if t == 0:
                previous_latent_state = initial_state
            else:
                previous_latent_state = latent_states[t-1]
            latent_state = latent_states[t]
            emission_state_random_variable = self.emission_network(
                previous_latent_state, latent_state
                )
            x = emission_state_random_variable.sample(
                        batch_size, num_particles).all_x
            all_obs[t] = x.data
            averaged_obs[t] = summarize_function(
                x,
                inference_result.log_weight).data

        return all_obs, averaged_obs, inference_result.log_weight.data
