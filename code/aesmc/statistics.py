from . import inference
from . import math
import torch




def empirical_expectation(value, log_weight, f):
    """Empirical expectation.

    input:
        value: Tensor/Variable
            [batch_size, num_particles, value_dim_1, ..., value_dim_N] (or
            [batch_size, num_particles])
        log_weight: Tensor/Variable [batch_size, num_particles]
        f: function which takes Tensor/Variable
            [batch_size, value_dim_1, ..., value_dim_N] (or
            [batch_size]) and returns a Tensor/Variable
            [batch_size, dim_1, ..., dim_M] (or [batch_size])
    output: empirical expectation Tensor/Variable
        [batch_size, dim_1, ..., dim_M] (or [batch_size])
    """

    assert(value.size()[:2] == log_weight.size())
    normalized_weights = torch.exp(math.lognormexp(log_weight, dim=1))

    # first particle
    f_temp = f(value[:, 0])
    w_temp = normalized_weights[:, 0]
    for i in range(f_temp.dim() - 1):
        w_temp = w_temp.unsqueeze(-1)

    emp_exp = w_temp.expand_as(f_temp) * f_temp

    # next particles
    for p in range(1, normalized_weights.size(1)):
        f_temp = f(value[:, p])
        w_temp = normalized_weights[:, p]
        for i in range(f_temp.dim() - 1):
            w_temp = w_temp.unsqueeze(-1)

        emp_exp += w_temp.expand_as(f_temp) * f_temp

    return emp_exp


def empirical_mean(value, log_weight):
    """Empirical mean.

    input:
        value: Tensor/Variable
            [batch_size, num_particles, dim_1, ..., dim_N] (or
            [batch_size, num_particles])
        log_weight: Tensor/Variable [batch_size, num_particles]
    output: empirical mean Tensor/Variable
        [batch_size, dim_1, ..., dim_N] (or [batch_size])
    """

    return empirical_expectation(value, log_weight, lambda x: x)


def empirical_variance(value, log_weight):
    """Empirical variance.

    input:
        value: Tensor/Variable
            [batch_size, num_particles, dim_1, ..., dim_N] (or
            [batch_size, num_particles])
        log_weight: Tensor/Variable [batch_size, num_particles]
    output: empirical mean Tensor/Variable
        [batch_size, dim_1, ..., dim_N] (or [batch_size])
    """

    emp_mean = empirical_mean(value, log_weight)
    emp_var = empirical_expectation(value-emp_mean, log_weight, lambda x: x**2)
    return emp_var




def reconstruct_observation_states(
    algorithm,
    observation_states,
    initial,
    transition,
    deterministic_transition,
    emission,
    proposal,
    num_particles
):
    """Reconstruct observations given a generative model and an inference
    algorithm.

    input:
        algorithm: 'is' or 'smc'
        initial: nn.Module with following forward signature
                def forward(self):
                    returns StateRandomVariable

        transition: nn.Module with following forward signature
                def forward(self, previous_latent_state):
                    returns StateRandomVariable

        deterministic_transition: None or nn.Module with following forward signature
                def forward(self, previous_latent_state, latent_state, observation_states, time):
                    returns State

        emission: nn.Module with following forward signature
                def forward(self, previous_latent_state, latent_state):
                    returns StateRandomVariable

        proposal: nn.Module with following forward signature
                def forward(self, previous_latent_state, observation_states, time):
                    returns StateRandomVariable
        num_particles: int; number of particles
    output:
        latent_states: list of aesmc.state.State objects
        reconstructed_observation_states: list of aesmc.state.State objects
        log_weight: Tensor/Variable [batch_size, num_particles]
    """

    latent_states, log_weight, _ = inference.infer(
        algorithm=algorithm,
        observation_states=observation_states,
        initial=initial,
        transition=transition,
        deterministic_transition=None,
        emission=emission,
        proposal=proposal,
        num_particles=num_particles,
        reparameterized=False,
        only_log_marginal_likelihood=False
    )

    inference_result = inference.infer(
        algorithm=algorithm,
        observation_states=observation_states,
        initial=initial,
        transition=transition,
        deterministic_transition=None,
        emission=emission,
        proposal=proposal,
        num_particles=num_particles,
        reparameterized=False,
        return_log_marginal_likelihood=False,
        return_latent_states=True,
        return_original_latent_states=False,
        return_log_weight=True,
        return_log_weights=False,
        return_ancestral_indices=False
    )

    return inference_result.latent_states, [
        emission.sample(latent_state)
        for latent_state in inference_result.latent_states
    ], inference_result.log_weight


def predict_observation_states(
    algorithm,
    observation_states,
    initial,
    transition,
    emission,
    proposal,
    num_particles,
    num_prediction_timesteps
):
    """Predict observations given a generative model and an inference
    algorithm.

    input:
        algorithm: 'is' or 'smc'
        observation_states: list of aesmc.state.State objects
        initial: nn.Module with following forward signature
                def forward(self):
                    returns StateRandomVariable

        transition: nn.Module with following forward signature
                def forward(self, previous_latent_state):
                    returns StateRandomVariable

        deterministic_transition: None or nn.Module with following forward signature
                def forward(self, previous_latent_state, latent_state, observation_states, time):
                    returns State

        emission: nn.Module with following forward signature
                def forward(self, previous_latent_state, latent_state):
                    returns StateRandomVariable

        proposal: nn.Module with following forward signature
                def forward(self, previous_latent_state, observation_states, time):
                    returns StateRandomVariable
        num_particles: int; number of particles
        num_prediction_timesteps: int; number of timesteps to predict
    output:
        predicted_latent_states: list of aesmc.state.State objects
        predicted_observation_states: list of aesmc.state.State objects
        log_weight: Tensor/Variable [batch_size, num_particles]
    """

    inference_result = inference.infer(
        algorithm=algorithm,
        observation_states=observation_states,
        initial=initial,
        transition=transition,
        emission=emission,
        proposal=proposal,
        num_particles=num_particles,
        reparameterized=False,
        return_log_marginal_likelihood=False,
        return_latent_states=True,
        return_original_latent_states=False,
        return_log_weight=True,
        return_log_weights=False,
        return_ancestral_indices=False
    )

    last_latent_state = inference_result.latent_states[-1]
    predicted_latent_states = []
    predicted_observation_states = []
    for time in range(num_prediction_timesteps):
        predicted_latent_states.append(
            transition.sample(last_latent_state)
        )
        predicted_observation_states.append(
            emission.sample(predicted_latent_states[-1])
        )
        last_latent_state = predicted_latent_states[-1]

    return predicted_latent_states, predicted_observation_states, \
        inference_result.log_weight


def reconstruct_and_predict_observation_states(
    algorithm,
    observation_states,
    initial,
    transition,
    emission,
    proposal,
    num_particles,
    num_prediction_timesteps
):
    """Reconstruct and predict observations given a generative model and an
    inference algorithm.

    More efficient than calling reconstruct_observation_states and
    predict_observation_states in turn.

    input:
        algorithm: 'is' or 'smc'
        observation_states: list of aesmc.state.State objects
        initial: nn.Module with following forward signature
                def forward(self):
                    returns StateRandomVariable

        transition: nn.Module with following forward signature
                def forward(self, previous_latent_state):
                    returns StateRandomVariable

        deterministic_transition: None or nn.Module with following forward signature
                def forward(self, previous_latent_state, latent_state, observation_states, time):
                    returns State

        emission: nn.Module with following forward signature
                def forward(self, previous_latent_state, latent_state):
                    returns StateRandomVariable

        proposal: nn.Module with following forward signature
                def forward(self, previous_latent_state, observation_states, time):
                    returns StateRandomVariable
        num_particles: int; number of particles
        num_prediction_timesteps: int; number of timesteps to predict
    output:
        latent_states: list of aesmc.state.State objects
        reconstructed_observation_states: list of aesmc.state.State objects
        predicted_latent_states: list of aesmc.state.State objects
        predicted_observation_states: list of aesmc.state.State objects
        log_weight: Tensor/Variable [batch_size, num_particles]
    """

    inference_result = inference.infer(
        algorithm=algorithm,
        observation_states=observation_states,
        initial=initial,
        transition=transition,
        emission=emission,
        proposal=proposal,
        num_particles=num_particles,
        reparameterized=False,
        return_log_marginal_likelihood=False,
        return_latent_states=True,
        return_original_latent_states=False,
        return_log_weight=True,
        return_log_weights=False,
        return_ancestral_indices=False
    )

    last_latent_state = inference_result.latent_states[-1]
    predicted_latent_states = []
    predicted_observation_states = []
    for time in range(num_prediction_timesteps):
        predicted_latent_states.append(
            transition.sample(last_latent_state)
        )
        predicted_observation_states.append(
            emission.sample(predicted_latent_states[-1])
        )
        last_latent_state = predicted_latent_states[-1]

    return inference_result.latent_states, \
        [
            emission.sample(latent_state) for latent_state in
            inference_result.latent_states
        ], \
        predicted_latent_states, \
        predicted_observation_states, \
        inference_result.log_weight
