import collections
import numpy as np
import torch
from . import state as st
from torch.autograd import Variable
import logging
import warnings


class RandomVariable():
    """Base class for random variables. Supported methods:
        - sample(batch_size, num_particles)
        - sample_reparameterized(batch_size, num_particles)
        - logpdf(value, batch_size, num_particles)
    """

    def sample(self, batch_size, num_particles):
        """Returns a sample of this random variable."""

        raise NotImplementedError

    def sample_reparameterized(self, batch_size, num_particles):
        """Returns a reparameterized sample of this random variable."""

        raise NotImplementedError

    def pdf(self, value, batch_size, num_particles):
        """Evaluate the density of this random variable at a value. Returns
        Tensor/Variable [batch_size, num_particles].
        """

        raise NotImplementedError

    def logpdf(self, value, batch_size, num_particles):
        """Evaluate the log density of this random variable at a value. Returns
        Tensor/Variable [batch_size, num_particles].
        """

        raise NotImplementedError

    def kl_divergence(self, other_random_variable):
        """
        Compute the analytic KL-divergence between this and given random variable,
        i.e. KL(self||other_random_variable)
        """

        raise NotImplementedError


class StateRandomVariable(RandomVariable):
    """Collection of RandomVariable objects. Implements sample,
    sample_reparameterized, logpdf methods.

    E.g.

        state_random_variable = StateRandomVariable(random_variables={
            'a': Normal(
                mean=torch.zeros(3, 2),
                variance=torch.ones(3, 2)
            )
        })
        state_random_variable.b = MultivariateIndependentNormal(
            mean=torch.zeros(3, 2, 4, 5),
            variance=torch.ones(3, 2, 4, 5)
        )
        state = state_random_variable.sample(
            batch_size=3,
            num_particles=2
        )
        state_logpdf = state_random_variable.logpdf(
            value=state,
            batch_size=3,
            num_particles=2
        )
    """
    def __init__(self, **kwargs):
        # Needed because we overwrote normal __setattr__ to only allow torch tensors/variables
        object.__setattr__(self, '_items', {})

        for name in kwargs:
            self.set_random_variable_(name, kwargs[name])


    def __setitem__(self, name, value):
        self.__setattr__(name,value)

    def __getitem__(self, key):
        # Access elements
        if isinstance(key, str):
            return getattr(self, key)

        # Trick to allow state to be returned by nn.Module
        # Purposely only supports `0` to catch potential misuses
        if key==0:
            if not self._values:
                raise KeyError("StateRandomVariable is empty")
            for key, value in self._values.items():
                return value

        raise KeyError('StateRandomVariable only supports slicing through the method slice_elements()')

    def __getattr__(self, name):
        if '_items' in self.__dict__:
            _items = self.__dict__['_items']
            if name in _items:
                return _items[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        if isinstance(value, RandomVariable):
            self.set_random_variable_(name, value)
        elif (
            ('_items' in self.__dict__) and
            (name in self__dict__['_items'])
        ):
            raise AttributeError(
                'cannot override assigned random variable {0} with a value '
                'that is not a RandomVariable: {1}'.format(name, value)
            )
        else:
            object.__setattr__(self, name, value)

    def random_variables(self):
        """Return a lazy iterator over random_variables"""
        for name, random_variable in self._items.items():
            yield random_variable

    def named_random_variables(self):
        """Return a lazy iterator over random_variables"""
        for name, random_variable in self._items.items():
            yield name, random_variable

    def sample(self, batch_size, num_particles):
        state = st.State()
        for name, random_variable in self.named_random_variables():
            state[name] = random_variable.sample(
                batch_size=batch_size,
                num_particles=num_particles
            )

        return state

    def sample_reparameterized(self, batch_size, num_particles):
        state = st.State()
        for name, random_variable in self.named_random_variables():
            setattr(state, name, random_variable.sample_reparameterized(
                batch_size=batch_size,
                num_particles=num_particles
            ))
        return state

    def set_random_variable_(self, name, random_variable):
        if not isinstance(random_variable, RandomVariable):
            raise TypeError(
                'random_variable {} is not a RandomVariable'.
                format(random_variable)
            )
        _items = self.__dict__['_items']
        _items[name] = random_variable

        return self


    def _find_common_keys(self, other):
        random_variable_keys = [key for key in self._items]
        other_keys = [key for key in other._items]
        common_keys = list(set(random_variable_keys) & set(other_keys))

        # logging.debug("Computing logpdf for states: {}".format(common_keys))

        if not set(common_keys) == set(random_variable_keys):
            logging.warning("Not all random variable key are used, only {} out of {}!".format(
               common_keys, random_variable_keys
            ))

        if not set(common_keys) == set(other_keys):
            logging.debug("Not all other keys are used/evaluated, only {} out of {}!".format(
               common_keys, other_keys 
            ))

        return common_keys

    def logpdf(self, value, batch_size, num_particles):
        # assert(
        #     set([key for key, v in self.named_random_variables()]) ==
        #     set([key for key in value._values])
        # )

        common_keys = self._find_common_keys(value)

        result = 0
        # for name, random_variable in self.named_random_variables():
        for name in common_keys:
            result += self._items[name].logpdf(
            # result += random_variable.logpdf(
                value=value[name],
                batch_size=batch_size,
                num_particles=num_particles
            )

        return result

    def kl_divergence(self, other_state_random_variable):

        common_keys = self._find_common_keys(other_state_random_variable)

        result = 0
        # for name, random_variable in self.named_random_variables():
        for name in common_keys:
            result += self._items[name].kl_divergence(
            # result += random_variable.logpdf(
                other_random_variable=other_state_random_variable[name],
            )

        return result


class MultivariateIndependentLaplace(RandomVariable):
    """MultivariateIndependentLaplace random variable"""
    def __init__(self, location, scale):
        """Initialize this distribution with location, scale.

        input:
            location: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
            scale: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
        """
        assert(location.size() == scale.size())
        assert(len(location.size()) > 2)
        self._location = location
        self._scale = scale

    def sample(self, batch_size, num_particles):
        assert(list(self._location.size()[:2]) == [batch_size, num_particles])
        uniforms = torch.Tensor(self._location.size()).uniform_() - 0.5
        if isinstance(self._location, Variable):
            uniforms = Variable(uniforms)
            return self._location.detach() - self._scale.detach() * \
                torch.sign(uniforms) * torch.log(1 - 2 * torch.abs(uniforms))
        else:
            return self._location - self._scale * torch.sign(uniforms) * \
                torch.log(1 - 2 * torch.abs(uniforms))

    def sample_reparameterized(self, batch_size, num_particles):
        assert(list(self._location.size()[:2]) == [batch_size, num_particles])

        standard_laplace = MultivariateIndependentLaplace(
            location=Variable(torch.zeros(self._location.size())),
            scale=Variable(torch.ones(self._scale.size()))
        )

        return self._location + self._scale * standard_laplace.sample(
            batch_size, num_particles
        )

    def pdf(self, value, batch_size, num_particles):
        assert(value.size() == self._location.size())
        assert(list(self._location.size()[:2]) == [batch_size, num_particles])

        return torch.prod(
            (
                torch.exp(-torch.abs(value - self._location) / self._scale) /
                (2 * self._scale)
            ).view(batch_size, num_particles, -1),
            dim=2
        )

    def logpdf(self, value, batch_size, num_particles):
        assert(value.size() == self._location.size())
        assert(list(self._location.size()[:2]) == [batch_size, num_particles])

        return torch.sum(
            (
                -torch.abs(value - self._location) /
                self._scale - torch.log(2 * self._scale)
            ).view(batch_size, num_particles, -1),
            dim=2
        )


class MultivariateIndependentNormal(RandomVariable):
    """MultivariateIndependentNormal random variable"""
    def __init__(self, mean, variance):
        """Initialize this distribution with mean, variance.

        input:
            mean: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
            variance: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
        """
        assert(mean.size() == variance.size())
        assert(len(mean.size()) > 2)
        self._mean = mean
        self._variance = variance

    def sample(self, batch_size, num_particles):
        assert(list(self._mean.size()[:2]) == [batch_size, num_particles])

        uniform_normals = torch.Tensor(self._mean.size()).normal_()
        return self._mean.detach() + \
            Variable(uniform_normals) * torch.sqrt(self._variance.detach())

    def sample_reparameterized(self, batch_size, num_particles):
        assert(list(self._mean.size()[:2]) == [batch_size, num_particles])

        standard_normal = MultivariateIndependentNormal(
            mean=Variable(torch.zeros(self._mean.size())),
            variance=Variable(torch.ones(self._variance.size()))
        )

        return self._mean + torch.sqrt(self._variance) * \
            standard_normal.sample(batch_size, num_particles)

    def pdf(self, value, batch_size, num_particles):
        assert(value.size() == self._mean.size())
        assert(list(self._mean.size()[:2]) == [batch_size, num_particles])

        return torch.prod(
            (
                1 / torch.sqrt(2 * self._variance * np.pi) * torch.exp(
                    -0.5 * (value - self._mean)**2 / self._variance
                )
            ).view(batch_size, num_particles, -1),
            dim=2
        )

    def logpdf(self, value, batch_size, num_particles):
        assert(value.size() == self._mean.size())
        assert(list(self._mean.size()[:2]) == [batch_size, num_particles])

        return torch.sum(
            (
                -0.5 * (value - self._mean)**2 / self._variance -
                0.5 * torch.log(2 * self._variance * np.pi)
            ).view(batch_size, num_particles, -1),
            dim=2
        )

    def kl_divergence(self, other_random_variable):
        """ Compute analytic KL divergence between two gaussians.

        Input: another MultivariateIndependent random variable
        Ouptus: KL_divergence [batch, particles]
        """
        assert(isinstance(other_random_variable, MultivariateIndependentNormal))
        batch_size, num_particles, *_ = self._mean.size()
        mean_1 = self._mean
        mean_2 = other_random_variable._mean
        var_1 = self._variance
        var_2 = other_random_variable._variance
        kld_element = (
            torch.log(torch.sqrt(var_2)) - torch.log(torch.sqrt(var_1)) +
            (var_1 + (mean_1 - mean_2).pow(2)) / (2 * var_2) - 0.5
            )

        return torch.sum(kld_element.view(batch_size, num_particles, -1), dim=2)


class MultivariateIndependentUniform(RandomVariable):
    def __init__(self, low, high):
        """
        low and high have [*dims]
        """
        super().__init__()
        """Initialize this distribution"""

        self.low = low
        self.scale = high-low
        self.dims = low.size()

    def sample(self, batch_size, num_particles):
        return self.sample_reparameterized(batch_size, num_particles).detach()

    def sample_reparameterized(self, batch_size, num_particles):
        uniforms = torch.Tensor(*self.dims).uniform_()
        if isinstance(self.low, Variable):
            uniforms= Variable(uniforms)
        uniforms = (uniforms * self.scale) + self.low
        return uniforms

    def pdf(self, value, batch_size, num_particles):
        result = [1/self.scale]
        mask = torch.zeros(*self.dims)
        mask[(value>low) & (value<low+scale)] = 1
        if isinstance(result, Variable):
            mask = Variable(mask)
        result = result * mask
        return result

    def logpdf(self, value, batch_size, num_particles):
        return torch.log(self.pdf(value, batch_size, num_particles))


class MultivariateIndependentPseudobernoulli(RandomVariable):
    """MultivariateIndependentPseudobernoulli random variable"""
    def __init__(self, probability):
        """Initialize this distribution with probability.

        input:
            probability: Tensor/Variable
                [batch_size, num_particles, dim_1, ..., dim_N]
        """
        assert(len(probability.size()) > 2)
        self._probability = probability

    def sample(self, batch_size, num_particles):
        assert(
            list(self._probability.size()[:2]) == [batch_size, num_particles]
        )
        if isinstance(self._probability, Variable):
            return self._probability.detach()
        else:
            return self._probability

    def sample_reparameterized(self, batch_size, num_particles):
        assert(
            list(self._probability.size()[:2]) == [batch_size, num_particles]
        )

        return self._probability

    def pdf(self, value, batch_size, num_particles):
        assert(value.size() == self._probability.size())
        assert(
            list(self._probability.size()[:2]) == [batch_size, num_particles]
        )

        return torch.prod(
            (
                self._probability**value * (1 - self._probability)**(1 - value)
            ).view(batch_size, num_particles, -1),
            dim=2
        )

    def logpdf(self, value, batch_size, num_particles, epsilon=1e-10):
        assert(value.size() == self._probability.size())
        assert(
            list(self._probability.size()[:2]) == [batch_size, num_particles]
        )

        return torch.sum(
            (
                value * torch.log(self._probability + epsilon) +
                (1 - value) * torch.log(1 - self._probability + epsilon)
            ).view(batch_size, num_particles, -1),
            dim=2
        )


class Laplace(RandomVariable):
    """Laplace random variable"""
    def __init__(self, location, scale):
        """Initialize this distribution with location, scale.

        input:
            location: Tensor/Variable [batch_size, num_particles]
            scale: Tensor/Variable [batch_size, num_particles]
        """
        assert(len(location.size()) == 2)
        self._multivariate_independent_laplace = MultivariateIndependentLaplace(
            location=location.unsqueeze(-1),
            scale=scale.unsqueeze(-1)
        )

    def sample(self, batch_size, num_particles):
        return self._multivariate_independent_laplace.sample(
            batch_size, num_particles
        )

    def sample_reparameterized(self, batch_size, num_particles):
        return self._multivariate_independent_laplace.sample_reparameterized(
            batch_size, num_particles
        )

    def pdf(self, value, batch_size, num_particles):
        return self._multivariate_independent_laplace.pdf(
            value.unsqueeze(-1), batch_size, num_particles
        )

    def logpdf(self, value, batch_size, num_particles):
        return self._multivariate_independent_laplace.logpdf(
            value.unsqueeze(-1), batch_size, num_particles
        )

class Choice(RandomVariable):
    def __init__(self, choices):
        """Initialize this distribution with the available choices.

        input:
            choices: Tensor/Variable [batch_size, number_particles, number_choices]
        """
        self.return_variable = isinstance(choices, Variable)
        self.is_cuda = choices.is_cuda

        if self.return_variable:
            choices = choices.data
        if self.is_cuda:
            choices = choices.cpu()
        self.choices = choices

    def sample(self, batch_size, num_particles):
        uniform = torch.FloatTensor(batch_size, num_particles).uniform_()*self.choices.size(2)
        indices = torch.floor(uniform).unsqueeze(2)
        sampled_choices = torch.gather(self.choices, dim=2, index=indices.long()).squeeze(-1)
        if self.is_cuda:
            sampled_choices = sampled_choices.cuda()
        if self.return_variable:
            sampled_choices = Variable(sampled_choices)
        return sampled_choices

    def sample_reparameterized(self, batch_size, num_particles):
        warnings.warn("Trying to sample_reparameterized from Choice(). Not supported! Sad.")
        return self.sample(batch_size, num_particles)

    # TODO: Mask
    def pdf(self, value, batch_size, num_particles):
        result = torch.Tensor([1/self.choices.size(-1)])
        result = result.unsqueeze(0).expand(batch_size, num_particles)
        mask = torch.zeros(batch_size, num_particles)

        # for c in self.choices:
        #     condition = (value == c)
        #     mask[condition] = 1
        # result = result * mask
        if self.is_cuda:
            result = result.cuda()
        if self.return_variable:
            result = Variable(result)
        return result

    def logpdf(self, value, batch_size, num_particles):
        return torch.log(self.pdf(value, batch_size, num_particles))


class Normal(RandomVariable):
    """Normal random variable"""
    def __init__(self, mean, variance):
        """Initialize this distribution with mean, variance.

        input:
            mean: Tensor/Variable [batch_size, num_particles]
            variance: Tensor/Variable [batch_size, num_particles]
        """
        assert(len(mean.size()) == 2)
        self._multivariate_independent_normal = MultivariateIndependentNormal(
            mean=mean.unsqueeze(-1),
            variance=variance.unsqueeze(-1)
        )

    def sample(self, batch_size, num_particles):
        return self._multivariate_independent_normal.sample(
            batch_size, num_particles
        ).squeeze(-1)

    def sample_reparameterized(self, batch_size, num_particles):
        return self._multivariate_independent_normal.sample_reparameterized(
            batch_size, num_particles
        ).squeeze(-1)

    def pdf(self, value, batch_size, num_particles):
        return self._multivariate_independent_normal.pdf(
            value.unsqueeze(-1), batch_size, num_particles
        )

    def logpdf(self, value, batch_size, num_particles):
        return self._multivariate_independent_normal.logpdf(
            value.unsqueeze(-1), batch_size, num_particles
        )

    def kl_divergence(self, other_random_variable):
        assert(isinstance(other_random_variable, Normal))
        return self._multivariate_independent_normal.kl_divergence(
            other_random_variable._multivariate_independent_normal
        )


class Pseudobernoulli(RandomVariable):
    """Pseudobernoulli random variable"""
    def __init__(self, probability):
        """Initialize this distribution with probability.

        input:
            probability: Tensor/Variable [batch_size, num_particles]
        """
        assert(len(probability.size()) == 2)
        self._multivariate_independent_pseudobernoulli = \
            MultivariateIndependentPseudobernoulli(
                probability=probability.unsqueeze(-1)
            )

    def sample(self, batch_size, num_particles):
        return self._multivariate_independent_pseudobernoulli.sample(
            batch_size, num_particles
        ).squeeze(-1)

    def sample_reparameterized(self, batch_size, num_particles):
        return self._multivariate_independent_pseudobernoulli.\
            sample_reparameterized(batch_size, num_particles).squeeze(-1)

    def pdf(self, value, batch_size, num_particles):
        return self._multivariate_independent_pseudobernoulli.pdf(
            value.unsqueeze(-1), batch_size, num_particles
        )

    def logpdf(self, value, batch_size, num_particles, epsilon=1e-10):
        return self._multivariate_independent_pseudobernoulli.logpdf(
            value.unsqueeze(-1), batch_size, num_particles, epsilon=epsilon
        )


class Dirac(RandomVariable):
    """Pseudobernoulli random variable"""
    def __init__(self, value):
        """Initialize this distribution with it's value.

        input:
            value: Tensor/Variable [batch_size, num_particles]
        """
        assert(len(value.size()) == 2)
        self._value = value

    def sample(self, batch_size, num_particles):
        if isinstance(self._value, Variable):
            return (Variable(torch.ones(batch_size, num_particles))*self._value).detach()
        else:
            return torch.ones(batch_size, num_particles)*self._value

    def sample_reparameterized(self, batch_size, num_particles):
        assert (isinstance(self._value, Variable))
        return Variable(torch.ones(batch_size, num_particles))*self._value

    def pdf(self, value, batch_size, num_particles):

        if isinstance(self._value, Variable):
            if self._value.data.equal(value.data):
                return Variable(torch.ones(batch_size, num_particles))
            else:
                return Variable(torch.zeros(batch_size, num_particles))
        else:
            if self._value.equal(value):
                return torch.ones(batch_size, num_particles)
            else:
                return torch.zeros(batch_size, num_particles)

    def logpdf(self, value, batch_size, num_particles, epsilon=1e-10):
        return torch.log(self.pdf(value, batch_size, num_particles))
