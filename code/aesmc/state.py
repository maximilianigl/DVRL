from torch.autograd import Variable
import collections
import torch


def resample(value, ancestral_index):
    """Resample the value without side effects.

    input:
        value: Tensor/Variable [batch_size, num_particles, dim_1, ..., dim_N]
            (or [batch_size, num_particles])
        ancestral_index: LongTensor/Variable [batch_size, num_particles]
    output: resampled value [batch_size, num_particles, dim_1, ..., dim_N]
        (or [batch_size, num_particles])
    """

    assert(ancestral_index.size() == value.size()[:2])
    ancestral_index_unsqueezed = ancestral_index

    for _ in range(len(value.size()) - 2):
        ancestral_index_unsqueezed = \
            ancestral_index_unsqueezed.unsqueeze(-1)

    return torch.gather(
        value,
        dim=1,
        index=ancestral_index_unsqueezed.expand_as(value)
    )

class State():
    """Collection of Tensor/Variable objects.
    E.g.

        state = State(
            x=torch.Tensor([[2]]),
            y=torch.Tensor([[3]]),
        )
        state.y = torch.Tensor([[3]])
        y = state['y']
        if y in state:
            pass
    """
    def __init__(self, **kwargs):
        # Needed because we overwrote normal __setattr__ to only allow torch tensors/variables
        object.__setattr__(self, '_items', {})

        for name in kwargs:
            self._set_value_(name, kwargs[name])

    def __contains__(self, key):
        return key in self._items

    def __getattr__(self, name):
        return self._items[name]

    def __setattr__(self, name, value):
        self._set_value_(name, value)

    def __getitem__(self, key):
        # Access elements
        if isinstance(key, str):
            return getattr(self, key)

        # Trick to allow state to be returned by nn.Module
        # Purposely only supports `0` to catch potential misuses
        if key==0:
            if not self._items:
                raise KeyError("State is empty")
            for key, value in self._items.items():
                return value

        raise KeyError('State only supports slicing through the method slice_elements()')

    def __setitem__(self, name, value):
        self._set_value_(name, value)

    def __str__(self):
        return str(self._items)

    def _set_value_(self, name, value):
        if not (isinstance(value, Variable) or torch.is_tensor(value)):
            raise TypeError('value {} is not a Tensor/Variable'.format(value))

        assert(len(value.size()) >= 2)

        for old_name, old_value in self._items.items():
            # [Batch, particles]
            assert(value.size()[:2] == old_value.size()[:2])
            break
        self._items[name] = value
        return self

    def index_elements(self, key):
        """Returns a new State which contains `key` applied to each element"""
        new_state = State()
        # If number, slice or tuple, apply it to elements
        for name, value in self._items.items():
            if isinstance(value, Variable):
                # Variable doesn't support index() but behaves correctly for []
                setattr(new_state, name, value[key])
            else:
                # Need to use index() instead of [] to keep dimension
                setattr(new_state, name, value.index(key))
        return new_state

    def unsequeeze_and_expand_all_(self, dim, size):
        "Unsqueezes all elements at the given dim and expands that dim to the given size"
        def fn(tensor):
            dims = list(tensor.size())
            dims.insert(dim, size)
            return tensor.unsqueeze(dim).expand(*dims).contiguous()

        return self.apply_each_(fn)

    def multiply_each(self, mask, only):
        new_state = State()
        for name, value in self._items.items():
            if name not in only:
                continue
            xfactor = mask
            dims_factor = len(mask.size())
            dims_value = len(value.size())
            assert(dims_factor <= dims_value)
            for i in range(dims_value - dims_factor):
                xfactor = xfactor.unsqueeze(-1)
            assert (len(xfactor.size()) == len(value.size()))
            setattr(new_state, name, value * xfactor)
        return new_state

    def apply_each_(self, fn):
        """Applies function fn to each of the values in this state and returns
        this state."""
        for name, value in self._items.items():
            self._items[name] = fn(value)
        return self

    def apply_each(self, fn):
        new_state = State()
        for name, value in self._items.items():
            setattr(new_state, name, fn(value))
            # new_state[name] = fn(value)
        return new_state

    def clone(self):
        """Returns a copy of this state."""
        state = State()
        for key, value in self._items.items():
            setattr(state,key,value.clone())
        return state

    def cpu_(self):
        return self.apply_each_(lambda x: x.cpu())

    def cuda_(self):
        return self.apply_each_(lambda x: x.cuda())

    def cuda(self):
        return self.apply_each(lambda x: x.cuda())

    def to(self, device):
        return self.apply_each(lambda x: x.to(device))

    def detach_(self):
        return self.apply_each_(lambda x: x.detach())

    def requires_grad_(self):
        return self.apply_each_(lambda x: x.requires_grad_())

    def detach(self):
        return self.apply_each(lambda x: x.detach())

    def resample(self, ancestral_index):
        """Resample this state (returns a new copy).

        Assumes that all elements have [batch_size, num_particles, ...]
        input:
            ancestral_index: LongTensor/Variable [batch_size, num_particles]
            dim: Particle dimension along which the resampling is performed
        output: new resampled state
        """
        new_state = self.clone()
        return new_state.resample_(ancestral_index)

    def resample_(self, ancestral_index):
        """Resample this state inplace.

        Assumes that all elements have [batch_size, num_particles, ...]
        input:
            ancestral_index: LongTensor/Variable [batch_size, num_particles]
            dim: Particle dimension along which the resampling is performed
        output: updated state
        """
        return self.apply_each_(lambda value: resample(value, ancestral_index))

    def to_tensor_(self):
        return self.apply_each_(lambda value: value.data)

    def to_variable_(self, **kwargs):
        return self.apply_each_(lambda value: Variable(value, **kwargs))

    def update(self, second_state):
        if second_state is not None:
            self._items.update(second_state._items)
        return self
