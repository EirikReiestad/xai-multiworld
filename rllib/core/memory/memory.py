import random
from collections import deque
from typing import Tuple, TypeVar, Generic, List, Type

T = TypeVar("T")


class Memory(Generic[T], deque):
    def __init__(self, capacity: int, item_type: Type[T]):
        """
        Initialize the replay buffer with a fixed capacity and item type.

        :param capacity: Maximum number of items to store in the buffer.
        :param item_type: The type of items to store, typically a namedtuple or a dataclass.
        """
        super().__init__(maxlen=capacity)
        self._item_type = item_type

    def add(self, **kwargs):
        """
        Add a single item to the buffer.

        :param kwargs: Key-value pairs corresponding to the fields of the item type.
                       These will be passed to the constructor of the specified item type.
        """
        self._validate_fields(set(kwargs.keys()))
        self.append(self._item_type(**kwargs))

    def add_dict(self, keys, **kwargs):
        """
        Add multiple items to the buffer using dictionaries of field values.

        :param keys: An iterable of keys used to index the provided dictionaries for each field.
        :param kwargs: Dictionaries for each field required by the item type.
                       Each dictionary must have the same keys as the provided `keys` parameter.
        """
        self._validate_fields(set(kwargs.keys()))
        for key in keys:
            self.add(**{field: kwargs[field][key] for field in kwargs})

    def sample(self, batch_size: int) -> Tuple[List[T], List[int]]:
        """
        Sample a batch of items from the buffer.

        :param batch_size: Number of items to sample from the buffer.
        :return: A list of sampled items of type T.
        :raises ValueError: If the buffer contains fewer items than the requested batch size.
        """
        indices = random.sample(range(len(self)), batch_size)  # Sample indices
        sampled_items = [self[i] for i in indices]  # Get the items based on indices
        return sampled_items, indices

    def reset(self):
        """
        Clear all items from the buffer.

        This resets the buffer to an empty state.
        """
        self.clear()

    def _validate_fields(self, provided_fields):
        """
        Validate that the provided fields match the expected fields of the item type.

        :param provided_fields: The set of field names provided.
        :raises ValueError: If the provided fields do not match the item type fields.
        """
        expected_fields = self._item_type._fields  # Namedtuple or dataclass fields
        if provided_fields != set(expected_fields):
            raise ValueError(
                f"Incorrect fields provided. Expected fields: {expected_fields}, "
                f"but received: {provided_fields}"
            )
