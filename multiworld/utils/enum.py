import aenum as enum
from typing import Any
import functools
from numpy.typing import NDArray
import numpy as np


@functools.cache
def _enum_array(enum_cls: enum.EnumMeta) -> NDArray:
    """
    Return an array of all values of the given enum.

    Parameters
    ----------
    enum_cls : enum.EnumMeta
        Enum class
    """
    return np.array([item.value for item in enum_cls])


@functools.cache
def _enum_index(enum_item: enum.Enum):
    """
    Return the index of the given enum item.

    Parameters
    ----------
    enum_item : enum.Enum
        Enum item
    """
    return list(enum_item.__class__).index(enum_item)


class IndexedEnum(enum.Enum):
    """
    Enum where each member has a corresponding integer index.
    """

    def __int__(self):
        return self.to_index()

    @classmethod
    def add_item(cls, name: str, value: Any):
        """
        Add a new enumeration member.
        """
        enum.extend_enum(cls, name, value)
        _enum_array.cache_clear()
        _enum_index.cache_clear()

    @classmethod
    def from_index(cls, index: int) -> str:
        """
        Get the enumeration member corresponding to a given index.
        """
        out = _enum_array(cls)[index]
        return cls(out) if out.ndim == 0 else out

    def to_index(self) -> int:
        """
        Get the index of this enumeration member.
        """
        return _enum_index(self)
