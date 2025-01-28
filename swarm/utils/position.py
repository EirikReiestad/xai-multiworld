from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Position:
    x: int
    y: int

    def __add__(self, other: "Position") -> "Position":
        """Add two Position objects (vector addition)."""
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Position") -> "Position":
        """Subtract two Position objects."""
        return Position(self.x - other.x, self.y - other.y)

    def __repr__(self) -> str:
        """Custom string representation."""
        return f"Position(x={self.x}, y={self.y})"

    def __call__(self) -> tuple[int, int]:
        """Return the position as a tuple."""
        return (self.x, self.y)

    def __ge__(self, other: "Position") -> bool:
        """Greater than or equal to comparison."""
        return self.x >= other.x and self.y >= other.y

    def __lt__(self, other: tuple[int, int]) -> bool:
        """Less than comparison."""
        return self.x < other[0] and self.y < other[1]

    def distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance between two positions."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def to_tuple_reversed(self) -> tuple[int, int]:
        """Return the position as a tuple with the values reversed (y, x)."""
        return (self.y, self.x)

    @classmethod
    def from_list(cls, pos: list[tuple[int, int]]) -> NDArray["Position"]:
        """Create a Position object from a list."""
        x, y = zip(*pos)
        return np.array([cls(*p) for p in zip(x, y)])

    def to_list(self) -> list[tuple[int, int]]:
        """Return the position as a list."""
        return [(self.x, self.y)]

    def to_numpy(self) -> NDArray[np.int_]:
        """Return the position as a numpy array."""
        return np.array([np.int_(self.x), np.int_(self.y)])
