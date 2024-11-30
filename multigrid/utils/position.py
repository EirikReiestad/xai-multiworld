from dataclasses import dataclass


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

    def distance_to(self, other: "Position") -> float:
        """Calculate Euclidean distance between two positions."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def to_tuple_reversed(self) -> tuple[int, int]:
        """Return the position as a tuple with the values reversed (y, x)."""
        return (self.y, self.x)
