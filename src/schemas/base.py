from typing import Annotated, Any, Generic

from pydantic import BaseModel, BeforeValidator, ConfigDict  # type: ignore
from pydantic.alias_generators import to_camel

from src.schemas.types import T


def round_probability(value: float) -> float:
    """Round a float value to two decimal places.

    Parameters
    ----------
        value (float): The float value to be rounded.

    Returns
    -------
        float: Rounded value.
    """
    try:
        return round(float(value), 2)
    except (TypeError, ValueError):
        return value


Float = Annotated[float, BeforeValidator(round_probability)]


class BaseSchema(BaseModel):
    """Base schema class that inherits from Pydantic BaseModel.

    This class provides common configuration for all schema classes including
    camelCase alias generation, population by field name, and attribute mapping.
    """

    model_config: ConfigDict = ConfigDict(  # type: ignore
        alias_generator=to_camel,  # Convert field names to camelCase
        populate_by_name=True,
        from_attributes=True,
        arbitrary_types_allowed=True,  # Allow non-standard types like DataFrame objects, etc.
        validate_assignment=True,  # Validate changes after creation
        str_strip_whitespace=True,  # Strip whitespace from strings
        frozen=True,  # Make instances immutable
        use_enum_values=True,  # Serialize enums as their values
    )


class ModelList(BaseModel, Generic[T]):
    """Generic container for lists of Pydantic BaseModel objects.

    This class provides type-safe handling of any BaseModel subclass,
    including LangChain Documents, with validation and utility methods.

    Parameters
    ----------
    items : list[T]
        List of BaseModel instances.

    Examples
    --------
    >>> # With Documents
    >>> docs = ModelList[Document](items=[
    ...     Document(page_content="Hello", metadata={"source": "test"}),
    ...     Document(page_content="World", metadata={"source": "test2"})
    ... ])

    >>> # With custom Pydantic models
    >>> class User(BaseModel):
    ...     name: str
    ...     age: int
    >>>
    >>> users = ModelList[User](items=[
    ...     User(name="Alice", age=30),
    ...     User(name="Bob", age=25)
    ... ])
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    items: list[T]

    def __len__(self) -> int:
        """Return the number of items."""
        return len(self.items)

    def __getitem__(self, index: int) -> T:
        """Get item by index."""
        return self.items[index]

    def append(self, item: T) -> None:
        """Add an item to the list."""
        self.items.append(item)

    def extend(self, items: list[T]) -> None:
        """Extend the list with multiple items."""
        self.items.extend(items)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Convert all items to dictionaries."""
        return [
            item.model_dump() if hasattr(item, "model_dump") else dict(item)
            for item in self.items
        ]

    @classmethod
    def from_dicts(
        cls, data: list[dict[str, Any]], model_class: type[T]
    ) -> "ModelList[T]":
        """Create a ModelList from a list of dictionaries.

        Parameters
        ----------
        data : list[dict[str, Any]]
            List of dictionaries to convert.
        model_class : type[T]
            The Pydantic model class to instantiate.

        Returns
        -------
        ModelList[T]
            A new ModelList instance.
        """
        items = [model_class(**item) for item in data]
        return cls(items=items)
