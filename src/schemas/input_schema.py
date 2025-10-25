from typing import Annotated

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field  # type: ignore
from pydantic.alias_generators import to_camel

from .types import DataSource, VectorSearchType, YesOrNo


def round_probability(value: float) -> float:
    """Round a float value to two decimal places.

    Returns:
    --------
        float: Rounded value.
    """
    if isinstance(value, float):
        return round(value, 2)
    return value


Float = Annotated[float, BeforeValidator(round_probability)]


class BaseSchema(BaseModel):
    """Base schema class that inherits from Pydantic BaseModel.

    This class provides common configuration for all schema classes including
    camelCase alias generation, population by field name, and attribute mapping.
    """

    model_config: ConfigDict = ConfigDict(  # type: ignore
        alias_generator=to_camel,
        populate_by_name=True,
        from_attributes=True,
        arbitrary_types_allowed=True,
    )


class RouteQuerySchema(BaseModel):
    """Route query model."""

    data_source: DataSource = Field(description="The data source to use for the query.")


class VectorSearchTypeSchema(BaseModel):
    """Vector search type model."""

    vector_search_type: VectorSearchType = Field(
        description="The vector search type to use for the query."
    )


class GradeRetrievalSchema(BaseModel):
    """Grade retrieval model."""

    is_relevant: YesOrNo = Field(
        description="Whether the retrieved documents are relevant to the user query."
    )


class GradeResponseSchema(BaseModel):
    """Grade response model."""

    is_relevant: YesOrNo = Field(
        description="Whether the response is relevant to the user query."
    )


class HallucinationSchema(BaseModel):
    """Check hallucination model."""

    is_hallucinating: YesOrNo = Field(
        description="Whether the response contains hallucinations."
    )
