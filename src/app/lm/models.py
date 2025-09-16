from typing import Any, Dict

from pydantic import BaseModel, Field, ValidationInfo, field_validator


class EntityType(BaseModel):
    name: str = Field(..., description="Name of the entity type")
    description: str | None = Field(
        default=None, description="Description of the entity type"
    )
    supertype: str | None = Field(
        default=None, description="Supertype of the entity type"
    )
    property_names: list[str] | None = Field(
        default=None, description="Property names of the entity type"
    )


class RelationType(BaseModel):
    name: str = Field(..., description="Name of the relation type")
    description: str | None = Field(
        default=None, description="Description of the relation type"
    )
    property_names: list[str] | None = Field(
        default=None, description="Property names of the relation type"
    )


class Schema(BaseModel):
    entity_types: list[EntityType] | None = Field(
        default=None, description="List of entity types"
    )
    relation_types: list[RelationType] | None = Field(
        default=None, description="List of relation types"
    )


class Entity(BaseModel):
    name: str = Field(..., description="Name of the entity")
    type: EntityType = Field(..., description="Type of the entity")
    properties: Dict[str, Any] | None = Field(
        default=None, description="Properties of the entity"
    )

    @field_validator("properties", mode="after")
    @classmethod
    def validate_properties(
        cls, v: Dict[str, Any], info: ValidationInfo
    ) -> Dict[str, Any]:
        entity_type = info.data.get("type")
        if entity_type is None:
            raise ValueError("Entity type is required")
        acceptable_properties = entity_type.property_names
        if acceptable_properties is not None:
            for key, value in v.items():
                if key not in acceptable_properties:
                    raise ValueError(
                        f"Invalid property: {key} for entity of type {entity_type.name}"
                    )
        return v


class Relation(BaseModel):
    name: str = Field(..., description="Name of the relation")
    type: RelationType = Field(..., description="Type of the relation")
    subject: Entity = Field(..., description="Subject of the relation")
    object: Entity = Field(..., description="Object of the relation")
    properties: Dict[str, Any] | None = Field(
        default=None, description="Properties of the relation"
    )

    @field_validator("properties", mode="after")
    @classmethod
    def validate_properties(
        cls, v: Dict[str, Any], info: ValidationInfo
    ) -> Dict[str, Any]:
        rel_type = info.data.get("type")
        if rel_type is None:
            raise ValueError("Relation type is required")
        acceptable_properties = rel_type.property_names
        if acceptable_properties is not None:
            for key, value in v.items():
                if key not in acceptable_properties:
                    raise ValueError(
                        f"Invalid property: {key} for relation of type {rel_type.name}"
                    )
        return v


class KnowledgeGraph(BaseModel):
    entities: list[Entity] = Field(..., description="List of entities")
    relations: list[Relation] = Field(..., description="List of relations")

    @field_validator("relations")
    @classmethod
    def validate_relations(
        cls, v: list[Relation], info: ValidationInfo
    ) -> list[Relation]:
        entities = info.data.get("entities")
        if entities is None:
            raise ValueError("Entities are required")

        for relation in v:
            if relation.subject not in entities:
                raise ValueError(f"Invalid subject entity: {relation.subject}")
            if relation.object not in entities:
                raise ValueError(f"Invalid object entity: {relation.object}")
        return v
