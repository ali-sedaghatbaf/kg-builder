from datetime import date
from enum import Enum
from typing import Any, Dict

from pydantic import BaseModel, Field, field_validator, ValidationInfo


class ContractType(Enum):
    AFFILIATE_AGREEMENT = "Affiliate Agreement"
    CO_BRANDING = "Co-Branding"


class Location(BaseModel):
    country: str | None = Field(default=None, description="Country of the location")
    state: str | None = Field(default=None, description="State of the location")
    address: str | None = Field(default=None, description="Address of the location")


class ContractParty(BaseModel):
    name: str = Field(..., description="Name of the party")
    location: Location | None = Field(default=None, description="Location of the party")


class BaseContract(BaseModel):
    parties: list[ContractParty] | None = Field(
        default=None, description="Parties involved in the contract"
    )
    agreement_date: date | None = Field(
        default=None, description="Date of the agreement"
    )
    effective_date: date | None = Field(
        default=None, description="Effective date of the contract"
    )
    expiration_date: date | None = Field(
        default=None, description="Expiration date of the contract"
    )
    governing_law: str | None = Field(
        default=None, description="Governing law of the contract"
    )
    termination_for_convenience: bool = Field(
        default=False,
        description="Indicates if the contract can be terminated for convenience",
    )
    anti_assignment: bool = Field(
        default=False,
        description="Indicates if the contract has an anti-assignment clause",
    )
    cap_on_liability: bool = Field(
        default=False, description="Indicates if the contract has a cap on liability"
    )


class AffiliateAgreement(BaseContract):
    exclusivity: bool = Field(
        default=False, description="Indicates if the agreement is exclusive"
    )
    non_compete: bool = Field(
        default=False, description="Indicates if there is a non-compete clause"
    )
    revenue_profit_sharing: bool = Field(
        default=False, description="Indicates if there is revenue profit sharing"
    )
    minimum_commitment: bool = Field(
        default=False, description="Indicates if there is a minimum commitment"
    )


class CoBranding(BaseContract):
    exclusivity: bool = Field(
        default=False, description="Indicates if the co-branding is exclusive"
    )
    ip_ownership: str | None = Field(
        default=None, description="Intellectual property ownership details"
    )
    license_grant: str | None = Field(default=None, description="License grant details")
    license_profile_sharing_terms: str | None = Field(
        default=None, description="License profile sharing terms"
    )


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


class Ontology(BaseModel):
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
        for relation in v:
            if relation.subject not in entities:
                raise ValueError(f"Invalid subject entity: {relation.subject}")
            if relation.object not in entities:
                raise ValueError(f"Invalid object entity: {relation.object}")
        return v
