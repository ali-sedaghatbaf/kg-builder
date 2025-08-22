from bdb import effective
from datetime import date
from enum import Enum

from numpy import minimum
from pydantic import BaseModel, Field


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
