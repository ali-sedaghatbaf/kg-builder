from dspy import InputField, OutputField, Signature

from src.lm.models import AffiliateAgreement, CoBranding, ContractType


class ContractClassification(Signature):
    contract_text: str = InputField(description="The text of the contract to classify.")
    contract_type: ContractType = OutputField(description="The type of the contract")
    thoughts: str = OutputField(description="The reasoning behind the classification")


class ContractContentExtraction(Signature):
    contract_text: str = InputField(description="The text of the contract to analyze.")
    contract_type: ContractType = InputField(description="The type of the contract")
    extracted_information: AffiliateAgreement | CoBranding = OutputField(
        description="The extracted information from the contract"
    )
