import os

import dspy


def setup_dspy():
    """Sets up the dspy configuration using environment variables.
    Raises:
        ValueError: If required environment variables are not set.
    """
    openai_model = os.getenv("OPENAI_MODEL")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    colbert_url = os.getenv("COLBERT_URL", "http://localhost:8000")

    if not openai_model or not openai_api_key:
        raise ValueError(
            "OPENAI_MODEL and OPENAI_API_KEY must be set as environment variables."
        )

    if not openai_model.startswith("openai/"):
        openai_model = "openai/" + openai_model

    dspy.settings.configure(
        lm=dspy.LM(model=openai_model, api_key=openai_api_key, temperature=0.0),
        rm=dspy.ColBERTv2(url=colbert_url),
    )
    return dspy.settings.lm
