"""E6: full semantic, mass, roughness, and projected-contact retrieval."""

from ..retrieval import RetrievalMode
from .helper import RetrievalVLMExperiment


class E6Strategy(RetrievalVLMExperiment):
    include_measured = True
    retrieval_mode = RetrievalMode.HYBRID
