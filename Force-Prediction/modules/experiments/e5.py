"""E5: semantic, mass, roughness, and projected-contact evidence."""

from ..retrieval import RetrievalMode
from .helper import RetrievalVLMExperiment


class E5Strategy(RetrievalVLMExperiment):
    include_measured = True
    retrieval_mode = RetrievalMode.HYBRID
