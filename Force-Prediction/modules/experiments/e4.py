"""E4: E3 plus continuous roughness evidence."""

from ..retrieval import RetrievalMode
from .helper import RetrievalVLMExperiment


class E4Strategy(RetrievalVLMExperiment):
    include_measured = True
    retrieval_mode = RetrievalMode.HYBRID
