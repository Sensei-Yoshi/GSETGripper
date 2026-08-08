"""E3: semantic-and-mass paired experiential retrieval."""

from ..retrieval import RetrievalMode
from .helper import RetrievalVLMExperiment


class E3Strategy(RetrievalVLMExperiment):
    include_measured = True
    retrieval_mode = RetrievalMode.HYBRID
