"""E4: semantic-and-mass paired experiential retrieval."""

from ..retrieval import RetrievalMode
from .helper import RetrievalVLMExperiment


class E4Strategy(RetrievalVLMExperiment):
    include_measured = True
    retrieval_mode = RetrievalMode.HYBRID
