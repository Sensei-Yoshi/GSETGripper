"""E4: hybrid semantic-and-sensor paired experiential retrieval."""

from ..retrieval import RetrievalMode
from .helper import RetrievalVLMExperiment


class E4Strategy(RetrievalVLMExperiment):
    include_measured = True
    retrieval_mode = RetrievalMode.HYBRID
