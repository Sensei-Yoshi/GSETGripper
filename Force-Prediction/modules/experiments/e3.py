"""E3: semantic-only paired experiential retrieval, without sensor exposure."""

from ..retrieval import RetrievalMode
from .helper import RetrievalVLMExperiment


class E3Strategy(RetrievalVLMExperiment):
    include_measured = False
    retrieval_mode = RetrievalMode.SEMANTIC_ONLY
