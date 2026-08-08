"""E2: semantic-only paired experiential retrieval, without sensor exposure."""

from ..retrieval import RetrievalMode
from .helper import RetrievalVLMExperiment


class E2Strategy(RetrievalVLMExperiment):
    include_measured = False
    retrieval_mode = RetrievalMode.SEMANTIC_ONLY
