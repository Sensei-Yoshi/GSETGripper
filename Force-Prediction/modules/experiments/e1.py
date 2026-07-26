"""E1: image-only zero-shot joint VLM."""

from .helper import JointVLMExperiment


class E1Strategy(JointVLMExperiment):
    include_measured = False
