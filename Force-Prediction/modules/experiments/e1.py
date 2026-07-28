"""E1: image-only zero-shot VLM for the active grippers."""

from .helper import JointVLMExperiment


class E1Strategy(JointVLMExperiment):
    include_measured = False
