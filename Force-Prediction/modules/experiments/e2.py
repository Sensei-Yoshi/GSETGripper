"""E2: image plus authoritative measurements, without retrieval."""

from .helper import JointVLMExperiment


class E2Strategy(JointVLMExperiment):
    include_measured = True
