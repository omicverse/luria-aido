"""engine/ - the state itself and the operators on it.

Everything in here is state-first: an intervention changes the latent state,
and every readout decodes from the state. Nothing predicts perturbation
effects directly from perturbation embeddings - those enter only through the
state update rule, and readouts never see them again.
"""
from luria_aido.engine.state import CellState

__all__ = ["CellState"]
