"""Luria-AIDO v0.1 - persistent-state virtual cell simulator.

Design (per RULER.md): NOT three independent predictors behind one import.
There is ONE shared latent cell state maintained by an engine. All readouts
(expression, regulatory activity, ...) decode from that same state, so they
are mutually consistent by construction - the defining property of a world
model and the basis of property tests P2/P3/P4.

Layers per AIDO:
  engine/ - the state itself, the intervention operators that update it,
            and the decoders that read from it (frozen encoders only;
            adapters are the only trainable parameters).
  shell/  - persistence of state across turns, and translation of user
            commands into engine steps.
"""
from luria_aido.cell import Cell
from luria_aido.version import __version__

__all__ = ["Cell", "__version__"]
