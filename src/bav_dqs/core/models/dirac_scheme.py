"""Time-integration scheme of the Dirac use case (pure Python, framework-free).

``INTEGRATOR_ORDER`` is the global error order of one BAV-DQS step (the circuit built by
``dirac_circuits.build_step_circuit`` is a first-order Lie-Trotter product formula). It is checked
numerically against the exact propagator in ``tests/test_trotter_order.py``, so it cannot silently drift
from the circuit. Richardson extrapolation must use ``order_p == INTEGRATOR_ORDER``.
"""
SCHEME = "lie"
INTEGRATOR_ORDER = 1
