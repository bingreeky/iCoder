"""Observable, applicability-gated VE-hard mutation contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from .ir import RootTaskIR


@dataclass(frozen=True)
class MutationContract:
    name: str
    applicable_classes: Tuple[str, ...]
    dimensions: Mapping[str, Sequence[Any]]
    functional_delta: Tuple[str, ...]
    temporal_delta: Tuple[str, ...]
    test_obligations: Tuple[str, ...]
    forbidden_mutations: Tuple[str, ...]
    added_ports: Tuple[Tuple[str, str, str], ...]

    def applicable(self, root: RootTaskIR) -> bool:
        return root.task_class in self.applicable_classes

    def parameters(self, ordinal: int) -> Dict[str, Any]:
        values: Dict[str, Any] = {}
        stride = 1
        for name, choices in self.dimensions.items():
            if not choices:
                raise ValueError(f"{self.name}.{name} has no choices")
            values[name] = choices[(ordinal // stride) % len(choices)]
            stride *= len(choices)
        return values


ALL_CLASSES = (
    "combinational_arithmetic", "sequential_counter", "fsm_protocol",
    "stream", "memory_shared_resource",
)

CONTRACTS: Dict[str, MutationContract] = {
    "controllerize_datapath": MutationContract(
        "controllerize_datapath", ALL_CLASSES,
        {
            "latency": (2, 4, 8),
            "start_mode": ("pulse", "level_until_accept"),
            "result_mode": ("done_pulse", "done_until_ack"),
        },
        ("Preserve the parent computation for every legal transaction.",),
        ("Latch inputs on start and produce a one-cycle done pulse after the configured latency.",),
        ("input latching", "exact latency", "one-cycle done", "back-to-back transactions"),
        ("no combinational done", "no sampling changing inputs after start"),
        (("input", "clk", "1"), ("input", "reset", "1"),
         ("input", "start", "1"), ("output", "busy", "1"),
         ("output", "done", "1"), ("input", "result_ack", "1")),
    ),
    "add_handshake_backpressure": MutationContract(
        "add_handshake_backpressure", ALL_CLASSES,
        {
            "capacity": (1, 2, 4),
            "latency": (1, 2, 4),
            "flow": ("registered", "elastic"),
        },
        ("Preserve transaction results and transaction order.",),
        ("Use valid/ready transfer semantics and hold payload stable under stall.",),
        ("stall stability", "no drop or duplicate", "back-to-back traffic", "reset while stalled"),
        ("no state advance under output stall", "no acceptance without valid and ready"),
        (("input", "clk", "1"), ("input", "reset", "1"),
         ("input", "in_valid", "1"), ("output", "in_ready", "1"),
         ("output", "out_valid", "1"), ("input", "out_ready", "1")),
    ),
    "add_buffer_fifo": MutationContract(
        "add_buffer_fifo", ("stream", "fsm_protocol", "memory_shared_resource"),
        {"depth": (2, 4, 8)},
        ("Buffer parent transactions without changing their values or order.",),
        ("Expose correct full/empty semantics and support simultaneous push/pop.",),
        ("full and empty boundaries", "pointer wrap", "simultaneous push/pop", "ordered drain"),
        ("no overwrite when full", "no read advancement when empty"),
        (("input", "clk", "1"), ("input", "reset", "1"),
         ("input", "push", "1"), ("input", "pop", "1"),
         ("output", "full", "1"), ("output", "empty", "1")),
    ),
    "add_timeout_retry_recovery": MutationContract(
        "add_timeout_retry_recovery", ("fsm_protocol", "stream", "memory_shared_resource"),
        {"timeout": (3, 5, 8), "retries": (1, 2, 3)},
        ("Preserve successful parent transactions and surface terminal failure.",),
        ("Timeout on the exact configured boundary, retry a bounded number of times, then recover to idle.",),
        ("timeout boundary", "retry count", "successful retry", "terminal recovery"),
        ("no unbounded retry", "no stale response accepted after recovery"),
        (("input", "clk", "1"), ("input", "reset", "1"),
         ("input", "request", "1"), ("input", "response_valid", "1"),
         ("output", "retry", "1"), ("output", "timeout_error", "1")),
    ),
    "add_arbitration_or_burst": MutationContract(
        "add_arbitration_or_burst", ("stream", "fsm_protocol", "memory_shared_resource"),
        {"mode": ("fixed-2", "rr-2", "rr-4", "burst-2", "burst-4", "burst-8")},
        ("Apply the parent operation independently to each accepted request.",),
        ("Arbitrate or retain ownership for the configured burst with observable fairness/order semantics.",),
        ("simultaneous request priority", "round-robin rotation", "burst count", "release under stall"),
        ("no starvation in round-robin mode", "no owner change mid-burst"),
        (("input", "clk", "1"), ("input", "reset", "1"),
         ("input", "request_valid", "4"), ("output", "grant", "4")),
    ),
}
