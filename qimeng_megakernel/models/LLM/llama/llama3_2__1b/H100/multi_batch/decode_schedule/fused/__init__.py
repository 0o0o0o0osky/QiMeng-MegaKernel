"""Hardware-independent fused oracle reused by the H100 multi-batch route."""

from ....single_batch.decode_schedule.fused import layer_scheduler, operations

__all__ = ["layer_scheduler", "operations"]
