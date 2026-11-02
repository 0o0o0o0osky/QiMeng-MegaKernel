"""S1 op: baseline and multi-batch fused UpGate projection with SiLU."""
import pytest
import torch
import torch.nn.functional as F

from qimeng_megakernel.models.LLM.llama.llama3_2__1b.A100.multi_batch.decode_schedule.fused import operations as fused
from qimeng_megakernel.verifier.common.compare import assert_max_rel
from qimeng_megakernel.verifier.common.stage_config import ElementwiseGate
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.real_data_utils import (
    LOCAL_CHECKPOINT_AVAILABLE,
    distinct_prompts_for_batch,
    make_globs_from_real_prefill_multi,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.single_batch.baseline_to_fused._baseline_globs import (
    expected_layer0_o_proj_from_baseline_hf,
    select_batch_row,
)
from qimeng_megakernel.verifier.LLM.llama.llama3_2__1b.A100.verifier_config import MULTI_BATCH_CONFIG as CONFIG

pytestmark = pytest.mark.skipif(
    not LOCAL_CHECKPOINT_AVAILABLE,
    reason="set MODEL_LLAMA3_2_1B to a local checkpoint directory",
)


@torch.inference_mode()
def test_upgate_silu_layer0_real_data() -> None:
    globs, model = make_globs_from_real_prefill_multi(
        distinct_prompts_for_batch(2), return_model=True
    )
    layer = model.model.layers[0]
    hidden = torch.stack(
        [
            expected_layer0_o_proj_from_baseline_hf(
                select_batch_row(globs, batch_idx), model
            )["hidden_states"]
            for batch_idx in range(2)
        ]
    )
    normed = layer.post_attention_layernorm(hidden)
    expected = F.silu(layer.mlp.gate_proj(normed)) * layer.mlp.up_proj(normed)

    fused.fused_rms_qkvproj_gemv_rope(globs, 0)
    fused.fused_decode_attention(globs, 0)
    fused.fused_oproj_gemv_residual(globs, 0)
    fused.fused_mlp_upgate(globs, 0)

    gate = CONFIG.gate("S1", "op")
    assert isinstance(gate, ElementwiseGate)
    assert gate.max_rel is not None and gate.abs_floor is not None and gate.atol is not None
    assert_max_rel(
        globs.silu_out,
        expected,
        max_rel=gate.max_rel,
        abs_floor=gate.abs_floor,
        atol=gate.atol,
        name="silu_out",
    )
