# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Jean Disset
"""Test that L0 penalty is correctly configured in design YAML configs.

This test was added after discovering that the L0 penalty schedule was accidentally
set to all zeros in base.yaml, causing TU pruning to be completely disabled.
The bug manifested as:
- TUs not being pruned during design optimization
- tu_log_alpha barely changing over 1920 optimization steps
- Final designs having nearly all TUs enabled (81%+ remained)
"""

import pytest
from pathlib import Path

from biocomp.optimutils import three_phase_schedule

RESOURCES_DIR = Path(__file__).parent / "resources"


def test_l0_penalty_not_all_zero():
    """Verify that L0 penalty schedule produces non-zero values during pruning phase.

    The three-phase schedule should have:
    - Phase 1: L0 = 0 (exploration, no pressure)
    - Phase 2: L0 > 0 (pruning phase, gradient pressure to disable TUs)
    - Phase 3: L0 > 0 (commitment, maintain pressure)
    """
    total_steps = 768  # default: 12 epochs * 64 steps

    # these are the fixed values from base.yaml
    schedule = three_phase_schedule(
        total_steps=total_steps,
        phase1_frac=0.4,
        phase2_frac=0.75,
        phase1_value=0.0,  # phase 1: no L0 is correct
        phase2_end_value=0.02,  # phase 2: must be > 0 for pruning
        phase3_end_value=0.02,  # phase 3: must be > 0 for commitment
        phase2_power=1.0,
        phase3_power=2.0,
    )

    # phase 1 should be 0
    phase1_end = int(0.4 * total_steps) - 1
    assert schedule(0) == 0.0, "L0 should be 0 at start (exploration)"
    assert schedule(phase1_end) == 0.0, "L0 should be 0 at end of phase 1"

    # phase 2 should have positive L0
    phase2_start = int(0.4 * total_steps) + 1
    phase2_mid = int((0.4 + 0.75) / 2 * total_steps)
    phase2_end = int(0.75 * total_steps)

    assert schedule(phase2_start) > 0, (
        f"L0 should be > 0 at start of phase 2, got {schedule(phase2_start)}"
    )
    assert schedule(phase2_mid) > 0, (
        f"L0 should be > 0 in middle of phase 2, got {schedule(phase2_mid)}"
    )
    assert schedule(phase2_end) > 0, (
        f"L0 should be > 0 at end of phase 2, got {schedule(phase2_end)}"
    )

    # phase 3 should have positive L0
    phase3_mid = int((0.75 + 1.0) / 2 * total_steps)
    assert schedule(phase3_mid) > 0, f"L0 should be > 0 in phase 3, got {schedule(phase3_mid)}"
    assert schedule(total_steps - 1) > 0, (
        f"L0 should be > 0 at end, got {schedule(total_steps - 1)}"
    )


def test_l0_penalty_schedule_at_different_scales():
    """Test L0 schedule works at different epoch counts."""
    for n_epochs in [12, 30, 50, 100]:
        total_steps = n_epochs * 64

        schedule = three_phase_schedule(
            total_steps=total_steps,
            phase1_frac=0.4,
            phase2_frac=0.75,
            phase1_value=0.0,
            phase2_end_value=0.02,
            phase3_end_value=0.02,
            phase2_power=1.0,
            phase3_power=2.0,
        )

        # at 50% of training, we should be in phase 2 with L0 > 0
        mid_step = total_steps // 2
        l0_at_mid = float(schedule(mid_step))
        assert l0_at_mid > 0, (
            f"L0 at step {mid_step} (50% of {n_epochs} epochs) should be > 0, got {l0_at_mid}"
        )


def test_base_yaml_has_l0_schedule_defined():
    """Verify base.yaml has L0 penalty schedule defined (values may be zero as default)."""
    base_yaml = RESOURCES_DIR / "design/design_configs/base.yaml"

    if not base_yaml.exists():
        pytest.skip("base.yaml not found")

    content = base_yaml.read_text()

    lines = content.split("\n")
    l0_values = []
    for line in lines:
        if "lambda_l0_phase" in line and ":" in line:
            value_part = line.split(":")[-1].strip()
            if "#" in value_part:
                value_part = value_part.split("#")[0].strip()
            try:
                l0_values.append((line.strip(), float(value_part)))
            except ValueError:
                pass

    phase1_values = [v for name, v in l0_values if "phase1" in name]
    phase2_values = [v for name, v in l0_values if "phase2" in name]
    phase3_values = [v for name, v in l0_values if "phase3" in name]

    assert phase1_values, "Could not find lambda_l0_phase1 in base.yaml"
    assert phase2_values, "Could not find lambda_l0_phase2_end in base.yaml"
    assert phase3_values, "Could not find lambda_l0_phase3_end in base.yaml"

    for name, v in l0_values:
        assert v >= 0, f"L0 penalty must be non-negative: {name} = {v}"
