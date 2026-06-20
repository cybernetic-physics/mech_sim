"""Benchmark suite registry.

The CLI driver iterates this registry to materialize every task in
the initial suite. Each entry pairs a family name with a generator
class so ``mech-bench generate-suite`` can produce N tasks per family.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from mech_bench.generators.base import (
    GeneratedTask,
    TaskGenerator,
    write_task_directory,
)
from mech_bench.generators.fourbar import FourbarPathGenerator
from mech_bench.generators.gear_train import (
    BeltPulleyRatioGenerator,
    ContactGearPairStubGenerator,
    CycloidalLayoutRatioGenerator,
    CycloidalLowNStubGenerator,
    RackPinionConversionGenerator,
    SpurGearRatioAnalyticGenerator,
)
from mech_bench.generators.slider_crank import SliderCrankStrokeGenerator
from mech_bench.generators.static_fit import (
    ShaftCollarClearanceGenerator,
    SimpleHingeFitGenerator,
    StaticFitBracketGenerator,
)

# Newly-added (Part B) generator families.
from mech_bench.generators.static_analytic import (
    BearingSeatClearanceGenerator,
    BoxLidRegisterFitGenerator,
    FlangeBoltCircleGenerator,
    KeyedShaftHubFitGenerator,
    MountingPlateHolePitchGenerator,
    PressFitHubInterferenceGenerator,
    PulleyBoreAlignmentStaticGenerator,
    SnapTabClearanceStaticGenerator,
    SpacerStackHeightGenerator,
    StandoffPatternSquareGenerator,
)
from mech_bench.generators.planar_kinematics_extra import (
    FourbarCrankRockerSweepGenerator,
    FourbarDwellPathGenerator,
    FourbarPumpHandleGenerator,
    FourbarStraightLineApproxGenerator,
    FourbarWiperArcGenerator,
    ReciprocatingPumpPlungerGenerator,
    RockerLimitStopTopologyGenerator,
    SliderCrankQuickReturnProxyGenerator,
    SliderCrankStrokePrecisionGenerator,
    ToggleOvercenterMarginGenerator,
)
from mech_bench.generators.transmission_analytic import (
    BevelGearRatioAnalyticGenerator,
    ChainSprocketRatioGenerator,
    CompoundGearVelocityGenerator,
    CompoundGearRatioAnalyticGenerator,
    IdlerGearDirectionAnalyticGenerator,
    LeadScrewLinearTravelGenerator,
    PlanetaryFixedRingVelocityGenerator,
    PlanetaryFixedRingRatioGenerator,
    PlanetaryFixedSunVelocityGenerator,
    PlanetaryFixedSunRatioGenerator,
    RackPinionForceDirectionGenerator,
    ShaftBearingCouplingVelocityGenerator,
    TimingBeltCenterDistanceGenerator,
    WormGearRatioAnalyticGenerator,
)
from mech_bench.generators.contact_synth import (
    BrakeCaliperContactStubGenerator,
    CamFollowerContactStubGenerator,
    DetentSpringContactStubGenerator,
    FrictionClutchTorqueStubGenerator,
    GearPairLoadTrialStubGenerator,
    GenevaIndexingStubGenerator,
    LatchReleaseForceStubGenerator,
    ParallelGripperRetentionStubGenerator,
    RackPinionContactStubGenerator,
    RatchetPawlEngagementStubGenerator,
)


# Ordered: Tier 0 → Tier 3, families listed in the task description.
SUITE: list[type[TaskGenerator]] = [
    # Pre-existing Tier 0
    StaticFitBracketGenerator,
    ShaftCollarClearanceGenerator,
    SimpleHingeFitGenerator,
    # Part B — Tier 0 additions
    MountingPlateHolePitchGenerator,
    FlangeBoltCircleGenerator,
    BearingSeatClearanceGenerator,
    PressFitHubInterferenceGenerator,
    KeyedShaftHubFitGenerator,
    SpacerStackHeightGenerator,
    StandoffPatternSquareGenerator,
    PulleyBoreAlignmentStaticGenerator,
    SnapTabClearanceStaticGenerator,
    BoxLidRegisterFitGenerator,
    # Pre-existing Tier 1
    FourbarPathGenerator,
    SliderCrankStrokeGenerator,
    # Part B — Tier 1 additions
    FourbarCrankRockerSweepGenerator,
    FourbarWiperArcGenerator,
    FourbarStraightLineApproxGenerator,
    FourbarDwellPathGenerator,
    FourbarPumpHandleGenerator,
    SliderCrankStrokePrecisionGenerator,
    SliderCrankQuickReturnProxyGenerator,
    ReciprocatingPumpPlungerGenerator,
    ToggleOvercenterMarginGenerator,
    RockerLimitStopTopologyGenerator,
    # Pre-existing Tier 2
    SpurGearRatioAnalyticGenerator,
    RackPinionConversionGenerator,
    BeltPulleyRatioGenerator,
    CycloidalLayoutRatioGenerator,
    # Part B — Tier 2 additions
    CompoundGearRatioAnalyticGenerator,
    CompoundGearVelocityGenerator,
    IdlerGearDirectionAnalyticGenerator,
    PlanetaryFixedRingRatioGenerator,
    PlanetaryFixedRingVelocityGenerator,
    PlanetaryFixedSunRatioGenerator,
    PlanetaryFixedSunVelocityGenerator,
    WormGearRatioAnalyticGenerator,
    LeadScrewLinearTravelGenerator,
    ShaftBearingCouplingVelocityGenerator,
    BevelGearRatioAnalyticGenerator,
    ChainSprocketRatioGenerator,
    TimingBeltCenterDistanceGenerator,
    RackPinionForceDirectionGenerator,
    # Pre-existing Tier 3 (capability-unavailable stubs)
    ContactGearPairStubGenerator,
    CycloidalLowNStubGenerator,
    # Part B — Tier 3 additions (synthetic fake-oracle stubs)
    CamFollowerContactStubGenerator,
    RatchetPawlEngagementStubGenerator,
    GenevaIndexingStubGenerator,
    FrictionClutchTorqueStubGenerator,
    BrakeCaliperContactStubGenerator,
    ParallelGripperRetentionStubGenerator,
    LatchReleaseForceStubGenerator,
    DetentSpringContactStubGenerator,
    GearPairLoadTrialStubGenerator,
    RackPinionContactStubGenerator,
]


def family_names() -> list[str]:
    return [g.family for g in SUITE]


def generator_for(family: str) -> type[TaskGenerator]:
    for g in SUITE:
        if g.family == family:
            return g
    raise KeyError(f"Unknown family: {family!r}. Known: {family_names()!r}")


def generate_suite(
    out_dir: Path,
    *,
    count_per_family: int = 5,
    base_seed: int = 0,
    families: Iterable[str] | None = None,
    difficulty: int | None = None,
) -> list[Path]:
    """Write every task in the suite to *out_dir* and return their paths.

    Each generator gets seeds ``base_seed + i`` for ``i`` in
    ``range(count_per_family)`` so re-running with the same seed
    deterministically reproduces the same suite.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    want = set(families) if families else None
    written: list[Path] = []
    for cls in SUITE:
        if want is not None and cls.family not in want:
            continue
        gen = cls()
        for i in range(count_per_family):
            seed = base_seed + i
            d = difficulty if difficulty is not None else _default_difficulty(
                cls.tier)
            task: GeneratedTask = gen.generate(seed=seed, difficulty=d)
            written.append(write_task_directory(task, out_dir))
    return written


def _default_difficulty(tier: str) -> int:
    return {
        "artifact_static": 1,
        "planar_kinematics": 2,
        "transmission_analytic": 2,
        "contact_dynamics": 3,
    }.get(tier, 2)
