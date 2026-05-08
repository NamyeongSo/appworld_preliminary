from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, cast


T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a context-engineering config is invalid."""


@dataclass(frozen=True)
class SamplingConfig:
    n_difficulty: int = 5
    n_pi_old: int = 10
    n_pi_theta: int = 10
    temperature_sampling: float = 0.1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> SamplingConfig:
        return _build_dataclass(
            cls,
            data,
            aliases={
                "n_difficulty": ("N_difficulty",),
                "n_pi_old": ("N_pi_old",),
                "n_pi_theta": ("N_pi_theta",),
            },
        )

    def __post_init__(self) -> None:
        _require_positive_int("n_difficulty", self.n_difficulty)
        _require_positive_int("n_pi_old", self.n_pi_old)
        _require_positive_int("n_pi_theta", self.n_pi_theta)
        _require_non_negative_float("temperature_sampling", self.temperature_sampling)


@dataclass(frozen=True)
class SmoothingConfig:
    k: float = 2.0
    alpha: float = 1.0
    action_match_level: str = "api_name_plus_argument_keys"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> SmoothingConfig:
        return _build_dataclass(cls, data, aliases={"k": ("K",)})

    def __post_init__(self) -> None:
        _require_positive_float("k", self.k)
        _require_non_negative_float("alpha", self.alpha)
        valid_action_match_levels = {
            "api_name",
            "api_name_plus_argument_keys",
            "api_name_plus_normalized_arguments",
        }
        if self.action_match_level not in valid_action_match_levels:
            raise ConfigError(
                "action_match_level must be one of "
                f"{sorted(valid_action_match_levels)}; got {self.action_match_level!r}"
            )


@dataclass(frozen=True)
class RatioFilterConfig:
    epsilon: float = 0.3
    high_ratio_policy: str = "defer_to_next_loop"
    low_ratio_policy: str = "discard"
    max_defer_rounds: int = 3

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> RatioFilterConfig:
        return _build_dataclass(cls, data)

    def __post_init__(self) -> None:
        _require_float_between("epsilon", self.epsilon, minimum=0.0, maximum=1.0)
        if self.high_ratio_policy != "defer_to_next_loop":
            raise ConfigError("high_ratio_policy currently supports only 'defer_to_next_loop'")
        if self.low_ratio_policy != "discard":
            raise ConfigError("low_ratio_policy currently supports only 'discard'")
        _require_positive_int("max_defer_rounds", self.max_defer_rounds)


@dataclass(frozen=True)
class DifficultyConfig:
    d_min: float = 0.2
    d_max: float = 1.0
    bin_edges: tuple[float, ...] = (0.0, 0.4, 0.6, 1.0)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> DifficultyConfig:
        return _build_dataclass(cls, data, aliases={"d_min": ("D_min",), "d_max": ("D_max",)})

    def __post_init__(self) -> None:
        _require_float_between("d_min", self.d_min, minimum=0.0, maximum=1.0, inclusive=True)
        _require_float_between("d_max", self.d_max, minimum=0.0, maximum=1.0, inclusive=True)
        if self.d_min > self.d_max:
            raise ConfigError(f"d_min ({self.d_min}) cannot exceed d_max ({self.d_max})")
        bin_edges = tuple(float(edge) for edge in self.bin_edges)
        if len(bin_edges) < 2:
            raise ConfigError("bin_edges must contain at least two values")
        if any(edge < 0.0 or edge > 1.0 for edge in bin_edges):
            raise ConfigError("bin_edges must stay within [0.0, 1.0]")
        if tuple(sorted(bin_edges)) != bin_edges:
            raise ConfigError("bin_edges must be sorted in non-decreasing order")
        object.__setattr__(self, "bin_edges", bin_edges)


@dataclass(frozen=True)
class CounterfactualConfig:
    success_timestep_strategy: str = "api_call_uniform"
    max_success_to_failure_attempts: int = 5
    max_backward_steps: int | str = "T"
    max_failure_to_success_attempts_per_step: int = 1

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> CounterfactualConfig:
        return _build_dataclass(cls, data)

    def __post_init__(self) -> None:
        if self.success_timestep_strategy != "api_call_uniform":
            raise ConfigError(
                "success_timestep_strategy currently supports only 'api_call_uniform'"
            )
        _require_positive_int(
            "max_success_to_failure_attempts", self.max_success_to_failure_attempts
        )
        if self.max_backward_steps != "T":
            _require_positive_int("max_backward_steps", self.max_backward_steps)
        _require_positive_int(
            "max_failure_to_success_attempts_per_step",
            self.max_failure_to_success_attempts_per_step,
        )


@dataclass(frozen=True)
class PolicyContextConfig:
    all_insight: bool = True
    max_context_items: int = 200

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> PolicyContextConfig:
        return _build_dataclass(cls, data)

    def __post_init__(self) -> None:
        _require_bool("all_insight", self.all_insight)
        _require_positive_int("max_context_items", self.max_context_items)


@dataclass(frozen=True)
class AppWorldConfig:
    agent_type: str = "react_code_agent"
    max_steps: int = 50

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> AppWorldConfig:
        return _build_dataclass(cls, data)

    def __post_init__(self) -> None:
        if not self.agent_type:
            raise ConfigError("agent_type cannot be empty")
        _require_positive_int("max_steps", self.max_steps)


@dataclass(frozen=True)
class OutputConfig:
    root: str | None = None
    loop_index: int = 0

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> OutputConfig:
        return _build_dataclass(cls, data)

    def __post_init__(self) -> None:
        if self.root is not None and not str(self.root).strip():
            raise ConfigError("root cannot be empty when provided")
        _require_non_negative_int("loop_index", self.loop_index)


@dataclass(frozen=True)
class OuterLoopConfig:
    max_loops: int = 1
    stop_when_no_deferred: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> OuterLoopConfig:
        return _build_dataclass(cls, data)

    def __post_init__(self) -> None:
        _require_positive_int("max_loops", self.max_loops)
        _require_bool("stop_when_no_deferred", self.stop_when_no_deferred)


@dataclass(frozen=True)
class EvaluationConfig:
    ablation_settings: tuple[str, ...] = (
        "no_context",
        "no_ratio_filtering",
        "no_defer",
        "no_curriculum",
        "no_counterfactual_replay",
    )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> EvaluationConfig:
        return _build_dataclass(cls, data)

    def __post_init__(self) -> None:
        ablation_settings = tuple(str(setting) for setting in self.ablation_settings)
        if not ablation_settings:
            raise ConfigError("ablation_settings must contain at least one setting")
        object.__setattr__(self, "ablation_settings", ablation_settings)


@dataclass(frozen=True)
class ContextEngineeringConfig:
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    ratio_filter: RatioFilterConfig = field(default_factory=RatioFilterConfig)
    difficulty: DifficultyConfig = field(default_factory=DifficultyConfig)
    counterfactual: CounterfactualConfig = field(default_factory=CounterfactualConfig)
    policy_context: PolicyContextConfig = field(default_factory=PolicyContextConfig)
    appworld: AppWorldConfig = field(default_factory=AppWorldConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    outer_loop: OuterLoopConfig = field(default_factory=OuterLoopConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    dry_run: bool = False

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None = None) -> ContextEngineeringConfig:
        if data is None:
            return cls()
        remaining = _copy_mapping(data, "context_engineering")
        config = cls(
            sampling=SamplingConfig.from_mapping(remaining.pop("sampling", None)),
            smoothing=SmoothingConfig.from_mapping(remaining.pop("smoothing", None)),
            ratio_filter=RatioFilterConfig.from_mapping(remaining.pop("ratio_filter", None)),
            difficulty=DifficultyConfig.from_mapping(remaining.pop("difficulty", None)),
            counterfactual=CounterfactualConfig.from_mapping(remaining.pop("counterfactual", None)),
            policy_context=PolicyContextConfig.from_mapping(remaining.pop("policy_context", None)),
            appworld=AppWorldConfig.from_mapping(remaining.pop("appworld", None)),
            output=OutputConfig.from_mapping(remaining.pop("output", None)),
            outer_loop=OuterLoopConfig.from_mapping(remaining.pop("outer_loop", None)),
            evaluation=EvaluationConfig.from_mapping(remaining.pop("evaluation", None)),
            dry_run=bool(remaining.pop("dry_run", False)),
        )
        if remaining:
            _raise_unknown_keys("context_engineering", remaining)
        return config

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], asdict(self))


def load_context_engineering_config(
    config: ContextEngineeringConfig | Mapping[str, Any] | str | Path | None = None,
) -> ContextEngineeringConfig:
    """Load a context-engineering config from a mapping or JSON file.

    Jsonnet experiment files are evaluated by the AppWorld experiment loader before
    reaching this package, so this loader intentionally accepts evaluated mappings
    rather than evaluating Jsonnet itself.
    """

    if isinstance(config, ContextEngineeringConfig):
        return config
    if config is None or isinstance(config, Mapping):
        return ContextEngineeringConfig.from_mapping(config)
    config_path = Path(config)
    if config_path.suffix != ".json":
        raise ConfigError(
            f"Only JSON config files are supported here; got {config_path}. "
            "Pass evaluated Jsonnet configs as mappings."
        )
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, Mapping):
        raise ConfigError(f"Config file {config_path} must contain a JSON object")
    return ContextEngineeringConfig.from_mapping(payload)


def _build_dataclass(
    cls: type[T],
    data: Mapping[str, Any] | None = None,
    *,
    aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> T:
    if data is None:
        return cls()
    remaining = _copy_mapping(data, cls.__name__)
    aliases = aliases or {}
    values: dict[str, Any] = {}
    for dataclass_field in fields(cls):
        field_name = dataclass_field.name
        if field_name in remaining:
            values[field_name] = remaining.pop(field_name)
            continue
        for alias in aliases.get(field_name, ()):  # Preserve doc-facing names like N_pi_old.
            if alias in remaining:
                values[field_name] = remaining.pop(alias)
                break
    if remaining:
        _raise_unknown_keys(cls.__name__, remaining)
    return cls(**values)


def _copy_mapping(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ConfigError(f"{name} must be a mapping; got {type(data).__name__}")
    return dict(data)


def _raise_unknown_keys(name: str, mapping: Mapping[str, Any]) -> None:
    keys = ", ".join(sorted(str(key) for key in mapping))
    raise ConfigError(f"Unexpected keys in {name}: {keys}")


def _require_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a bool; got {value!r}")


def _require_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConfigError(f"{name} must be a positive integer; got {value!r}")


def _require_non_negative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ConfigError(f"{name} must be a non-negative integer; got {value!r}")


def _require_positive_float(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or float(value) <= 0.0:
        raise ConfigError(f"{name} must be a positive number; got {value!r}")


def _require_non_negative_float(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or float(value) < 0.0:
        raise ConfigError(f"{name} must be a non-negative number; got {value!r}")


def _require_float_between(
    name: str,
    value: Any,
    *,
    minimum: float,
    maximum: float,
    inclusive: bool = False,
) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"{name} must be a number; got {value!r}")
    value_as_float = float(value)
    if inclusive:
        valid = minimum <= value_as_float <= maximum
    else:
        valid = minimum <= value_as_float < maximum
    if not valid:
        upper = "]" if inclusive else ")"
        raise ConfigError(f"{name} must be in [{minimum}, {maximum}{upper}; got {value!r}")
