from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional


SUPPORTED_BOUNDARIES = {"zero_exterior_ghost", "forward_compatible_padding"}
SUPPORTED_SPATIAL_ORDERS = {2, 4}
SUPPORTED_DAMPING_PROFILES = {"quadratic"}
SUPPORTED_DAMPING_VELOCITY_REFERENCES = {"minimum"}
SUPPORTED_DAMPING_CORNER_COMBINATIONS = {"maximum"}
SUPPORTED_SOURCE_TYPES = {"point_ricker_spectrum"}
SUPPORTED_SOURCE_POSITION_MODES = {"fractional", "grid_index"}
SUPPORTED_SOURCE_PHASE_MODES = {"zero"}
SUPPORTED_INLINE_VELOCITY_MODES = {"inline_layered"}


class ConfigError(ValueError):
    """Raised when a JSON configuration is incomplete or unsupported."""


@dataclass(frozen=True)
class InputConfig:
    velocity_file: str
    model_index: Optional[int]


@dataclass(frozen=True)
class InlineVelocityConfig:
    mode: str
    values_m_per_s: list[list[float]]


@dataclass(frozen=True)
class GridConfig:
    dx_m: float
    dz_m: float
    spatial_order: int = 2


@dataclass(frozen=True)
class DampingProfileConfig:
    profile: str = "quadratic"
    power: float = 2.0
    target_decay: float = 1.0e-7
    strength_scale: float = 1.0
    velocity_reference: str = "minimum"
    corner_combination: str = "maximum"


@dataclass(frozen=True)
class BoundaryConfig:
    type: str
    top_padding_cells: int = 0
    bottom_padding_cells: int = 0
    left_padding_cells: int = 0
    right_padding_cells: int = 0
    damping: DampingProfileConfig = DampingProfileConfig()


@dataclass(frozen=True)
class SourcePositionConfig:
    mode: str
    x_fraction: Optional[float] = None
    z_fraction: Optional[float] = None
    ix: Optional[int] = None
    iz: Optional[int] = None


@dataclass(frozen=True)
class SourceConfig:
    type: str
    position: SourcePositionConfig
    peak_frequency_hz: float
    strength: float
    phase_mode: str


@dataclass(frozen=True)
class ReceiverConfig:
    positions: tuple[SourcePositionConfig, ...] = ()


@dataclass(frozen=True)
class OutputConfig:
    directory: str
    export_npz: bool = True
    export_mtx: bool = True
    save_velocity_preview: bool = True


@dataclass(frozen=True)
class RunConfig:
    run_name: str
    input: Optional[InputConfig]
    velocity: Optional[InlineVelocityConfig]
    grid: GridConfig
    frequencies_hz: tuple[float, ...]
    boundary: BoundaryConfig
    source: SourceConfig
    receivers: ReceiverConfig
    output: OutputConfig
    raw: dict[str, Any]
    config_path: Path


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ConfigError("Top-level JSON value must be an object.")

    run_name = _require_nonempty_str(raw, "run_name", "config")

    has_input = "input" in raw
    has_velocity = "velocity" in raw
    if has_input == has_velocity:
        raise ConfigError("Config must contain exactly one of 'input' or 'velocity'.")

    input_config = _parse_input(raw["input"]) if has_input else None
    velocity_config = _parse_inline_velocity(raw["velocity"]) if has_velocity else None

    grid_raw = _require_mapping(raw, "grid", "config")
    spatial_order = grid_raw.get("spatial_order", 2)
    if isinstance(spatial_order, bool) or not isinstance(spatial_order, int):
        raise ConfigError("grid.spatial_order must be an integer.")
    if spatial_order not in SUPPORTED_SPATIAL_ORDERS:
        raise ConfigError(
            f"Unsupported grid.spatial_order {spatial_order!r}; supported values: "
            f"{sorted(SUPPORTED_SPATIAL_ORDERS)}."
        )
    grid = GridConfig(
        dx_m=_require_positive_float(grid_raw, "dx_m", "grid"),
        dz_m=_require_positive_float(grid_raw, "dz_m", "grid"),
        spatial_order=spatial_order,
    )

    frequencies_raw = _require_key(raw, "frequencies_hz", "config")
    if not isinstance(frequencies_raw, list) or not frequencies_raw:
        raise ConfigError("config.frequencies_hz must be a non-empty list.")
    frequencies = tuple(
        _positive_float_value(v, f"config.frequencies_hz[{i}]")
        for i, v in enumerate(frequencies_raw)
    )

    boundary_raw = _require_mapping(raw, "boundary", "config")
    boundary_type = _require_nonempty_str(boundary_raw, "type", "boundary")
    if boundary_type not in SUPPORTED_BOUNDARIES:
        raise ConfigError(
            "Unsupported boundary.type "
            f"{boundary_type!r}; supported values: {sorted(SUPPORTED_BOUNDARIES)}."
        )
    boundary = _parse_boundary(boundary_raw, boundary_type)

    source = _parse_source(_require_mapping(raw, "source", "config"))
    receivers = _parse_receivers(raw.get("receivers"))
    output = _parse_output(_require_mapping(raw, "output", "config"))

    return RunConfig(
        run_name=run_name,
        input=input_config,
        velocity=velocity_config,
        grid=grid,
        frequencies_hz=frequencies,
        boundary=boundary,
        source=source,
        receivers=receivers,
        output=output,
        raw=raw,
        config_path=config_path,
    )


def find_project_root(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser().resolve()
    search_start = path.parent if path.is_file() else path
    for candidate in (search_start, *search_start.parents):
        if (candidate / "pyproject.toml").exists() and (
            candidate / "src" / "fd_converter"
        ).exists():
            return candidate
    if search_start.name == "configs":
        return search_start.parent
    return Path.cwd().resolve()


def resolve_input_path(path_value: str, config_path: Path, project_root: Path) -> Path:
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()

    candidates = [
        config_path.parent / raw_path,
        project_root / raw_path,
        Path.cwd().resolve() / raw_path,
        project_root.parent / raw_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (project_root / raw_path).resolve()


def resolve_output_dir(path_value: str, project_root: Path) -> Path:
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        return raw_path.resolve()
    return (project_root / raw_path).resolve()


def _parse_input(raw: Any) -> InputConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config.input must be an object.")
    velocity_file = _require_nonempty_str(raw, "velocity_file", "input")
    model_index = raw.get("model_index")
    if model_index is not None:
        model_index = _nonnegative_int_value(model_index, "input.model_index")
    return InputConfig(velocity_file=velocity_file, model_index=model_index)


def _parse_inline_velocity(raw: Any) -> InlineVelocityConfig:
    if not isinstance(raw, dict):
        raise ConfigError("config.velocity must be an object.")
    mode = _require_nonempty_str(raw, "mode", "velocity")
    if mode not in SUPPORTED_INLINE_VELOCITY_MODES:
        raise ConfigError(
            f"Unsupported velocity.mode {mode!r}; supported values: "
            f"{sorted(SUPPORTED_INLINE_VELOCITY_MODES)}."
        )
    values = _require_key(raw, "values_m_per_s", "velocity")
    if not isinstance(values, list) or not values:
        raise ConfigError("velocity.values_m_per_s must be a non-empty 2D list.")
    normalized: list[list[float]] = []
    row_length: Optional[int] = None
    for iz, row in enumerate(values):
        if not isinstance(row, list) or not row:
            raise ConfigError(
                f"velocity.values_m_per_s[{iz}] must be a non-empty list."
            )
        if row_length is None:
            row_length = len(row)
        elif len(row) != row_length:
            raise ConfigError("velocity.values_m_per_s rows must have equal length.")
        normalized.append(
            [_positive_float_value(v, f"velocity.values_m_per_s[{iz}][{ix}]") for ix, v in enumerate(row)]
        )
    return InlineVelocityConfig(mode=mode, values_m_per_s=normalized)


def _parse_source(raw: Mapping[str, Any]) -> SourceConfig:
    source_type = _require_nonempty_str(raw, "type", "source")
    if source_type not in SUPPORTED_SOURCE_TYPES:
        raise ConfigError(
            f"Unsupported source.type {source_type!r}; supported values: "
            f"{sorted(SUPPORTED_SOURCE_TYPES)}."
        )
    position = _parse_source_position(_require_mapping(raw, "position", "source"))
    peak_frequency = _require_positive_float(raw, "peak_frequency_hz", "source")
    strength = _require_finite_float(raw, "strength", "source")
    phase_mode = _require_nonempty_str(raw, "phase_mode", "source")
    if phase_mode not in SUPPORTED_SOURCE_PHASE_MODES:
        raise ConfigError(
            f"Unsupported source.phase_mode {phase_mode!r}; supported values: "
            f"{sorted(SUPPORTED_SOURCE_PHASE_MODES)}."
        )
    return SourceConfig(
        type=source_type,
        position=position,
        peak_frequency_hz=peak_frequency,
        strength=strength,
        phase_mode=phase_mode,
    )


def _parse_boundary(
    raw: Mapping[str, Any], boundary_type: str
) -> BoundaryConfig:
    padding = {
        name: _optional_nonnegative_int(raw, name, 0, "boundary")
        for name in (
            "top_padding_cells",
            "bottom_padding_cells",
            "left_padding_cells",
            "right_padding_cells",
        )
    }
    if boundary_type == "zero_exterior_ghost" and any(padding.values()):
        raise ConfigError(
            "boundary.type='zero_exterior_ghost' requires all padding cells to be 0."
        )
    if boundary_type == "forward_compatible_padding":
        invalid_widths = [value for value in padding.values() if value == 1]
        if invalid_widths:
            raise ConfigError(
                "Positive forward-compatible padding widths must be at least 2 "
                "because the reference damping thickness is (width - 1) * spacing."
            )

    damping_raw = raw.get("damping", {})
    if not isinstance(damping_raw, dict):
        raise ConfigError("boundary.damping must be an object.")
    profile = damping_raw.get("profile", "quadratic")
    if profile not in SUPPORTED_DAMPING_PROFILES:
        raise ConfigError(
            f"Unsupported boundary.damping.profile {profile!r}; supported values: "
            f"{sorted(SUPPORTED_DAMPING_PROFILES)}."
        )
    velocity_reference = damping_raw.get("velocity_reference", "minimum")
    if velocity_reference not in SUPPORTED_DAMPING_VELOCITY_REFERENCES:
        raise ConfigError(
            "Unsupported boundary.damping.velocity_reference "
            f"{velocity_reference!r}; supported values: "
            f"{sorted(SUPPORTED_DAMPING_VELOCITY_REFERENCES)}."
        )
    corner_combination = damping_raw.get("corner_combination", "maximum")
    if corner_combination not in SUPPORTED_DAMPING_CORNER_COMBINATIONS:
        raise ConfigError(
            "Unsupported boundary.damping.corner_combination "
            f"{corner_combination!r}; supported values: "
            f"{sorted(SUPPORTED_DAMPING_CORNER_COMBINATIONS)}."
        )
    target_decay = _optional_positive_float(
        damping_raw, "target_decay", 1.0e-7, "boundary.damping"
    )
    if target_decay >= 1.0:
        raise ConfigError("boundary.damping.target_decay must be in (0, 1).")
    damping = DampingProfileConfig(
        profile=profile,
        power=_optional_positive_float(
            damping_raw, "power", 2.0, "boundary.damping"
        ),
        target_decay=target_decay,
        strength_scale=_optional_nonnegative_float(
            damping_raw, "strength_scale", 1.0, "boundary.damping"
        ),
        velocity_reference=velocity_reference,
        corner_combination=corner_combination,
    )
    return BoundaryConfig(type=boundary_type, damping=damping, **padding)


def _parse_receivers(raw: Any) -> ReceiverConfig:
    if raw is None:
        return ReceiverConfig()
    if not isinstance(raw, dict):
        raise ConfigError("config.receivers must be an object.")
    positions = raw.get("positions", [])
    if not isinstance(positions, list):
        raise ConfigError("receivers.positions must be a list.")
    parsed: list[SourcePositionConfig] = []
    for index, position in enumerate(positions):
        if not isinstance(position, dict):
            raise ConfigError(f"receivers.positions[{index}] must be an object.")
        parsed.append(
            _parse_grid_position(position, f"receivers.positions[{index}]")
        )
    return ReceiverConfig(positions=tuple(parsed))


def _parse_source_position(raw: Mapping[str, Any]) -> SourcePositionConfig:
    return _parse_grid_position(raw, "source.position")


def _parse_grid_position(
    raw: Mapping[str, Any], context: str
) -> SourcePositionConfig:
    mode = _require_nonempty_str(raw, "mode", context)
    if mode not in SUPPORTED_SOURCE_POSITION_MODES:
        raise ConfigError(
            f"Unsupported {context}.mode {mode!r}; supported values: "
            f"{sorted(SUPPORTED_SOURCE_POSITION_MODES)}."
        )
    if mode == "fractional":
        x_fraction = _fraction_value(
            _require_key(raw, "x_fraction", context),
            f"{context}.x_fraction",
        )
        z_fraction = _fraction_value(
            _require_key(raw, "z_fraction", context),
            f"{context}.z_fraction",
        )
        return SourcePositionConfig(
            mode=mode, x_fraction=x_fraction, z_fraction=z_fraction
        )
    ix = _nonnegative_int_value(
        _require_key(raw, "ix", context), f"{context}.ix"
    )
    iz = _nonnegative_int_value(
        _require_key(raw, "iz", context), f"{context}.iz"
    )
    return SourcePositionConfig(mode=mode, ix=ix, iz=iz)


def _parse_output(raw: Mapping[str, Any]) -> OutputConfig:
    directory = _require_nonempty_str(raw, "directory", "output")
    export_npz = _optional_bool(raw, "export_npz", True, "output")
    export_mtx = _optional_bool(raw, "export_mtx", True, "output")
    save_velocity_preview = _optional_bool(
        raw, "save_velocity_preview", True, "output"
    )
    return OutputConfig(
        directory=directory,
        export_npz=export_npz,
        export_mtx=export_mtx,
        save_velocity_preview=save_velocity_preview,
    )


def _require_key(raw: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in raw:
        raise ConfigError(f"Missing required key: {context}.{key}.")
    return raw[key]


def _require_mapping(raw: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = _require_key(raw, key, context)
    if not isinstance(value, dict):
        raise ConfigError(f"{context}.{key} must be an object.")
    return value


def _require_nonempty_str(raw: Mapping[str, Any], key: str, context: str) -> str:
    value = _require_key(raw, key, context)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{context}.{key} must be a non-empty string.")
    return value


def _require_positive_float(raw: Mapping[str, Any], key: str, context: str) -> float:
    return _positive_float_value(_require_key(raw, key, context), f"{context}.{key}")


def _require_finite_float(raw: Mapping[str, Any], key: str, context: str) -> float:
    return _finite_float_value(_require_key(raw, key, context), f"{context}.{key}")


def _finite_float_value(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite.")
    return result


def _positive_float_value(value: Any, name: str) -> float:
    result = _finite_float_value(value, name)
    if result <= 0.0:
        raise ConfigError(f"{name} must be > 0.")
    return result


def _fraction_value(value: Any, name: str) -> float:
    result = _finite_float_value(value, name)
    if not 0.0 <= result <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1].")
    return result


def _nonnegative_int_value(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer >= 0.")
    if value < 0:
        raise ConfigError(f"{name} must be >= 0.")
    return value


def _optional_bool(
    raw: Mapping[str, Any], key: str, default: bool, context: str
) -> bool:
    value = raw.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{context}.{key} must be a boolean.")
    return value


def _optional_nonnegative_int(
    raw: Mapping[str, Any], key: str, default: int, context: str
) -> int:
    return _nonnegative_int_value(raw.get(key, default), f"{context}.{key}")


def _optional_positive_float(
    raw: Mapping[str, Any], key: str, default: float, context: str
) -> float:
    return _positive_float_value(raw.get(key, default), f"{context}.{key}")


def _optional_nonnegative_float(
    raw: Mapping[str, Any], key: str, default: float, context: str
) -> float:
    value = _finite_float_value(raw.get(key, default), f"{context}.{key}")
    if value < 0.0:
        raise ConfigError(f"{context}.{key} must be >= 0.")
    return value
