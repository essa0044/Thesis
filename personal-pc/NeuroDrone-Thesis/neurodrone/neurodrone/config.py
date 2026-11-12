"""Every tunable lives here. Defaults are tuned on the synthetic simulator ONLY -
re-tune on recorded flights from your own sensor before trusting them (see README)."""
from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from typing import Optional, Tuple


@dataclass
class SpikingConfig:
    """Settings of the fully spiking detector (snn_spiking.SpikingLoomingSNN), used when
    snn.impl = "spiking". The geometry and the layer-1 time window (deltas_px, dt_min_us,
    dt_max_us, r_min_frac, roi_y, band_edges_frac) are shared with the hybrid detector.
    Tuned on the synthetic simulator: false alarms start at theta0 ~3-4 and beta ~0.04-0.08,
    so the defaults keep a ~1.5-2x margin. Re-tune on real recordings (scripts/replay_snn.py)."""
    step_us: int = 1000  # clock step of layers 2-3
    # layer 2: region expansion (E) neurons; inputs are counts of layer-1 spikes
    tau_m_s: float = 0.05  # membrane leak
    kappa: float = 1.0  # weight of the inhibitory IN spikes relative to the excitatory OUT spikes
    theta0: float = 6.0  # base threshold, in (net) layer-1 spikes
    beta: float = 0.12  # adaptive threshold: + beta per unit of recent region activity
    tau_a_s: float = 0.05  # time constant of that activity trace
    v_floor: float = -1.0  # membrane floor, in multiples of theta0
    # layer 3: band alarm neurons
    coinc_us: int = 20_000  # left- and right-half expansion spikes this close together = coincidence
    # readout
    tau_rate_s: float = 0.10
    rate_alarm: float = 15.0  # needs >= 2 alarm spikes within ~70 ms (one spike adds 1/tau_rate = 10)
    adv_rate_full: float = 100.0  # advisory E-neuron spike rate that counts as urgency 1.0


@dataclass
class SNNConfig:
    impl: str = "spiking"  # "spiking" = snn_spiking.SpikingLoomingSNN (default) | "hybrid" = snn.LoomingSNN (baseline)
    # Elementary-motion-detector baselines in pixels. None -> derived from sensor width.
    deltas_px: Optional[Tuple[int, ...]] = None
    dt_min_us: int = 2_000  # ignore pixel pairs that fire almost simultaneously (edge bursts)
    dt_max_us: int = 150_000  # slowest motion detected: delta / dt_max px/s
    kappa: float = 1.0  # weight of inward correlations (1.0 = pure divergence)
    z_eps: float = 4.0  # regulariser for the Poisson z-score (damps tiny counts)
    r_min_frac: float = 0.05  # ignore pixels closer than this (x width) to the focus of expansion
    roi_y: Tuple[float, float] = (0.10, 0.90)  # rows used (exclude ground texture if you can)
    # Mirrored-pair coincidence neurons: |x - centre| bands (fractions of width). One neuron per band
    # fires only if BOTH the left and right halves of that band show outward expansion.
    band_edges_frac: Tuple[float, ...] = (0.0, 0.14, 0.28, 0.5)
    # Leaky integrate-and-fire neurons (one per sector: left / centre / right)
    tau_m_s: float = 0.05
    w_syn: float = 1.0
    theta: float = 4.0  # firing threshold, z-score units (false-alarm cliff in sim ~2; keeps ~2x margin)
    v_floor: float = -2.0  # membrane floor, in multiples of theta
    tau_rate_s: float = 0.10  # spike-rate trace used for urgency / alarm
    rate_alarm: float = 15.0  # needs >=2 spikes within ~70 ms (one spike contributes 1/tau_rate = 10/s)
    # The hybrid uses the fields above; the fully spiking detector uses the shared geometry/time-window
    # fields above plus these:
    spiking: SpikingConfig = field(default_factory=SpikingConfig)


@dataclass
class CoordinatorConfig:
    # Mission following
    cruise_speed: float = 1.0  # m/s
    max_speed: float = 1.5  # hard clamp on any commanded speed
    accel_limit: float = 1.0  # m/s^2, limits speeding up (slowing down is immediate)
    yaw_gain: float = 0.8  # (deg/s) per degree of bearing error
    max_yaw_rate: float = 40.0
    yaw_gate_deg_s: float = 10.0  # above this yaw rate: turn in place, do not translate
    # Advisory side-push away from the busier side. OFF by default: in closed-loop tests it pushed the
    # drone off the target line (the target looks like an obstacle up close) and is biased by rotation.
    lateral_gain: float = 0.0  # m/s of sideways push at full left/right urgency imbalance
    lateral_gate_yaw: float = 6.0  # push disabled above this measured/commanded rate (rotation flow)
    # Slowing on SNN urgency is OFF by default: looming flow is proportional to speed, so slowing down
    # starves the very signal that triggers the alarm (observed: alarm at 1.3 m instead of 5 m).
    urgency_slowdown: bool = False
    slow_start: float = 0.3
    slow_end: float = 0.8
    search_yaw_rate: float = 10.0
    search_max_deg: float = 360.0
    arrive_bearing_deg: float = 12.0
    # Target box must span >= this fraction of the image width for an alarm to count as "the target".
    # Was 0.20; lowered to 0.15 because the spiking detector alarms earlier (target still smaller):
    # at 0.20 it made one unnecessary sidestep in front of its own target (bearing_offset scenario).
    # The hybrid detector gives identical closed-loop results at 0.15 and 0.20.
    arrive_w_frac: float = 0.15
    # Staleness / liveness
    snn_timeout_s: float = 0.25  # reflex layer silent for longer than this -> never move
    goal_ttl_s: float = 8.0  # a VLM goal older than this is ignored
    blind_timeout_s: float = 3.0  # reflex layer lost this long -> failsafe action
    failsafe_action: str = "land"  # "land" | "hold"
    # Alarm handling: the reflex brakes, then the VLM adjudicates (target vs obstacle) on a FRESH image
    adjudicate_timeout_s: float = 10.0  # no fresh VLM goal by then -> assume obstacle and dodge
    adj_arm_s: float = 1.0  # after this long stopped, an alarm means something is coming at us -> dodge now
    # Range-sensor reflex (depth camera / ToF). If enabled it is REQUIRED: stale -> vehicle holds.
    use_clearance: bool = False
    clearance_timeout_s: float = 0.5
    clearance_margin_m: float = 0.8  # free space to keep ahead once stopped
    brake_decel: float = 2.0  # m/s^2 we assume the autopilot can brake at (be conservative)
    reaction_s: float = 0.4  # sensing -> autopilot response latency allowance
    # Reactive dodge (open loop, because the flow cue vanishes once we stop)
    dodge_speed: float = 1.0
    dodge_distance_m: float = 2.0
    settle_s: float = 0.5
    episode_window_s: float = 25.0
    max_episodes: int = 4  # more dodges than this inside the window -> give up and hold
    escalation: float = 0.5  # each repeat dodge is this much longer (x base)
    # Vehicle-level guards
    min_battery: float = 0.25
    max_dist_home_m: float = 40.0
    max_alt_m: float = 30.0


@dataclass
class VLMConfig:
    base_url: str = "http://127.0.0.1:8080/v1"  # any OpenAI-compatible server
    model: str = "qwen2.5-vl-3b"
    api_key: str = "none"
    period_s: float = 2.0  # start-to-start; 0 = run back-to-back ("continuous")
    timeout_s: float = 20.0
    max_tokens: int = 80
    image_max_side: int = 448
    jpeg_quality: int = 85
    lost_patience: int = 2  # consecutive "not found" answers before switching to search
    coords: str = "auto"  # box convention: auto | norm1000 | pixels | unit (Qwen VLMs in Ollama: norm1000)


@dataclass
class CameraConfig:
    device: str = "0"  # index, video path, or GStreamer pipeline string
    width: int = 640
    height: int = 480
    hfov_deg: float = 69.0  # horizontal FOV of the RGB camera (RealSense D435 RGB ~ 69 deg)


@dataclass
class RuntimeConfig:
    control_hz: float = 25.0
    log_path: Optional[str] = None


@dataclass
class Config:
    snn: SNNConfig = field(default_factory=SNNConfig)
    coordinator: CoordinatorConfig = field(default_factory=CoordinatorConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    camera: CameraConfig = field(default_factory=CameraConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _apply(obj, d: dict) -> None:
    for k, v in d.items():
        if not hasattr(obj, k):
            raise KeyError(f"unknown config key: {k!r} (in {type(obj).__name__})")
        cur = getattr(obj, k)
        if is_dataclass(cur):
            _apply(cur, v or {})
        else:
            setattr(obj, k, tuple(v) if isinstance(v, list) else v)


def load_config(path: Optional[str] = None) -> Config:
    """Defaults, optionally overridden by a YAML file (needs pyyaml)."""
    cfg = Config()
    if path:
        import yaml  # imported lazily so the core has no hard dependency on it

        with open(path, "r") as f:
            _apply(cfg, yaml.safe_load(f) or {})
    return cfg
