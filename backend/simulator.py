"""
Finite-difference 2D wave simulator used by the live workspace.
"""
from __future__ import annotations

import math
from typing import Dict

import numpy as np


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class WaveSim2D:
    def __init__(self, resolution: int = 96) -> None:
        self.params: Dict[str, object] = {
            "scenario": "b",
            "speed": 1.0,
            "amplitude": 1.0,
            "omega": 11.0,
            "source_strength": 1.0,
            "damping": 0.02,
            "time_scale": 0.9,
            "resolution": resolution,
            "boundary": "absorbing",
            "visualization": "ripple",
            "separation": 0.38,
            "gap_width": 0.25,
            "c_upper": 1.0,
            "c_lower": 0.55,
            "brush_radius": 0.08,
            "medium_density": 1.45,
            "grid_snap": False,
            "preset": "none",
        }
        self.time = 0.0
        self.frame = 0
        self.active_emitters: Dict[str, Dict[str, float]] = {}
        self.static_sources: Dict[str, Dict[str, float]] = {}
        self.resize(int(resolution))
        self.load_scenario(str(self.params["scenario"]))

    def resize(self, resolution: int) -> None:
        resolution = int(_clamp(resolution, 48, 192))
        self.resolution = resolution
        self.dx = 2.0 / max(resolution - 1, 1)
        shape = (resolution, resolution)
        self.u_prev = np.zeros(shape, dtype=np.float32)
        self.u_curr = np.zeros(shape, dtype=np.float32)
        self.obstacle_mask = np.zeros(shape, dtype=bool)
        self.medium_speed = np.ones(shape, dtype=np.float32)
        self.extra_damping = np.zeros(shape, dtype=np.float32)
        self.source_id_mask = np.zeros(shape, dtype=np.int32)
        self._source_index: Dict[int, str] = {}
        coords = np.linspace(-1.0, 1.0, resolution, dtype=np.float32)
        self.X, self.Y = np.meshgrid(coords, coords)
        self.edge_absorb = self._make_edge_absorb()
        self.edge_infinite = self._make_infinite_sponge()
        self._rebuild_runtime_maps()

    def _make_edge_absorb(self) -> np.ndarray:
        margin = max(3, self.resolution // 14)
        absorb = np.ones((self.resolution, self.resolution), dtype=np.float32)
        for i in range(margin):
            f = 1.0 - ((margin - i) / margin) ** 2 * 0.16
            absorb[i, :] *= f
            absorb[-1 - i, :] *= f
            absorb[:, i] *= f
            absorb[:, -1 - i] *= f
        return absorb

    def _make_infinite_sponge(self) -> np.ndarray:
        margin = max(8, self.resolution // 8)
        sponge = np.ones((self.resolution, self.resolution), dtype=np.float32)
        for i in range(margin):
            t = (margin - i) / margin
            f = 1.0 - (t ** 3) * 0.06
            sponge[i, :] *= f
            sponge[-1 - i, :] *= f
            sponge[:, i] *= f
            sponge[:, -1 - i] *= f
        return sponge

    def sync_params(self, update: Dict[str, object]) -> None:
        if "resolution" in update and int(update["resolution"]) != self.resolution:
            self.resize(int(update["resolution"]))
            self.load_scenario(str(update.get("scenario", self.params["scenario"])))

        prev_scenario = str(self.params.get("scenario", "b"))
        self.params.update(update)
        scenario = str(self.params.get("scenario", "b"))
        if scenario != prev_scenario:
            self.load_scenario(scenario)
        else:
            self._rebuild_runtime_maps()

    def reset_fields(self) -> None:
        self.u_prev.fill(0.0)
        self.u_curr.fill(0.0)
        self.time = 0.0
        self.frame = 0

    def load_scenario(self, scenario: str) -> None:
        self.params["scenario"] = scenario
        self.reset_fields()
        self.active_emitters.clear()
        self.static_sources.clear()
        self.obstacle_mask.fill(False)
        self.medium_speed.fill(1.0)
        self.extra_damping.fill(0.0)
        self.source_id_mask.fill(0)
        self._source_index.clear()

        if scenario == "a":
            self.params["boundary"] = "absorbing"
            self.active_emitters["primary"] = {
                "x": float(self.params.get("source_x", 0.0)),
                "y": float(self.params.get("source_y", 0.0)),
                "radius": 0.07,
                "strength": 1.2,
            }
        elif scenario == "b":
            sep = float(self.params.get("separation", 0.38))
            self.params["boundary"] = "absorbing"
            self.active_emitters["src_a"] = {"x": 0.0, "y": sep, "radius": 0.055, "strength": 1.0}
            self.active_emitters["src_b"] = {"x": 0.0, "y": -sep, "radius": 0.055, "strength": 1.0}
        elif scenario == "c":
            self.params["boundary"] = "fixed"
            self.active_emitters["resonance"] = {"x": 0.0, "y": -0.25, "radius": 0.06, "strength": 1.15}
        elif scenario == "d":
            self.params["boundary"] = "absorbing"
            self.active_emitters["left_beam"] = {"x": -0.72, "y": 0.0, "radius": 0.055, "strength": 1.1}
            gap = float(self.params.get("gap_width", 0.25))
            self._paint_barrier_line(0.0, -1.0, 0.0, -gap / 2.0, 0.022)
            self._paint_barrier_line(0.0, gap / 2.0, 0.0, 1.0, 0.022)
        elif scenario == "e":
            self.params["boundary"] = "absorbing"
            upper = float(self.params.get("c_upper", 1.0))
            lower = float(self.params.get("c_lower", 0.55))
            self.medium_speed[self.Y >= 0.0] = upper
            self.medium_speed[self.Y < 0.0] = lower
            self.active_emitters["seismic"] = {"x": 0.0, "y": -0.55, "radius": 0.075, "strength": 1.35}

        self._apply_preset(str(self.params.get("preset", "none")))
        self._rebuild_runtime_maps()

    def _rebuild_runtime_maps(self) -> None:
        base_speed = float(self.params.get("speed", 1.0))
        self.c_map = np.maximum(0.05, base_speed * self.medium_speed).astype(np.float32)
        scene_damping = float(self.params.get("damping", 0.02))
        self.damping_map = (scene_damping + self.extra_damping).astype(np.float32)
        self.max_c = float(np.max(self.c_map))
        self.dt_cfl = 0.45 * self.dx / max(self.max_c, 1e-6)
        time_scale = float(self.params.get("time_scale", 0.9))
        self.dt = min(self.dt_cfl * max(0.05, time_scale), self.dt_cfl)

    def step(self, steps: int = 1) -> None:
        boundary = str(self.params.get("boundary", "absorbing"))
        for _ in range(max(1, steps)):
            self._apply_dynamic_preset()
            lap = self._laplacian(self.u_curr, boundary)
            coeff = (self.c_map * self.dt / self.dx) ** 2
            damp = self.damping_map * self.dt
            u_next = (2.0 - damp) * self.u_curr - (1.0 - damp) * self.u_prev + coeff * lap
            self._inject_emitters(u_next)
            self._enforce_obstacles(u_next)
            self._apply_boundary(u_next, boundary)
            self.u_prev, self.u_curr = self.u_curr, u_next.astype(np.float32, copy=False)
            self.time += self.dt
            self.frame += 1

    def _laplacian(self, u: np.ndarray, boundary: str) -> np.ndarray:
        if boundary == "periodic":
            return (
                np.roll(u, 1, axis=0)
                + np.roll(u, -1, axis=0)
                + np.roll(u, 1, axis=1)
                + np.roll(u, -1, axis=1)
                - 4.0 * u
            )
        pad_mode = "edge" if boundary == "reflective" else "constant"
        padded = np.pad(u, 1, mode=pad_mode)
        return (
            padded[1:-1, :-2]
            + padded[1:-1, 2:]
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            - 4.0 * padded[1:-1, 1:-1]
        )

    def _apply_boundary(self, u: np.ndarray, boundary: str) -> None:
        if boundary == "fixed":
            u[0, :] = 0.0
            u[-1, :] = 0.0
            u[:, 0] = 0.0
            u[:, -1] = 0.0
        elif boundary == "reflective":
            u[0, :] = u[1, :]
            u[-1, :] = u[-2, :]
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
        elif boundary == "absorbing":
            u *= self.edge_absorb
            u[0, :] *= 0.2
            u[-1, :] *= 0.2
            u[:, 0] *= 0.2
            u[:, -1] *= 0.2
        elif boundary == "infinite":
            u *= self.edge_infinite

    def _inject_emitters(self, u_next: np.ndarray) -> None:
        omega = float(self.params.get("omega", 11.0))
        amplitude = float(self.params.get("amplitude", 1.0))
        source_strength = float(self.params.get("source_strength", 1.0))
        for emitter in list(self.active_emitters.values()) + list(self.static_sources.values()):
            radius = float(emitter.get("radius", 0.06))
            strength = float(emitter.get("strength", 1.0))
            x0 = float(emitter.get("x", 0.0))
            y0 = float(emitter.get("y", 0.0))
            gaussian = np.exp(-((self.X - x0) ** 2 + (self.Y - y0) ** 2) / max(radius * radius, 1e-5))
            phase = float(emitter.get("phase", 0.0))
            signal = math.sin(omega * self.time + phase)
            u_next += gaussian * signal * amplitude * source_strength * strength * self.dt * 7.5

    def _enforce_obstacles(self, u: np.ndarray) -> None:
        u[self.obstacle_mask] = 0.0

    def add_impulse(self, x: float, y: float, strength: float = 1.0, radius: float = 0.08) -> None:
        gaussian = np.exp(-((self.X - x) ** 2 + (self.Y - y) ** 2) / max(radius * radius, 1e-5))
        impulse = gaussian * float(strength)
        self.u_curr += impulse.astype(np.float32)
        self.u_prev += impulse.astype(np.float32)
        self._enforce_obstacles(self.u_curr)
        self._enforce_obstacles(self.u_prev)

    def set_pointer_emitter(self, emitter_id: str, x: float, y: float, active: bool, radius: float, strength: float) -> None:
        if not active:
            self.active_emitters.pop(emitter_id, None)
            return
        self.active_emitters[emitter_id] = {"x": x, "y": y, "radius": radius, "strength": strength}

    def apply_action(self, action: Dict[str, object]) -> None:
        kind = str(action.get("kind", ""))
        if kind == "impulse":
            self.add_impulse(
                float(action.get("x", 0.0)),
                float(action.get("y", 0.0)),
                float(action.get("strength", 1.0)),
                float(action.get("radius", 0.08)),
            )
        elif kind == "emitter":
            self.set_pointer_emitter(
                str(action.get("id", "pointer")),
                float(action.get("x", 0.0)),
                float(action.get("y", 0.0)),
                bool(action.get("active", False)),
                float(action.get("radius", 0.07)),
                float(action.get("strength", 1.0)),
            )
        elif kind == "paint":
            self._apply_paint_action(action)
        elif kind == "preset":
            self.params["preset"] = str(action.get("preset", "none"))
            self.load_scenario(str(self.params.get("scenario", "b")))
        elif kind == "reset_scene":
            self.load_scenario(str(self.params.get("scenario", "b")))
        elif kind == "load_scene":
            self.load_scenario(str(action.get("scenario", self.params.get("scenario", "b"))))
            scene_params = action.get("params", {})
            if isinstance(scene_params, dict):
                self.sync_params(scene_params)
            for scene_action in action.get("actions", []):
                if isinstance(scene_action, dict):
                    self.apply_action(scene_action)

    def _apply_paint_action(self, action: Dict[str, object]) -> None:
        tool = str(action.get("tool", "barrier"))
        x0 = float(action.get("x0", action.get("x", 0.0)))
        y0 = float(action.get("y0", action.get("y", 0.0)))
        x1 = float(action.get("x1", x0))
        y1 = float(action.get("y1", y0))
        radius = float(action.get("radius", 0.06))
        if tool == "barrier":
            self._paint_barrier_line(x0, y0, x1, y1, radius)
        elif tool == "medium":
            density = float(action.get("density", self.params.get("medium_density", 1.45)))
            self._paint_medium_line(x0, y0, x1, y1, radius, density)
        elif tool == "erase":
            self._erase_line(x0, y0, x1, y1, radius)
        elif tool == "source":
            self._stamp_source(x1, y1, radius, float(action.get("strength", 1.0)))
        self._rebuild_runtime_maps()

    def _line_points(self, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
        dist = max(abs(x1 - x0), abs(y1 - y0))
        n = max(1, int(dist / 0.02))
        ts = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)
        xs = x0 + (x1 - x0) * ts
        ys = y0 + (y1 - y0) * ts
        return np.stack([xs, ys], axis=1)

    def _brush_mask(self, x: float, y: float, radius: float) -> np.ndarray:
        return (self.X - x) ** 2 + (self.Y - y) ** 2 <= radius * radius

    def _paint_barrier_line(self, x0: float, y0: float, x1: float, y1: float, radius: float) -> None:
        for x, y in self._line_points(x0, y0, x1, y1):
            self.obstacle_mask |= self._brush_mask(float(x), float(y), radius)
        self._enforce_obstacles(self.u_curr)
        self._enforce_obstacles(self.u_prev)

    def _paint_medium_line(self, x0: float, y0: float, x1: float, y1: float, radius: float, density: float) -> None:
        speed_scale = 1.0 / max(0.35, density)
        for x, y in self._line_points(x0, y0, x1, y1):
            mask = self._brush_mask(float(x), float(y), radius)
            self.medium_speed[mask] = speed_scale
            self.extra_damping[mask] = np.maximum(self.extra_damping[mask], 0.01)

    def _erase_line(self, x0: float, y0: float, x1: float, y1: float, radius: float) -> None:
        for x, y in self._line_points(x0, y0, x1, y1):
            mask = self._brush_mask(float(x), float(y), radius)
            self.obstacle_mask[mask] = False
            self.medium_speed[mask] = 1.0
            self.extra_damping[mask] = 0.0
            self.source_id_mask[mask] = 0
        live_ids = {int(v) for v in np.unique(self.source_id_mask) if int(v) > 0}
        self.static_sources = {
            sid: data
            for idx, sid in self._source_index.items()
            if idx in live_ids
            for data in [self.static_sources.get(sid)]
            if data is not None
        }
        self._source_index = {idx: sid for idx, sid in self._source_index.items() if idx in live_ids}

    def _stamp_source(self, x: float, y: float, radius: float, strength: float) -> None:
        source_name = f"src_{len(self.static_sources) + 1}"
        self.static_sources[source_name] = {"x": x, "y": y, "radius": radius, "strength": strength}
        source_idx = max(self._source_index.keys(), default=0) + 1
        self._source_index[source_idx] = source_name
        self.source_id_mask[self._brush_mask(x, y, radius)] = source_idx

    def _apply_preset(self, preset: str) -> None:
        if preset == "stone_drop":
            self.add_impulse(0.0, 0.0, strength=1.4, radius=0.09)
        elif preset == "sonic_wave_beam":
            self.active_emitters["beam_a"] = {"x": -0.82, "y": -0.18, "radius": 0.04, "strength": 0.8}
            self.active_emitters["beam_b"] = {"x": -0.82, "y": 0.18, "radius": 0.04, "strength": 0.8}
        elif preset == "wave_maze":
            self._paint_barrier_line(-0.4, -1.0, -0.4, 0.25, 0.02)
            self._paint_barrier_line(0.0, -0.2, 0.0, 1.0, 0.02)
            self._paint_barrier_line(0.42, -1.0, 0.42, 0.4, 0.02)
            self._erase_line(-0.4, -0.05, -0.4, 0.05, 0.05)
            self._erase_line(0.0, 0.42, 0.0, 0.58, 0.05)
            self._erase_line(0.42, 0.0, 0.42, 0.15, 0.05)
        elif preset == "resonance_chamber":
            self.params["boundary"] = "fixed"
            self.active_emitters["chamber"] = {"x": 0.0, "y": -0.3, "radius": 0.06, "strength": 1.3}
        elif preset == "echo_room":
            self.params["boundary"] = "reflective"
            self._paint_barrier_line(-0.1, -0.4, 0.25, 0.25, 0.025)

    def _apply_dynamic_preset(self) -> None:
        preset = str(self.params.get("preset", "none"))
        if preset == "random_rain" and self.frame % 3 == 0:
            x = float(np.random.uniform(-0.95, 0.95))
            y = float(np.random.uniform(-0.95, 0.95))
            self.add_impulse(x, y, strength=0.45, radius=0.04)
        elif preset == "earthquake_burst" and self.frame < 90 and self.frame % 4 == 0:
            x = float(np.random.uniform(-0.25, 0.25))
            self.add_impulse(x, -0.88, strength=0.85, radius=0.05)

    def metrics(self) -> Dict[str, float]:
        vel = (self.u_curr - self.u_prev) / max(self.dt, 1e-6)
        grad_x = np.gradient(self.u_curr, axis=1) / max(self.dx, 1e-6)
        grad_y = np.gradient(self.u_curr, axis=0) / max(self.dx, 1e-6)
        energy = 0.5 * (vel ** 2 + self.c_map ** 2 * (grad_x ** 2 + grad_y ** 2))
        return {
            "energy_mean": float(np.mean(energy)),
            "energy_max": float(np.max(energy)),
            "dt": float(self.dt),
            "dt_cfl": float(self.dt_cfl),
            "cfl_ratio": float(self.dt / max(self.dt_cfl, 1e-6)),
        }

    def current_grid(self) -> np.ndarray:
        return self.u_curr
