from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from simulator import WaveSim2D

try:
    from inference import load_models, query_pinn

    load_models()
    PINN_READY = True
except Exception:
    PINN_READY = False


router = APIRouter(prefix="/api/earthquake", tags=["earthquake"])


PRESETS: Dict[str, Dict[str, object]] = {
    "nepal": {
        "id": "nepal",
        "label": "Nepal Earthquake Scenario",
        "magnitude": 7.8,
        "depth_km": 15.0,
        "epicenter_x": 0.34,
        "epicenter_y": 0.63,
        "medium_type": "basin",
        "wave_speed": 1.05,
        "duration": 18.0,
        "resolution": 60,
        "map_scale_km": 28.0,
    },
    "urban": {
        "id": "urban",
        "label": "Urban City Scenario",
        "magnitude": 6.6,
        "depth_km": 11.0,
        "epicenter_x": 0.48,
        "epicenter_y": 0.56,
        "medium_type": "urban",
        "wave_speed": 1.0,
        "duration": 14.0,
        "resolution": 58,
        "map_scale_km": 20.0,
    },
    "disaster": {
        "id": "disaster",
        "label": "High Magnitude Disaster",
        "magnitude": 8.9,
        "depth_km": 21.0,
        "epicenter_x": 0.28,
        "epicenter_y": 0.68,
        "medium_type": "mountain",
        "wave_speed": 1.18,
        "duration": 22.0,
        "resolution": 64,
        "map_scale_km": 35.0,
    },
}


class EarthquakeRequest(BaseModel):
    preset: str = Field(default="nepal")
    magnitude: float = Field(default=7.8, ge=1.0, le=10.0)
    depth_km: float = Field(default=15.0, ge=1.0, le=80.0)
    epicenter_x: float = Field(default=0.34, ge=0.02, le=0.98)
    epicenter_y: float = Field(default=0.63, ge=0.02, le=0.98)
    medium_type: str = Field(default="basin")
    wave_speed: float = Field(default=1.05, ge=0.5, le=2.0)
    duration: float = Field(default=18.0, ge=4.0, le=40.0)
    resolution: int = Field(default=60, ge=40, le=84)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _linspace_grid(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    coords = np.linspace(0.0, 1.0, resolution, dtype=np.float32)
    return np.meshgrid(coords, coords)


def _terrain_maps(resolution: int, medium_type: str, preset: str) -> tuple[np.ndarray, np.ndarray]:
    xg, yg = _linspace_grid(resolution)
    speed = np.ones((resolution, resolution), dtype=np.float32)
    amp = np.ones((resolution, resolution), dtype=np.float32)

    basin = np.exp(-(((xg - 0.55) ** 2) / 0.065 + ((yg - 0.52) ** 2) / 0.028)).astype(np.float32)
    urban_core = np.exp(-(((xg - 0.5) ** 2) / 0.09 + ((yg - 0.48) ** 2) / 0.05)).astype(np.float32)
    ridge = np.exp(-(((xg - 0.22) ** 2) / 0.03 + ((yg - 0.34) ** 2) / 0.12)).astype(np.float32)

    medium = medium_type.lower()
    if medium == "rock":
        speed *= 1.18
        amp *= 0.88
    elif medium == "soft_soil":
        speed *= 0.82
        amp *= 1.22
    elif medium == "mountain":
        speed *= 1.1 - ridge * 0.18
        amp *= 0.95 + ridge * 0.12
    elif medium == "urban":
        speed *= 0.92 - urban_core * 0.1
        amp *= 1.05 + urban_core * 0.42
    else:
        speed *= 0.96 - basin * 0.25
        amp *= 1.04 + basin * 0.55

    if preset == "nepal":
        speed *= 0.98 - basin * 0.08
        amp *= 1.02 + basin * 0.18
    elif preset == "disaster":
        amp *= 1.08

    return np.clip(speed, 0.5, 1.3), np.clip(amp, 0.8, 1.85)


def _building_catalog(preset: str) -> List[Dict[str, object]]:
    if preset == "urban":
        return [
            {"id": "b1", "name": "Hospital Tower", "x": 0.43, "y": 0.36, "height": 10, "material": "strong"},
            {"id": "b2", "name": "Market Block", "x": 0.58, "y": 0.41, "height": 7, "material": "weak"},
            {"id": "b3", "name": "School", "x": 0.66, "y": 0.56, "height": 5, "material": "weak"},
            {"id": "b4", "name": "Apartment Row", "x": 0.38, "y": 0.58, "height": 8, "material": "strong"},
        ]
    if preset == "disaster":
        return [
            {"id": "b1", "name": "Bridge Control", "x": 0.62, "y": 0.28, "height": 4, "material": "strong"},
            {"id": "b2", "name": "Slope Village", "x": 0.51, "y": 0.52, "height": 4, "material": "weak"},
            {"id": "b3", "name": "Relief Depot", "x": 0.73, "y": 0.61, "height": 6, "material": "strong"},
            {"id": "b4", "name": "Old Settlement", "x": 0.34, "y": 0.63, "height": 5, "material": "weak"},
        ]
    return [
        {"id": "b1", "name": "Heritage Block", "x": 0.48, "y": 0.38, "height": 6, "material": "weak"},
        {"id": "b2", "name": "Clinic", "x": 0.61, "y": 0.45, "height": 4, "material": "strong"},
        {"id": "b3", "name": "School", "x": 0.68, "y": 0.58, "height": 5, "material": "weak"},
        {"id": "b4", "name": "Apartment Cluster", "x": 0.39, "y": 0.57, "height": 8, "material": "strong"},
    ]


def _normalized_coords(x: float, y: float) -> tuple[float, float]:
    return _clamp(x, 0.02, 0.98) * 2.0 - 1.0, _clamp(y, 0.02, 0.98) * 2.0 - 1.0


def _status_for_damage(damage: float) -> str:
    if damage >= 0.78:
        return "collapse"
    if damage >= 0.46:
        return "risky"
    return "safe"


def _material_factor(material: str) -> float:
    return 1.25 if material == "weak" else 0.82


def _configure_sim(resolution: int, speed_scale: np.ndarray, wave_speed: float, damping: float) -> WaveSim2D:
    sim = WaveSim2D(resolution)
    sim.active_emitters.clear()
    sim.static_sources.clear()
    sim.obstacle_mask.fill(False)
    sim.source_id_mask.fill(0)
    sim.params["boundary"] = "absorbing"
    sim.params["damping"] = damping
    sim.params["speed"] = wave_speed
    sim.medium_speed[:] = speed_scale
    sim.extra_damping.fill(0.0)
    sim._rebuild_runtime_maps()
    sim.reset_fields()
    return sim


def _sample_grid(grid: np.ndarray, x: float, y: float) -> float:
    ix = int(_clamp(round(x * (grid.shape[1] - 1)), 0, grid.shape[1] - 1))
    iy = int(_clamp(round(y * (grid.shape[0] - 1)), 0, grid.shape[0] - 1))
    return float(grid[iy, ix])


def _front_radius(grid: np.ndarray, epicenter: tuple[float, float], threshold: float) -> float:
    mask = np.abs(grid) > threshold
    if not np.any(mask):
        return 0.0
    res = grid.shape[0]
    xg, yg = _linspace_grid(res)
    dist = np.sqrt((xg - epicenter[0]) ** 2 + (yg - epicenter[1]) ** 2)
    return float(np.max(dist[mask]))


def _ai_damage_prediction(
    request: EarthquakeRequest,
    amp_map: np.ndarray,
    max_heat: np.ndarray,
) -> tuple[np.ndarray, str, str]:
    res = max_heat.shape[0]
    xg, yg = _linspace_grid(res)
    dist = np.sqrt((xg - request.epicenter_x) ** 2 + (yg - request.epicenter_y) ** 2)
    mag_term = (request.magnitude / 10.0) ** 1.55
    depth_term = 1.0 / (1.0 + request.depth_km / 18.0)
    surrogate = mag_term * depth_term * amp_map * np.exp(-dist * 4.8)
    surrogate *= 1.0 + np.clip(max_heat, 0.0, 1.0) * 0.25

    mode = "heuristic-surrogate"
    if PINN_READY:
        try:
            pinn = query_pinn(
                "e",
                {"c_upper": 1.0, "c_lower": float(np.min(amp_map) / max(np.max(amp_map), 1e-6))},
                t=max(0.8, request.duration * 0.18),
                resolution=res,
            )
            if pinn is not None:
                pinn_norm = np.abs(pinn)
                pinn_norm /= max(float(np.max(pinn_norm)), 1e-6)
                surrogate = 0.65 * surrogate + 0.35 * (pinn_norm * amp_map)
                mode = "pinn-assisted"
        except Exception:
            mode = "heuristic-surrogate"

    surrogate = np.clip(surrogate, 0.0, None)
    surrogate /= max(float(np.max(surrogate)), 1e-6)

    peak_idx = np.unravel_index(np.argmax(surrogate), surrogate.shape)
    peak_x = peak_idx[1] / max(res - 1, 1)
    peak_y = peak_idx[0] / max(res - 1, 1)
    explanation = (
        f"The AI marks the strongest damage around x={peak_x:.2f}, y={peak_y:.2f} because slower ground "
        f"amplifies wave energy there while shallow depth keeps the shaking concentrated."
    )
    return surrogate.astype(np.float32), explanation, mode


def _summarize_damage(buildings: List[Dict[str, object]]) -> Dict[str, int]:
    summary = {"safe": 0, "risky": 0, "collapse": 0}
    for building in buildings:
        summary[str(building["status"])] += 1
    return summary


@router.get("/presets")
def get_presets():
    return {"presets": list(PRESETS.values())}


@router.post("/simulate")
def simulate_earthquake(request: EarthquakeRequest):
    preset = PRESETS.get(request.preset, PRESETS["nepal"])
    map_scale_km = float(preset["map_scale_km"])
    resolution = int(request.resolution)
    speed_map, amp_map = _terrain_maps(resolution, request.medium_type, request.preset)

    p_sim = _configure_sim(resolution, speed_map * 1.08, request.wave_speed * 1.18, damping=0.018)
    s_sim = _configure_sim(resolution, speed_map * 0.82, request.wave_speed * 0.72, damping=0.032)

    epicenter = _normalized_coords(request.epicenter_x, request.epicenter_y)
    buildings = _building_catalog(request.preset)

    total_frames = int(_clamp(round(request.duration * 3.2), 24, 72))
    source_frames = max(6, int(total_frames * 0.24))
    source_radius = 0.045 + request.depth_km / 500.0
    source_strength = 0.22 + (request.magnitude ** 1.72) * 0.032
    depth_decay = 1.0 / (1.0 + request.depth_km / 24.0)

    max_heat = np.zeros((resolution, resolution), dtype=np.float32)
    max_wave = 0.0
    frames = []

    for frame_idx in range(total_frames):
        for _ in range(2):
            if frame_idx < source_frames:
                phase = math.sin((frame_idx + 1) / source_frames * math.pi)
                p_sim.add_impulse(epicenter[0], epicenter[1], strength=source_strength * phase * depth_decay, radius=source_radius)
                s_sim.add_impulse(epicenter[0], epicenter[1], strength=source_strength * phase * depth_decay * 0.72, radius=source_radius * 1.1)
            p_sim.step(steps=1)
            s_sim.step(steps=1)

        p_grid = p_sim.current_grid().copy()
        s_grid = s_sim.current_grid().copy()
        combined = (0.78 * p_grid + 1.08 * s_grid) * amp_map
        heat = np.clip(np.abs(p_grid) * 0.55 + np.abs(s_grid) * 1.05, 0.0, None) * amp_map
        if float(np.max(heat)) > 0:
            heat = heat / float(np.max(heat))
        max_heat = np.maximum(max_heat, heat.astype(np.float32))
        max_wave = max(max_wave, float(np.max(np.abs(combined))))

        frame_buildings = []
        for building in buildings:
            local_heat = _sample_grid(heat, float(building["x"]), float(building["y"]))
            local_amp = _sample_grid(amp_map, float(building["x"]), float(building["y"]))
            damage = local_heat * (0.34 + float(building["height"]) / 17.0) * _material_factor(str(building["material"])) * local_amp
            sway = local_heat * (8.0 + float(building["height"]) * 1.8)
            status = _status_for_damage(float(damage))
            building["damage_score"] = max(float(building.get("damage_score", 0.0)), float(damage))
            building["status"] = _status_for_damage(float(building["damage_score"]))
            frame_buildings.append(
                {
                    "id": building["id"],
                    "sway": round(float(sway), 3),
                    "damage": round(float(damage), 3),
                    "status": status,
                    "intensity": round(float(local_heat), 3),
                }
            )

        frame_radius = _front_radius(heat, (request.epicenter_x, request.epicenter_y), 0.18)
        frames.append(
            {
                "t": round(frame_idx * (request.duration / max(total_frames - 1, 1)), 2),
                "p_wave": np.round(p_grid, 3).flatten().tolist(),
                "s_wave": np.round(s_grid, 3).flatten().tolist(),
                "heat": np.round(heat, 3).flatten().tolist(),
                "p_radius": round(_front_radius(p_grid, (request.epicenter_x, request.epicenter_y), 0.09), 3),
                "s_radius": round(_front_radius(s_grid, (request.epicenter_x, request.epicenter_y), 0.075), 3),
                "affected_radius_km": round(frame_radius * map_scale_km, 2),
                "building_states": frame_buildings,
            }
        )

    ai_damage, explanation, ai_mode = _ai_damage_prediction(request, amp_map, max_heat)
    affected_area_pct = float(np.mean(max_heat > 0.28) * 100.0)
    critical_area_pct = float(np.mean(max_heat > 0.56) * 100.0)
    estimated_damage_pct = float(np.mean(ai_damage > 0.34) * 100.0)
    affected_radius_km = _front_radius(max_heat, (request.epicenter_x, request.epicenter_y), 0.28) * map_scale_km

    final_buildings = []
    for building in buildings:
        final_buildings.append(
            {
                "id": building["id"],
                "name": building["name"],
                "x": building["x"],
                "y": building["y"],
                "height": building["height"],
                "material": building["material"],
                "damage_score": round(float(building.get("damage_score", 0.0)), 3),
                "status": str(building.get("status", "safe")),
            }
        )

    summary = _summarize_damage(final_buildings)
    confidence = float(100.0 - min(18.0, abs(estimated_damage_pct - critical_area_pct) * 0.42))

    return {
        "preset": request.preset,
        "resolution": resolution,
        "epicenter": {"x": request.epicenter_x, "y": request.epicenter_y},
        "map_scale_km": map_scale_km,
        "frames": frames,
        "buildings": final_buildings,
        "max_heat": np.round(max_heat, 3).flatten().tolist(),
        "ai_prediction": {
            "damage_grid": np.round(ai_damage, 3).flatten().tolist(),
            "mode": ai_mode,
            "confidence": round(confidence, 1),
            "explanation": explanation,
        },
        "metrics": {
            "max_wave_amplitude": round(max_wave, 3),
            "affected_radius_km": round(affected_radius_km, 2),
            "affected_area_pct": round(affected_area_pct, 1),
            "estimated_damage_pct": round(estimated_damage_pct, 1),
            "critical_area_pct": round(critical_area_pct, 1),
        },
        "damage_summary": summary,
    }
