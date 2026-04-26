"""
wave.io backend.
Serves the frontend and streams a live finite-difference wave simulation.
"""
import asyncio
import json
import os

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from earthquake import router as earthquake_router
from simulator import WaveSim2D

app = FastAPI(title="wave.io")
app.include_router(earthquake_router)

_FRONTEND = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if os.path.isdir(_FRONTEND):
    app.mount("/static", StaticFiles(directory=_FRONTEND), name="static")

    @app.get("/")
    def index():
        return FileResponse(os.path.join(_FRONTEND, "index.html"))

    @app.get("/workspace")
    def workspace():
        return FileResponse(os.path.join(_FRONTEND, "index.html"))

    def _page(name: str):
        return FileResponse(os.path.join(_FRONTEND, name))

    @app.get("/landing")
    def landing():
        return _page("landing.html")

    @app.get("/scenarios")
    def scenarios():
        return _page("scenarios.html")

    @app.get("/validation")
    def validation():
        return _page("validation.html")

    @app.get("/explain")
    def explain():
        return _page("explain.html")

    @app.get("/about")
    def about():
        return _page("about.html")

    @app.get("/earthquake")
    def earthquake():
        return _page("earthquake.html")

    @app.get("/{page_name}.html")
    def page_html(page_name: str):
        page_map = {
            "index": "index.html",
            "workspace": "index.html",
            "landing": "landing.html",
            "scenarios": "scenarios.html",
            "validation": "validation.html",
            "explain": "explain.html",
            "about": "about.html",
            "earthquake": "earthquake.html",
        }
        filename = page_map.get(page_name)
        if not filename:
            raise HTTPException(status_code=404, detail="Not Found")
        return _page(filename)


@app.get("/health")
def health():
    return {"status": "ok", "engine": "fdtd"}


_DEFAULT_PARAMS = {
    "scenario": "b",
    "speed": 1.0,
    "amplitude": 1.0,
    "omega": 11.0,
    "source_strength": 1.0,
    "damping": 0.02,
    "time_scale": 0.9,
    "separation": 0.38,
    "gap_width": 0.25,
    "c_upper": 1.0,
    "c_lower": 0.55,
    "boundary": "absorbing",
    "preset": "none",
    "resolution": 96,
    "source_x": 0.0,
    "source_y": 0.0,
    "medium_density": 1.45,
    "brush_radius": 0.08,
}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()

    params = dict(_DEFAULT_PARAMS)
    sim = WaveSim2D(int(params["resolution"]))
    sim.sync_params(params)

    async def receiver():
        try:
            while True:
                raw = await websocket.receive_text()
                update = json.loads(raw)
                action = update.pop("action", None)
                if update:
                    params.update(update)
                    sim.sync_params(update)
                if isinstance(action, dict):
                    sim.apply_action(action)
        except Exception:
            pass

    recv_task = asyncio.create_task(receiver())

    try:
        while True:
            loop_start = asyncio.get_event_loop().time()

            sim.step(steps=1)
            grid = sim.current_grid()
            metrics = sim.metrics()

            std = float(np.std(grid))
            if std > 1e-6:
                grid = np.tanh(grid / (std * 2.8))

            payload = json.dumps({
                "t": round(sim.time, 4),
                "grid": np.round(grid, 3).flatten().tolist(),
                "scenario": str(params["scenario"]),
                "res": int(sim.resolution),
                "metrics": {
                    "energy_mean": round(metrics["energy_mean"], 6),
                    "energy_max": round(metrics["energy_max"], 6),
                    "dt": round(metrics["dt"], 6),
                    "dt_cfl": round(metrics["dt_cfl"], 6),
                    "cfl_ratio": round(metrics["cfl_ratio"], 4),
                },
            })
            await websocket.send_text(payload)

            elapsed = asyncio.get_event_loop().time() - loop_start
            await asyncio.sleep(max(0.0, 0.016 - elapsed))

    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel()
