"""
Realtime WebSocket endpoint for live KPI updates.
Pushes simulated transaction events to connected clients.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import List, Dict
import asyncio
import json
import random
from datetime import datetime

router = APIRouter(prefix="/api/realtime", tags=["realtime"])


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for conn in self.active_connections:
            try:
                await conn.send_json(message)
            except Exception:
                dead.append(conn)
        for conn in dead:
            self.disconnect(conn)


manager = ConnectionManager()


@router.websocket("/ws")
async def realtime_stream(websocket: WebSocket):
    """
    WebSocket endpoint for realtime data stream.

    Server pushes simulated KPI updates and AI insights every few seconds.
    Client can also send commands (e.g., subscribe to specific KPIs).
    """
    await manager.connect(websocket)
    try:
        # Send welcome message
        await websocket.send_json({
            "type": "welcome",
            "message": "Connected to Nexlytics realtime stream",
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Periodic update loop
        counter = 0
        while True:
            await asyncio.sleep(3)  # Update every 3 seconds
            counter += 1

            # Simulate KPI fluctuation
            event = {
                "type": "kpi_update",
                "timestamp": datetime.utcnow().isoformat(),
                "tick": counter,
                "data": {
                    "revenue": round(80_000_000 + random.uniform(-3_000_000, 4_000_000), 2),
                    "expense": round(45_000_000 + random.uniform(-2_000_000, 2_500_000), 2),
                    "transactions": random.randint(150, 300),
                    "active_users": random.randint(800, 1200),
                },
            }
            await websocket.send_json(event)

            # Occasionally trigger an insight
            if counter % 5 == 0:
                insight_types = [
                    {
                        "type": "alert",
                        "severity": "warning",
                        "title": "Revenue Drop Detected",
                        "message": "Revenue in Region Jakarta dropped 12% in the last hour. Possible cause: stockout on Laptop Pro.",
                        "confidence": 0.78,
                    },
                    {
                        "type": "recommendation",
                        "severity": "info",
                        "title": "Restock Suggestion",
                        "message": "Based on demand forecasting, Smartphone X requires restocking within 3 days.",
                        "confidence": 0.85,
                    },
                    {
                        "type": "trend",
                        "severity": "info",
                        "title": "Channel Growth",
                        "message": "Marketplace A shows 18% growth this week, outpacing other channels.",
                        "confidence": 0.91,
                    },
                ]
                await websocket.send_json({
                    "type": "ai_insight",
                    "timestamp": datetime.utcnow().isoformat(),
                    "insight": random.choice(insight_types),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        manager.disconnect(websocket)
