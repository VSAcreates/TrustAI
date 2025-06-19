from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import logging
import time
from datetime import datetime
import random
from typing import List, Dict

# Import our fraud detection components
from transaction_simulator import generate_transaction, generate_transaction_stream
from fraud_detector import FraudDetector

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Fraud Detection API")

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the fraud detector
detector = FraudDetector()

# Store active connections
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: str):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except WebSocketDisconnect:
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for conn in disconnected:
            self.active_connections.remove(conn)

manager = ConnectionManager()

# In-memory storage for demo
recent_transactions = []
fraud_alerts = []
summary_stats = {
    "total_transactions": 0,
    "total_fraud": 0,
    "fraud_rate": 0.0,
    "highest_risk_score": 0.0,
    "total_amount": 0.0,
    "fraud_amount": 0.0
}

@app.get("/")
async def root():
    return {"message": "Fraud Detection API is running"}

@app.get("/status")
async def status():
    return {
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "detector": "active" if detector else "not loaded"
    }

@app.get("/transactions/recent")
async def get_recent_transactions(limit: int = 10):
    """Get recent transactions"""
    return recent_transactions[:limit]

@app.get("/alerts")
async def get_alerts(limit: int = 10):
    """Get fraud alerts"""
    return fraud_alerts[:limit]

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    return summary_stats

@app.post("/simulate")
async def simulate_transaction(fraud: bool = False, fraud_type: str = None):
    """Generate a simulated transaction"""
    transaction = generate_transaction(forced_fraud=fraud, fraud_type=fraud_type)
    result = detector.process_transaction(transaction)
    
    # Update stats and storage
    update_stats(transaction, result)
    
    return {
        "transaction": transaction,
        "analysis": result
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send initial data
        await websocket.send_json({
            "type": "init",
            "transactions": recent_transactions[:5],
            "alerts": fraud_alerts[:3],
            "stats": summary_stats
        })
        
        # Listen for commands
        while True:
            data = await websocket.receive_text()
            command = json.loads(data)
            
            if command["action"] == "get_updates":
                await websocket.send_json({
                    "type": "update",
                    "transactions": recent_transactions[:5],
                    "alerts": fraud_alerts[:3],
                    "stats": summary_stats
                })
            
            elif command["action"] == "simulate":
                # Generate transaction based on client request
                transaction = generate_transaction(
                    forced_fraud=command.get("force_fraud", False),
                    fraud_type=command.get("fraud_type")
                )
                
                # Process it
                result = detector.process_transaction(transaction)
                
                # Update stats and storage
                update_stats(transaction, result)
                
                # Send back the result
                await websocket.send_json({
                    "type": "new_transaction",
                    "transaction": transaction,
                    "analysis": result
                })
                
                # Broadcast to all clients if it's fraud
                if result["is_fraud"]:
                    await manager.broadcast(json.dumps({
                        "type": "fraud_alert",
                        "alert": {
                            "transaction_id": transaction["transaction_id"],
                            "timestamp": transaction["timestamp"],
                            "amount": transaction["amount"],
                            "risk_score": result["risk_score"],
                            "country": transaction["country"],
                            "risk_factors": result["risk_factors"]
                        }
                    }))
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def update_stats(transaction, result):
    """Update in-memory statistics"""
    global summary_stats, recent_transactions, fraud_alerts
    
    # Add to recent transactions (keep most recent first)
    recent_transactions.insert(0, {
        "transaction_id": transaction["transaction_id"],
        "amount": transaction["amount"],
        "timestamp": transaction["timestamp"],
        "country": transaction["country"],
        "merchant": transaction["merchant_name"],
        "risk_score": result["risk_score"],
        "is_fraud": result["is_fraud"]
    })
    
    # Trim list
    recent_transactions = recent_transactions[:100]
    
    # Add to alerts if fraud
    if result["is_fraud"]:
        fraud_alerts.insert(0, {
            "transaction_id": transaction["transaction_id"],
            "amount": transaction["amount"],
            "timestamp": transaction["timestamp"],
            "country": transaction["country"],
            "risk_score": result["risk_score"],
            "risk_factors": result["risk_factors"]
        })
        fraud_alerts = fraud_alerts[:100]
    
    # Update summary stats
    summary_stats["total_transactions"] += 1
    summary_stats["total_amount"] += transaction["amount"]
    
    if result["is_fraud"]:
        summary_stats["total_fraud"] += 1
        summary_stats["fraud_amount"] += transaction["amount"]
    
    summary_stats["fraud_rate"] = (summary_stats["total_fraud"] / summary_stats["total_transactions"]) * 100
    summary_stats["highest_risk_score"] = max(summary_stats["highest_risk_score"], result["risk_score"])

async def background_transaction_simulator():
    """Generate transactions in the background"""
    while True:
        # Sleep to control rate
        await asyncio.sleep(random.uniform(2.0, 5.0))
        
        # Generate transaction with default fraud probability
        transaction = generate_transaction()
        
        # Process transaction
        result = detector.process_transaction(transaction)
        
        # Update stats and storage
        update_stats(transaction, result)
        
        # Broadcast to all connected clients
        await manager.broadcast(json.dumps({
            "type": "new_transaction",
            "transaction": {
                "transaction_id": transaction["transaction_id"],
                "amount": transaction["amount"],
                "timestamp": transaction["timestamp"],
                "country": transaction["country"],
                "merchant": transaction["merchant_name"],
                "risk_score": result["risk_score"],
                "is_fraud": result["is_fraud"]
            }
        }))
        
        # Send fraud alert if detected
        if result["is_fraud"]:
            await manager.broadcast(json.dumps({
                "type": "fraud_alert",
                "alert": {
                    "transaction_id": transaction["transaction_id"],
                    "timestamp": transaction["timestamp"],
                    "amount": transaction["amount"],
                    "risk_score": result["risk_score"],
                    "country": transaction["country"],
                    "risk_factors": result["risk_factors"]
                }
            }))

@app.on_event("startup")
async def startup_event():
    # Start background task to simulate transactions
    asyncio.create_task(background_transaction_simulator())
    logger.info("Transaction simulator started")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)