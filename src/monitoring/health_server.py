"""
Production Health Monitoring Endpoint
Provides system status, model health, and operational metrics
"""

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from pathlib import Path
import json
import psutil
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(title="TelegramSoccer Health Monitor", version="1.0.0")


def check_model_health() -> dict:
    """Check if trained models exist and are recent"""
    models_dir = Path("models/knowledge_enhanced")
    models = {
        "over_1_5": models_dir / "over_1_5_model.pkl",
        "over_2_5": models_dir / "over_2_5_model.pkl",
        "btts": models_dir / "btts_model.pkl"
    }
    
    status = {}
    all_healthy = True
    
    for model_name, model_path in models.items():
        if model_path.exists():
            # Check if model is recent (updated within 7 days)
            mtime = datetime.fromtimestamp(model_path.stat().st_mtime)
            age_days = (datetime.now() - mtime).days
            is_recent = age_days < 7
            
            status[model_name] = {
                "exists": True,
                "last_updated": mtime.isoformat(),
                "age_days": age_days,
                "healthy": is_recent
            }
            
            if not is_recent:
                all_healthy = False
        else:
            status[model_name] = {
                "exists": False,
                "healthy": False
            }
            all_healthy = False
    
    return {
        "models": status,
        "overall_healthy": all_healthy
    }


def check_data_health() -> dict:
    """Check if training data exists and is sufficient"""
    data_file = Path("data/historical/massive_training_data.csv")
    
    if not data_file.exists():
        return {
            "exists": False,
            "healthy": False,
            "message": "Training data not found"
        }
    
    # Check file size (should be >1MB for 14K matches)
    size_mb = data_file.stat().st_size / (1024 * 1024)
    
    return {
        "exists": True,
        "size_mb": round(size_mb, 2),
        "healthy": size_mb > 1.0,
        "path": str(data_file)
    }


def check_system_resources() -> dict:
    """Check system CPU, memory, disk usage"""
    return {
        "cpu_percent": psutil.cpu_percent(interval=1),
        "memory_percent": psutil.virtual_memory().percent,
        "disk_percent": psutil.disk_usage('/').percent,
        "healthy": (
            psutil.cpu_percent() < 90 and 
            psutil.virtual_memory().percent < 90 and
            psutil.disk_usage('/').percent < 90
        )
    }


@app.get("/health")
def health_check():
    """
    Comprehensive health check endpoint
    Returns 200 if system is healthy, 503 if degraded
    """
    model_health = check_model_health()
    data_health = check_data_health()
    system_health = check_system_resources()
    
    overall_healthy = (
        model_health["overall_healthy"] and
        data_health["healthy"] and
        system_health["healthy"]
    )
    
    response = {
        "status": "healthy" if overall_healthy else "degraded",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "models": model_health,
            "data": data_health,
            "system": system_health
        }
    }
    
    status_code = 200 if overall_healthy else 503
    return JSONResponse(content=response, status_code=status_code)


@app.get("/health/simple")
def simple_health():
    """Simple OK/FAIL health check"""
    model_health = check_model_health()
    data_health = check_data_health()
    
    if model_health["overall_healthy"] and data_health["healthy"]:
        return {"status": "ok"}
    else:
        return JSONResponse(
            content={"status": "fail"},
            status_code=503
        )


@app.get("/metrics")
def metrics():
    """
    Prometheus-compatible metrics endpoint
    """
    model_health = check_model_health()
    data_health = check_data_health()
    system_health = check_system_resources()
    
    metrics_text = f"""# HELP telegramsoccer_models_healthy Models health status
# TYPE telegramsoccer_models_healthy gauge
telegramsoccer_models_healthy {{'model_type'="over_1_5"}} {1 if model_health["models"]["over_1_5"]["healthy"] else 0}
telegramsoccer_models_healthy {{'model_type'="over_2_5"}} {1 if model_health["models"]["over_2_5"]["healthy"] else 0}
telegramsoccer_models_healthy {{'model_type'="btts"}} {1 if model_health["models"]["btts"]["healthy"] else 0}

# HELP telegramsoccer_data_size_mb Training data size in MB
# TYPE telegramsoccer_data_size_mb gauge
telegramsoccer_data_size_mb {data_health.get("size_mb", 0)}

# HELP telegramsoccer_cpu_percent CPU usage percentage
# TYPE telegramsoccer_cpu_percent gauge
telegramsoccer_cpu_percent {system_health["cpu_percent"]}

# HELP telegramsoccer_memory_percent Memory usage percentage
# TYPE telegramsoccer_memory_percent gauge
telegramsoccer_memory_percent {system_health["memory_percent"]}

# HELP telegramsoccer_disk_percent Disk usage percentage
# TYPE telegramsoccer_disk_percent gauge
telegramsoccer_disk_percent {system_health["disk_percent"]}
"""
    
    return Response(content=metrics_text, media_type="text/plain")


@app.get("/")
def root():
    """Root endpoint with system information"""
    return {
        "service": "TelegramSoccer Health Monitor",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Comprehensive health check (JSON)",
            "/health/simple": "Simple OK/FAIL check",
            "/metrics": "Prometheus metrics"
        },
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
