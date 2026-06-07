"""Web 服务启动入口"""
import uvicorn
from web.config import HOST, PORT

if __name__ == "__main__":
    uvicorn.run(
        "web.app:app",
        host=HOST,
        port=PORT,
        reload=False,
        log_level="info",
    )
