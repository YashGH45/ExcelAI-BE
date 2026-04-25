import psutil
import uvicorn


def free_port(port: int) -> None:
    for proc in psutil.process_iter(["pid", "name"]):
        try:
            for conn in proc.net_connections(kind="inet"):
                if conn.laddr.port == port:
                    print(f"Killing process {proc.pid} ({proc.name()}) on port {port}")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


if __name__ == "__main__":
    free_port(8000)
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="debug",
    )

