import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "backend:app",
        host="0.0.0.0", port=8000, reload=True,
        reload_includes=["server.py", "backend.py"], reload_excludes="*.py")