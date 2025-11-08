from fastapi import FastAPI
from api.routes.routes import router
import uvicorn
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="EduAssist",
    description="Hello! This is EduAssist!!!"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Hoặc chỉ định frontend domain cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include tất cả các endpoint từ router
app.include_router(router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
