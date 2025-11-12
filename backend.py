#!/usr/bin/env python
# coding: utf-8

# In[8]:


import threading
from typing import Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from safetensors.torch import load_file
from LinearRegression import *
import torch
device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.backends.cuda.is_available() else 'cpu'
torch.set_default_device(device)

model: LinearRegression = None


# In[ ]:


@asynccontextmanager
async def lifespan(app: FastAPI):
    global init_models_thread
    init_models_thread.start()
    yield

app = FastAPI(title=__name__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allowed origins
    allow_credentials=True,         # Allow cookies/auth headers
    allow_methods=["*"],            # Allow all HTTP methods
    allow_headers=["*"],            # Allow all HTTP headers
)


# In[ ]:


class Predict(BaseModel):
    x: int

@app.post("/predict")
async def predict(
    p: Predict,
):
    global model
    if model is None:
        raise HTTPException(
            status_code=400,
            detail=[{"msg": "Models are not ready. Check the backend log for keyword 'init_models done.'"}]
        )

    with torch.inference_mode():
        y = model(p.x)
        return {"y_predict": y.item()}


# In[2]:


def init_models():
    print("init_models start...", flush=True)
    global model, device
    state_dict = load_file("models/model.safetensors")
    model = LinearRegression()
    model.to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print("init_models done!", flush=True)

init_models_thread = threading.Thread(target=init_models)


# In[ ]:


@app.get("/")
async def index():
    global model
    if model is None:
        return "Not ready!"
    return "Hi."

