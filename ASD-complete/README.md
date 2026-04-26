# ASD Detection App

End-to-end web application for Autism Spectrum Disorder (ASD) detection from facial images using a pre-trained PyTorch model.

## Project Structure

```text
backend/
  main.py
  model.py
  vit_asd_best.pth # your trained PyTorch weights
  requirements.txt
  utils.py
frontend/
  index.html
  package.json
  vite.config.js
  .env.example
  src/
    App.css
    App.jsx
    index.css
    main.jsx
    components/
      Result.jsx
      Upload.jsx
```

## Backend Setup

1. Create a Python virtual environment.
2. Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. Copy your trained `.pth` file to `backend/vit_asd_best.pth`.
   If your training notebook used `GELU` instead of `ReLU` in the custom ViT head, set
   `MODEL_HEAD_ACTIVATION=gelu` before starting the API.
4. Start the API:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend Setup

1. Install dependencies:

```bash
cd frontend
npm install
```

2. Start the Vite development server:

```bash
npm run dev
```

The frontend runs on `http://localhost:5173` and sends predictions to `http://localhost:8000`.

## Important Assumption

The backend now auto-detects your `vit_asd_best.pth` checkpoint and loads it as a ViT-style model. It also still supports a few common torchvision CNN checkpoints. If your training code used a custom ViT head activation that differs from the default `ReLU`, update `build_vit_model()` in [backend/model.py](backend/model.py).
