import os
import base64
import json
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import uvicorn

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Model for the request body
class AnalyzeRequest(BaseModel):
    image: str

def generate_pixel_art(image_data_url: str) -> dict:
    try:
        if "," not in image_data_url:
            return {"error": "Invalid image data URL format."}

        header, encoded_image = image_data_url.split(",", 1)
        try:
            mime_type = header.split(";", 1)[0].replace("data:", "")
        except Exception:
            mime_type = "image/png"

        try:
            image_bytes = base64.b64decode(encoded_image)
        except Exception as e:
            return {"error": f"Base64 decoding failed: {str(e)}"}

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY environment variable is not set on server."}

        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")

        prompt = (
            "Analyze this image and provide a beautiful, artistically curated color palette "
            "of 8 to 12 distinct hex colors that capture the essence of the scene. "
            "Also provide a short, one-sentence theme description. "
            "Return ONLY a JSON object. No markdown, no explanation. "
            "Format: { \"palette\": [\"#hex1\", \"#hex2\", ...], \"theme\": \"description\" }"
        )

        response = client.models.generate_content(
            model=model,
            contents=[prompt, genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
        )

        text = response.text or "{}"
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": f"AI returned invalid JSON: {text[:100]}..."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

@app.post("/api/analyze")
async def analyze_image(request: AnalyzeRequest):
    result = generate_pixel_art(request.image)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return {"result": result}

# Serve index.html as the root
@app.get("/")
async def read_index():
    return FileResponse("index.html")

# Serve other static files (if any)
app.mount("/static", StaticFiles(directory="."), name="static")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
