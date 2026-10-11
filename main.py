import os
import base64
import json
import re
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google import genai
import uvicorn

app = FastAPI()

# Model for the request body
class AnalyzeRequest(BaseModel):
    image: str

def generate_pixel_art(image_data_url: str) -> dict:
    try:
        header, encoded_image = image_data_url.split(",", 1)
        mime_type = header.split(";", 1)[0].removeprefix("data:")
        image_bytes = base64.b64decode(encoded_image)

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")

        client = genai.Client(api_key=api_key)
        model = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")

        prompt = (
            "Analyze this image and convert it into a simplified 16x16 pixel art version. "
            "1. Create a palette of up to 8 distinct hex colors found in the image as a simple list. "
            "2. Map the image to a 16x16 grid where each cell is an integer (0 to palette_length-1). "
            "3. Return the result ONLY as a JSON object. No explanation, no markdown. "
            "Structure: { \"palette\": [\"#hex1\", \"#hex2\", ...], \"grid\": [ [0, 1, ...], [ ... ], ... ] } "
            "Ensure the grid is exactly 16 rows and 16 columns."
        )

        response = client.models.generate_content(
            model=model,
            contents=[prompt, genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
        )

        text = response.text or "{}"
        # Robust JSON extraction
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            text = match.group(0)

        return json.loads(text)
    except Exception as e:
        return {"error": str(e)}

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
