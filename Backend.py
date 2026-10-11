import os
import sys
import base64
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from google import genai

HOST = "127.0.0.1"
PORT = 8000


def generate_pixel_art(image_data_url: str) -> dict:
    header, encoded_image = image_data_url.split(",", 1)
    mime_type = header.split(";", 1)[0].removeprefix("data:")
    image_bytes = base64.b64decode(encoded_image)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    model = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
    prompt = (
        "Analyze this image and convert it into a simplified 16x16 pixel art version. "
        "1. Create a palette of up to 8 distinct hex colors found in the image as a simple list. "
        "2. Map the image to a 16x16 grid where each cell is an integer (0 to palette_length-1). "
        "3. Return the result ONLY as a JSON object. No explanation, no markdown. "
        "Structure: { \"palette\": [\"#hex1\", \"#hex2\", ...], \"grid\": [ [0, 1, ...], [ ... ], ... ] } "
        "Ensure the grid is exactly 16 rows and 16 columns."
    )
    print(f"Sending {len(image_bytes) // 1024} KB image to Google Gemini API using {model}...", flush=True)
    response = client.models.generate_content(
        model=model,
        contents=[prompt, genai.types.Part.from_bytes(data=image_bytes, mime_type=mime_type)],
    )
    print("Google Gemini API response received.", flush=True)
    text = response.text or "{}"

    # More robust JSON extraction: find content between the first '{' and last '}'
    import re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "Gemma returned an invalid JSON format.", "raw": text}



class AppHandler(SimpleHTTPRequestHandler):
    def do_POST(self) -> None:
        if urlparse(self.path).path != "/api/analyze":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            result = generate_pixel_art(payload["image"])
            body = json.dumps({"result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (KeyError, ValueError, TypeError, base64.binascii.Error) as error:
            self.send_error(400, f"Invalid image request: {error}")
        except genai.errors.APIError as error:
            self.send_error(502, f"Gemma request failed: {error}")


def main() -> None:
    if len(sys.argv) == 2 and sys.argv[1] == "--serve":
        if not os.getenv("GEMINI_API_KEY"):
            raise SystemExit("GEMINI_API_KEY is not set. Export it before starting the server.")
        print(f"Open http://{HOST}:{PORT} in your browser")
        ThreadingHTTPServer((HOST, PORT), AppHandler).serve_forever()
        return

    prompt = " ".join(sys.argv[1:]).strip()
    if not prompt:
        prompt = input("What should Gemma create? ").strip()
    if not prompt:
        raise SystemExit("Please provide a prompt.")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY is not set. Export it before running this script."
        )

    client = genai.Client(api_key=api_key)
    model = os.getenv("GEMMA_MODEL", "gemma-4-26b-a4b-it")
    response = client.models.generate_content(model=model, contents=prompt)
    print(response.text)


if __name__ == "__main__":
    main()