from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import os
from datetime import datetime
from urllib.parse import parse_qs
import cgi
import random

# Configuration
UPLOAD_FOLDER = "./uploads"
PORT = 8080

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


class ImageUploadHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path == '/upload':
            try:
                # Parse multipart form data
                content_type = self.headers['Content-Type']
                if 'multipart/form-data' not in content_type:
                    self.send_error(400, "Expected multipart/form-data")
                    return

                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={'REQUEST_METHOD': 'POST'}
                )

                # Get image file
                if 'image' not in form:
                    self.send_error(400, "No image field in request")
                    return

                image_field = form['image']
                if not image_field.file:
                    self.send_error(400, "No image data")
                    return

                # Get metadata (optional)
                metadata = {}
                if 'metadata' in form:
                    metadata = json.loads(form['metadata'].value)

                # Generate random 6-digit filename
                random_digits = f"{random.randint(100000, 999999)}"
                ext = os.path.splitext(image_field.filename or "image.jpg")[1]
                image_filename = f"{random_digits}{ext}"

                # Save image
                image_path = os.path.join(UPLOAD_FOLDER, image_filename)
                with open(image_path, 'wb') as f:
                    f.write(image_field.file.read())

                # Save metadata
                # metadata['upload_time'] = timestamp
                # metadata['original_filename'] = original_filename
                # metadata_path = os.path.join(UPLOAD_FOLDER, f"{base_name}.json")
                # with open(metadata_path, 'w') as f:
                #     json.dump(metadata, f, indent=2)

                # Send response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                response = {
                    'status': 'success',
                    'filename': image_filename,
                    'message': 'Image uploaded successfully'
                }
                self.wfile.write(json.dumps(response).encode())

                print(f"✓ Saved: {image_filename}")

            except Exception as e:
                self.send_error(500, f"Error: {str(e)}")
                print(f"✗ Error: {e}")
        else:
            self.send_error(404, "Endpoint not found")

    def log_message(self, format, *args):
        # Suppress default logging
        pass


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), ImageUploadHandler)
    print(f"🚀 Image upload server running on port {PORT}")
    print(f"📁 Upload folder: {os.path.abspath(UPLOAD_FOLDER)}")
    print(f"📡 Endpoint: http://<your-ip>:{PORT}/upload")
    print("Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 Server stopped")
        server.shutdown()