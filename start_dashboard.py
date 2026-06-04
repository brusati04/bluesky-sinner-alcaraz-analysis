import http.server
import socketserver
import webbrowser
import threading
import os
import sys

PORT = 8000
DIRECTORY = "web"

class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def open_browser():
    url = f"http://localhost:{PORT}"
    print(f"Opening browser at {url}...")
    webbrowser.open(url)

def main():
    if not os.path.isdir(DIRECTORY):
        print(f"Error: Directory '{DIRECTORY}' does not exist.")
        print("Please make sure you are running this script from the project root folder.")
        sys.exit(1)

    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
            print(f"\n=========================================")
            print(f"[*] Sinner vs Alcaraz Dashboard Server")
            print(f"-> Running at: http://localhost:{PORT}")
            print(f"-> Press Ctrl+C in this terminal to stop")
            print(f"=========================================\n")
            
            timer = threading.Timer(1.0, open_browser)
            timer.start()

            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\nStopping dashboard server. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
