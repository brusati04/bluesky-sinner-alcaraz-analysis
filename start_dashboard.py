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
        # Serve files relative to the 'web' directory
        super().__init__(*args, directory=DIRECTORY, **kwargs)

def open_browser():
    url = f"http://localhost:{PORT}"
    print(f"Opening browser at {url}...")
    webbrowser.open(url)

def main():
    # Check if the web directory exists
    if not os.path.isdir(DIRECTORY):
        print(f"Error: Directory '{DIRECTORY}' does not exist.")
        print("Please make sure you are running this script from the project root folder.")
        sys.exit(1)

    # Enable port reuse so restarting doesn't throw 'address already in use' errors
    socketserver.TCPServer.allow_reuse_address = True

    try:
        with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
            print(f"\n=========================================")
            print(f"[*] Sinner vs Alcaraz Dashboard Server")
            print(f"-> Running at: http://localhost:{PORT}")
            print(f"-> Press Ctrl+C in this terminal to stop")
            print(f"=========================================\n")
            
            # Start a timer to open the browser 1 second after server starts
            timer = threading.Timer(1.0, open_browser)
            timer.start()

            # Start serving
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\nStopping dashboard server. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
