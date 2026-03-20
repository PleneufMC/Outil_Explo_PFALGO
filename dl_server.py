import os, sys
os.chdir('/home/user/webapp')
from http.server import HTTPServer, SimpleHTTPRequestHandler
port = int(sys.argv[1]) if len(sys.argv) > 1 else 9090
server = HTTPServer(('0.0.0.0', port), SimpleHTTPRequestHandler)
print(f"Serving {os.getcwd()} on :{port}", flush=True)
server.serve_forever()
