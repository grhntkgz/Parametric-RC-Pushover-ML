from http.server import ThreadingHTTPServer
from dashboard.server import DashboardHandler
server = ThreadingHTTPServer(("127.0.0.1", 8766), DashboardHandler)
print("SAP2000 generator live viewer: http://127.0.0.1:8766", flush=True)
server.serve_forever()
