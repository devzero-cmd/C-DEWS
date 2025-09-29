import urllib.request

SHOT_URL = "http://192.168.1.16:8080/"

try:
    resp = urllib.request.urlopen(SHOT_URL, timeout=5)
    data = resp.read()
    print("Connection successful, bytes received:", len(data))
except Exception as e:
    print("Connection failed:", e)
