import requests
import json
import time

url = "http://127.0.0.1:8000/api/v1/kyc/submit"
payload = {
    "full_name": "hayyan1",
    "nationality": "Pakistan",
    "phone_number": "0303-0303030",
    "id_card_num": "112233445566"
}
headers = {"Content-Type": "application/json"}

print(f"Sending POST to {url} with payload {payload}")
start = time.time()
try:
    response = requests.post(url, json=payload, timeout=60)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except requests.exceptions.Timeout:
    print("REQUEST TIMED OUT AFTER 60 SECONDS! It is truly hanging!")
except Exception as e:
    print(f"Error: {e}")
print(f"Time taken: {time.time() - start:.2f}s")
