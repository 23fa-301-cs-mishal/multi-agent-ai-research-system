import requests

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

r = requests.get("https://openrouter.ai", headers=headers, timeout=15)

if not r.text.strip():
    print("Empty response! Status:", r.status_code, "Headers:", r.headers)
else:
    data = r.json()["data"]
    free_models = [
        m for m in data
        if m["id"].endswith(":free") and m["id"] != "openrouter/free"
    ]
    free_models.sort(key=lambda x: x["context_length"], reverse=True)

    for m in free_models:
        print(m["context_length"], "-->", m["id"])
