import os
import requests

# Create directory for background images
os.makedirs("data/backgrounds", exist_ok=True)

# Pexels API Key (Get one at https://www.pexels.com/api/)
PEXELS_API_KEY = "OZesYL2PjHvT2kGR0rAtiMC0k2sNRwqqj2fRcHrvUc2wvJIym11Qlf0x"
PEXELS_URL = "https://api.pexels.com/v1/search"

# Query for background images
params = {"query": "background", "per_page": 100}
headers = {"Authorization": PEXELS_API_KEY}

response = requests.get(PEXELS_URL, headers=headers, params=params)
data = response.json()

# Download images
for i, photo in enumerate(data["photos"]):
    image_url = photo["src"]["large"]
    img_data = requests.get(image_url).content

    with open(f"data/backgrounds/background_{i+1}.jpg", "wb") as f:
        f.write(img_data)

    print(f"✅ Downloaded: background_{i+1}.jpg")

print("🎉 All background images downloaded!")
