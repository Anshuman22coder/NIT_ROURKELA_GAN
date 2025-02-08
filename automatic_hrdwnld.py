import os
import requests

PEXELS_API_KEY = "OZesYL2PjHvT2kGR0rAtiMC0k2sNRwqqj2fRcHrvUc2wvJIym11Qlf0x"  # Replace with your API key
PEXELS_URL = "https://api.pexels.com/v1/search?query=nature&per_page=50"

headers = {"Authorization": PEXELS_API_KEY}

# Create directory
os.makedirs("data/superres/HR", exist_ok=True)

# Fetch image URLs
response = requests.get(PEXELS_URL, headers=headers)
if response.status_code == 200:
    data = response.json()
    images = data["photos"]
    
    for i, img in enumerate(images):
        img_url = img["src"]["original"]
        img_data = requests.get(img_url).content
        with open(f"data/superres/HR/image_{i+1}.jpg", "wb") as f:
            f.write(img_data)
        print(f"✅ Downloaded: image_{i+1}.jpg")
else:
    print("❌ Failed to fetch images from Pexels API")
