
import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv("JUSTONE_API_TOKEN")
print("Token loaded:", bool(API_TOKEN))
print("Placeholder still present:", API_TOKEN == "YOUR_NEW_API_TOKEN_HERE")

if not API_TOKEN:
    raise RuntimeError(
        "API token missing. Add JUSTONE_API_TOKEN to your .env file."
    )

url = "https://api.justoneapi.com/api/amazon/get-product-top-reviews/v1"

params = {
    "token": API_TOKEN,
    "asin": "B0C248QHSS",
    "country": "IN",
}

try:
    response = requests.get(url, params=params, timeout=120)

    print("HTTP status:", response.status_code)
    response.raise_for_status()

    result = response.json()

    print("Response status:", result.get("code", result.get("status")))

    data = result.get("data") or {}
    reviews = data.get("reviews") or []

    print("Returned fields:", list(data.keys()))
    print("Number of reviews:", len(reviews))

    if reviews:
        print("\nFirst review:")
        print(reviews[0])
    else:
        print("\nNo individual reviews returned.")
        print("Response message:", result.get("message", "No message provided"))

except requests.RequestException as error:
    print("API request failed:", str(error))
except ValueError:
    print("The API did not return valid JSON.")

