import requests

# Predicción individual
response = requests.post("http://localhost:8000/predict", json={
    "BHK": 2,
    "Size": 1100,
    "Area_Type": 1,
    "City": 2,
    "Furnishing_Status": 1,
    "Tenant_Preferred": 0,
    "Bathroom": 2,
    "floor_number": 3,
    "total_floors": 10
})

print(response.json())