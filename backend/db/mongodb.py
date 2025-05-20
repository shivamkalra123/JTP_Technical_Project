from motor.motor_asyncio import AsyncIOMotorClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://admin:admin@cluster0.jdzfnt4.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
client = AsyncIOMotorClient(MONGO_URI)
db = client["Users"]

prefs_collection = db["user"]          # user prefs collection, uses $userid field
profiles_collection = db["user_profiles"]
