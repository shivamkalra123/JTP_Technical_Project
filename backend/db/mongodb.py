from motor.motor_asyncio import AsyncIOMotorClient
import os
from dotenv import load_dotenv
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["Users"]

prefs_collection = db["user"]          # user prefs collection, uses $userid field
profiles_collection = db["user_profiles"]
