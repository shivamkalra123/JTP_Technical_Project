from motor.motor_asyncio import AsyncIOMotorClient
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://host.docker.internal:27017")
DB_NAME = "Users"
COLLECTIONS = ["user", "user_profiles"]

async def create_collections():
    client = AsyncIOMotorClient(MONGO_URI)
    db = client[DB_NAME]

    existing_collections = await db.list_collection_names()

    for collection in COLLECTIONS:
        if collection not in existing_collections:
            await db.create_collection(collection)
            print(f"✅ Created collection: {collection}")
        else:
            print(f"ℹ️ Collection already exists: {collection}")

    # Optional: Create indexes
    await db["user"].create_index("user_id", unique=True)
    print("✅ Index created on user_id in 'user' collection")

    client.close()

if __name__ == "__main__":
    asyncio.run(create_collections())
