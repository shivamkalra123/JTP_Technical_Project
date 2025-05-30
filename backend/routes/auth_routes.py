from fastapi import APIRouter, HTTPException
from models.prompt_model import UserCreate, UserLogin
from services.auth_service import hash_password, verify_password, generate_uuid
from services.jwt_service import create_access_token
from db.mongodb import prefs_collection

router = APIRouter()

@router.post("/signup")
async def signup(user: UserCreate):
    existing_user = await prefs_collection.find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already exists")

    user_id = generate_uuid()
    hashed_pw = hash_password(user.password)

    user_doc = {
        "user_id": user_id,
        "email": user.email,
        "hashed_password": hashed_pw
    }

    await prefs_collection.insert_one(user_doc)
    return {"message": "Signup successful", "user_id": user_id}

@router.post("/login")
async def login(user: UserLogin):
    db_user = await prefs_collection.find_one({"email": user.email})
    if not db_user or not verify_password(user.password, db_user["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"user_id": db_user["user_id"], "email": db_user["email"]})
    return {"access_token": token, "token_type": "bearer", "user_id": db_user["user_id"]}
