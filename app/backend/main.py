from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File,Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import models, schemas, security
from db import get_db, engine
from datetime import datetime
import shutil
from pathlib import Path
from screening_service import analyze_image
from config import settings

# Initialize the database and FastAPI app
models.Base.metadata.create_all(bind=engine)
app = FastAPI()

origins = [
    "http://localhost",            
    "http://localhost:3000",       
    # Your production domain
]

# Add CORS middleware to the app.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,         # Allow requests from specified origins only
    allow_credentials=True,        # Allow cookies to be sent with cross-origin requests
    allow_methods=["*"],           # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],           # Allow all headers
)
    
def get_current_user(
    username: str,  # username as a parameter
    db: Session = Depends(get_db)
):
    # Query for the user based on the provided username
    user = db.query(models.User).filter(models.User.username == username).first()
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@app.post("/signup", response_model=schemas.User)
def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = security.get_password_hash(user.password)
    db_user = models.User(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

@app.post("/login")
def login(user_credentials: schemas.UserLogin, db: Session = Depends(get_db)):
    # Check if user exists
    user = db.query(models.User).filter(models.User.username == user_credentials.username).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Verify password
    if not security.verify_password(user_credentials.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect password"
        )

    return {"message": "Login successful", "username": user.username, "userId" : user.id}

@app.post("/screenings/", response_model=schemas.Screening)
async def create_screening(
    file: UploadFile = File(...),  # Expect file as form-data upload
    current_user: models.User = Depends(get_current_user),  # Get the current authenticated user
    db: Session = Depends(get_db)
):
    # Ensure uploads directory exists
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    
    # Save the uploaded file
    file_path = upload_dir / f"{datetime.now().timestamp()}_{file.filename}"
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Read image as bytes for processing
    with file_path.open("rb") as img_file:
        image_bytes = img_file.read()
    
    # Analyze image using bytes data
    result, confidence = analyze_image(image_bytes)
    confidence_value = float(confidence)
    
    # Create screening record
    screening = models.Screening(
        user_id=current_user.id,
        image_path=str(file_path),
        result=result,
        confidence=confidence_value,
        created_at=datetime.now()
    )
    db.add(screening)
    db.commit()
    db.refresh(screening)
    return screening
@app.get("/screenings/", response_model=list[schemas.Screening])
def get_user_screenings(
    username: str,  # Pass the username directly as a query parameter
    db: Session = Depends(get_db)
):
    # Retrieve user by username
    current_user = db.query(models.User).filter(models.User.username == username).first()
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")

    return db.query(models.Screening).filter(models.Screening.user_id == current_user.id).all()

@app.get("/users/me", response_model=schemas.User)
async def read_users_me(username: str, db: Session = Depends(get_db)):
    # Retrieve user by username
    current_user = db.query(models.User).filter(models.User.username == username).first()
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")

    return current_user

@app.get("/user/statistics/", response_model=dict)
def get_user_statistics(
    username: str,
    db: Session = Depends(get_db)
):
    # Get user
    user = db.query(models.User).filter(models.User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user's screenings
    screenings = db.query(models.Screening).filter(
        models.Screening.user_id == user.id
    ).all()
    
    # Calculate statistics
    total_screenings = len(screenings)
    if total_screenings == 0:
        return {
            "total_screenings": 0,
            "benign_percentage": 0,
            "malignant_percentage": 0,
            "average_confidence": 0,
            "last_screening_date": None
        }
    
    benign_count = sum(1 for s in screenings if s.result == "Benign")
    malignant_count = total_screenings - benign_count
    
    stats = {
        "total_screenings": total_screenings,
        "benign_percentage": (benign_count / total_screenings) * 100,
        "malignant_percentage": (malignant_count / total_screenings) * 100,
        "average_confidence": sum(s.confidence for s in screenings) / total_screenings * 100,
        "last_screening_date": max(s.created_at for s in screenings)
    }
    
    return stats