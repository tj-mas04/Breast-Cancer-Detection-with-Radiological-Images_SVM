from pydantic import BaseModel, EmailStr
from datetime import datetime
from typing import Optional

class UserBase(BaseModel):
    username: str
    email: EmailStr

class UserLogin(BaseModel):
    username: str
    password: str
class UserCreate(UserBase):
    password: str

class User(UserBase):
    id: int
    is_active: bool

    class Config:
        from_attributes = True

class ScreeningBase(BaseModel):
    result: str
    confidence: float

class ScreeningCreate(ScreeningBase):
    pass

class Screening(ScreeningBase):
    id: int
    user_id: int
    image_path: str
    created_at: datetime

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str