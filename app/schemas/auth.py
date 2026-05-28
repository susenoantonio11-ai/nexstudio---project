"""Pydantic schemas for auth endpoints."""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional


class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: Optional[str] = None
    tier: str
    is_active: bool
    is_verified: bool

    class Config:
        from_attributes = True


TokenResponse.model_rebuild()
