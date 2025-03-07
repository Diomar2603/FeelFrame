from fastapi import APIRouter
from app.models.user import User
from app.services.user_service import UserService

router = APIRouter()

@router.post("/users/", response_model=User)
def create_user(user: User):
    return UserService.add_user(user)

@router.get("/users/", response_model=list[User])
def get_users():
    return UserService.list_users()
