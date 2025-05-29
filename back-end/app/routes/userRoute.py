from fastapi import APIRouter
from app.models.user import User
from app.services.userService import UserService

router = APIRouter(prefix="/users", tags=["Users"])

@router.post("/", response_model=User)
def create_user(user: User):
    return UserService.add_user(user)

@router.get("/", response_model=list[User])
def get_users():
    return UserService.list_users()
