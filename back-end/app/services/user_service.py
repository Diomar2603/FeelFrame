from app.models.user import User

class UserService:
    users = []

    @classmethod
    def add_user(cls, user: User):
        cls.users.append(user)
        return user

    @classmethod
    def list_users(cls):
        return cls.users
