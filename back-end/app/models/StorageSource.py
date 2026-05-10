from enum import Enum


class StorageSource(str, Enum):
    LOCAL    = "local"
    FIREBASE = "firebase"
    OTHERS   = "others"
