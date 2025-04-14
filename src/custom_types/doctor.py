from pydantic import BaseModel
from typing import Optional


class Doctor(BaseModel):
    id: Optional[str] = None
    firstName: Optional[str] = None
    lastName: Optional[str] = None
    email: Optional[str] = None
    clinicId: Optional[str] = None
    specialty: Optional[str] = None
