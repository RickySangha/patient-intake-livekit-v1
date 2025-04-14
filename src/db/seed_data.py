import asyncio
import argparse
import datetime
import logging
import os
import uuid
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

from google.cloud import firestore

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("seed_data")

# Import types for reference (not used directly in seeding)
# This helps ensure our seed data is compatible with the models
try:
    from custom_types.patient_context import PatientContext
    from custom_types.complaint_details import (
        ChestPainDetails,
        CoughDetails,
        BackPainDetails,
    )
    from custom_types.call_details import CallDetails

    logger.info("Successfully imported type models for reference")
except ImportError:
    logger.warning(
        "Could not import type models - this is expected if running from a different directory"
    )

# Constants for collection names
DOCTORS_COLLECTION = "doctors"
PATIENTS_COLLECTION = "patients"
APPOINTMENTS_COLLECTION = "appointments"
CALLS_COLLECTION = "calls"
CALL_ATTEMPTS_COLLECTION = "call_attempts"
USERS_COLLECTION = "users"
CLINICS_COLLECTION = "clinics"

# Load environment variables
load_dotenv()

# Global Firestore AsyncClient
_db_client: Optional[firestore.AsyncClient] = None


async def initialize_firestore() -> firestore.AsyncClient:
    """Initialize and return Firestore AsyncClient."""
    global _db_client

    if _db_client is not None:
        return _db_client

    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if not key_path:
        raise RuntimeError(
            "GOOGLE_APPLICATION_CREDENTIALS environment variable not set"
        )

    if not os.path.exists(key_path):
        raise RuntimeError(f"Service account key file not found at: {key_path}")

    try:
        DATABASE_ID = "medical-agent"  # Can be changed to environment variable
        _db_client = firestore.AsyncClient(database=DATABASE_ID)
        logger.info(f"Firestore AsyncClient initialized (database='{DATABASE_ID}')")
        return _db_client
    except Exception as e:
        logger.exception(f"Failed to initialize Firestore: {e}")
        raise


async def clear_collection(collection_name: str) -> None:
    """Clear all documents in a collection."""
    db = await initialize_firestore()
    batch_size = 100  # Firestore limits batch operations

    logger.info(f"Clearing collection: {collection_name}")

    # Get all documents in collection
    docs = db.collection(collection_name).limit(batch_size).stream()

    deleted = 0
    async for doc_page in docs:
        # Delete document
        await db.collection(collection_name).document(doc_page.id).delete()
        deleted += 1

    logger.info(f"Deleted {deleted} documents from {collection_name}")


async def clear_subcollections(
    parent_collection: str, document_id: str, subcollections: List[str]
) -> None:
    """Clear specific subcollections under a document."""
    db = await initialize_firestore()

    for subcollection in subcollections:
        path = f"{parent_collection}/{document_id}/{subcollection}"
        logger.info(f"Clearing subcollection: {path}")

        # Get all documents in subcollection
        docs = db.collection(path).stream()

        deleted = 0
        async for doc in docs:
            # Delete document
            await db.collection(path).document(doc.id).delete()
            deleted += 1

        logger.info(f"Deleted {deleted} documents from {path}")


async def create_clinics() -> Dict[str, str]:
    """Create sample clinic data and return mapping of names to IDs."""
    db = await initialize_firestore()
    clinics = {}

    clinic_data = [
        {
            "name": "Surrey Medical Centre",
            "address": "123 Main Street",
            "city": "Surrey",
            "province": "BC",
            "postalCode": "V3T 0A1",
            "phoneNumber": "604-555-1234",
            "email": "info@surreymedical.com",
            "doctorIds": [],
            "patientIds": [],
            "open_settings": {
                "agent_name": "Amy",
                "auto_call_enabled": True,
                "auto_call_settings": {
                    "disabled_appointment_types": ["prescription_refill"]
                },
            },
            "doctor_only_settings": {"rules": {"specific_patient_ids": []}},
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        {
            "name": "Vancouver Health Clinic",
            "address": "456 Oak Avenue",
            "city": "Vancouver",
            "province": "BC",
            "postalCode": "V6B 2T5",
            "phoneNumber": "604-555-5678",
            "email": "contact@vanhealthclinic.com",
            "doctorIds": [],
            "patientIds": [],
            "open_settings": {"agent_name": "Sam", "auto_call_enabled": False},
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
    ]

    logger.info("Creating clinic data...")
    for clinic in clinic_data:
        # Use consistent IDs for development/testing
        clinic_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, clinic["name"]))
        await db.collection(CLINICS_COLLECTION).document(clinic_id).set(clinic)
        clinics[clinic["name"]] = clinic_id
        logger.info(f"Created clinic: {clinic['name']} with ID: {clinic_id}")

    return clinics


async def create_doctors(clinic_ids: Dict[str, str]) -> Dict[str, str]:
    """Create sample doctor data and return mapping of names to IDs."""
    db = await initialize_firestore()
    doctors = {}

    doctor_data = [
        {
            "firstName": "John",
            "lastName": "Smith",
            "email": "dr.smith@surreymedical.com",
            "clinicId": clinic_ids["Surrey Medical Centre"],
            "specialty": "Family Medicine",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        {
            "firstName": "Emily",
            "lastName": "Jones",
            "email": "dr.jones@vanhealthclinic.com",
            "clinicId": clinic_ids["Vancouver Health Clinic"],
            "specialty": "Internal Medicine",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
    ]

    logger.info("Creating doctor data...")
    for doctor in doctor_data:
        # Use consistent IDs for development/testing
        doctor_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS, f"{doctor['firstName']}.{doctor['lastName']}"
            )
        )
        await db.collection(DOCTORS_COLLECTION).document(doctor_id).set(doctor)
        doctors[f"{doctor['firstName']} {doctor['lastName']}"] = doctor_id

        # Update clinic with doctor ID
        clinic_ref = db.collection(CLINICS_COLLECTION).document(doctor["clinicId"])
        await clinic_ref.update(
            {
                "doctorIds": firestore.ArrayUnion([doctor_id]),
                "updatedAt": firestore.SERVER_TIMESTAMP,
            }
        )

        logger.info(
            f"Created doctor: {doctor['firstName']} {doctor['lastName']} with ID: {doctor_id}"
        )

    return doctors


async def create_patients(
    doctor_ids: Dict[str, str], clinic_ids: Dict[str, str]
) -> Dict[str, str]:
    """Create sample patient data and return mapping of names to IDs."""
    db = await initialize_firestore()
    patients = {}

    patient_data = [
        {
            "firstName": "Michael",
            "lastName": "Johnson",
            "age": 45,
            "gender": "male",
            "date_of_birth": datetime.datetime(1980, 5, 12),
            "phone_number": "604-555-9876",
            "email": "michael.johnson@example.com",
            "doctorId": doctor_ids["John Smith"],
            "recentComplaints": [],  # Will be updated with references later
            "medicalSummary": {
                "knownConditions": ["hypertension", "high cholesterol"],
                "lastUpdated": firestore.SERVER_TIMESTAMP,
            },
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        {
            "firstName": "Sarah",
            "lastName": "Williams",
            "age": 32,
            "gender": "female",
            "date_of_birth": datetime.datetime(1993, 8, 27),
            "phone_number": "604-555-4321",
            "email": "sarah.williams@example.com",
            "doctorId": doctor_ids["Emily Jones"],
            "recentComplaints": [],
            "medicalSummary": {
                "knownConditions": ["asthma"],
                "lastUpdated": firestore.SERVER_TIMESTAMP,
            },
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
    ]

    logger.info("Creating patient data...")
    for patient in patient_data:
        # Use consistent IDs for development/testing
        patient_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS, f"{patient['firstName']}.{patient['lastName']}"
            )
        )
        await db.collection(PATIENTS_COLLECTION).document(patient_id).set(patient)
        patients[f"{patient['firstName']} {patient['lastName']}"] = patient_id

        # Determine clinic ID based on doctor
        clinic_id = None
        doctor_id = patient["doctorId"]

        # Loop through the clinics to find which one has this doctor
        for clinic_name, c_id in clinic_ids.items():
            clinic_ref = db.collection(CLINICS_COLLECTION).document(c_id)
            clinic_doc = await clinic_ref.get()

            # Fixed: Use clinic_doc.get() properly
            if clinic_doc.exists:
                doctor_ids_array = clinic_doc.get("doctorIds")
                if doctor_ids_array and doctor_id in doctor_ids_array:
                    clinic_id = c_id
                    break

        # Update clinic with patient ID if found
        if clinic_id:
            clinic_ref = db.collection(CLINICS_COLLECTION).document(clinic_id)
            await clinic_ref.update(
                {
                    "patientIds": firestore.ArrayUnion([patient_id]),
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                }
            )

        logger.info(
            f"Created patient: {patient['firstName']} {patient['lastName']} with ID: {patient_id}"
        )

    return patients


async def create_medical_history(patient_ids: Dict[str, str]) -> None:
    """Create sample medical history data for patients."""
    db = await initialize_firestore()

    # Medical history for Michael Johnson
    michael_history = {
        "recordedAt": datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=180),
        "source": "doctor_input",
        "past_conditions": ["hypertension", "high cholesterol"],
        "surgeries": ["appendectomy (2010)"],
        "medications": ["lisinopril 10mg daily", "simvastatin 20mg daily"],
        "allergies": ["penicillin"],
        "notes": "Patient has been managing hypertension well with current medication.",
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    # Medical history for Sarah Williams
    sarah_history = {
        "recordedAt": datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(days=90),
        "source": "doctor_input",
        "past_conditions": ["asthma", "seasonal allergies"],
        "surgeries": [],
        "medications": [
            "albuterol inhaler as needed",
            "loratadine 10mg daily (seasonal)",
        ],
        "allergies": ["sulfa drugs", "cats"],
        "notes": "Patient's asthma well-controlled with current treatment plan.",
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    logger.info("Creating medical history data...")

    # Create Michael's medical history
    michael_id = patient_ids["Michael Johnson"]
    michael_history_ref = db.collection(
        f"{PATIENTS_COLLECTION}/{michael_id}/medicalHistory"
    ).document()
    await michael_history_ref.set(michael_history)
    logger.info(
        f"Created medical history for Michael Johnson with ID: {michael_history_ref.id}"
    )

    # Create Sarah's medical history
    sarah_id = patient_ids["Sarah Williams"]
    sarah_history_ref = db.collection(
        f"{PATIENTS_COLLECTION}/{sarah_id}/medicalHistory"
    ).document()
    await sarah_history_ref.set(sarah_history)
    logger.info(
        f"Created medical history for Sarah Williams with ID: {sarah_history_ref.id}"
    )


async def create_complaints(patient_ids: Dict[str, str]) -> None:
    """Create sample complaint data for patients."""
    db = await initialize_firestore()

    # Past complaints for Michael Johnson
    michael_complaints = [
        {
            "type": "chest_pain",
            "recordedAt": datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=240),
            "callId": f"call_{uuid.uuid4()}",
            "details": {
                "location": "center of chest",
                "quality": "pressure-like",
                "severity": "5/10",
                "associated_symptoms": "shortness of breath",
                "relieving_factors": "rest",
                "aggravating_factors": "exercise, stress",
            },
            "resolved": True,
            "followUpNotes": "ECG normal. Stress test negative. Likely musculoskeletal in origin.",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        {
            "type": "back_pain",
            "recordedAt": datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=60),
            "callId": f"call_{uuid.uuid4()}",
            "details": {
                "location": "lower back",
                "quality": "dull, aching",
                "severity": "6/10",
                "duration": "3 weeks",
                "onset": "gradual",
                "aggravating_factors": "sitting for long periods",
                "relieving_factors": "stretching, heat",
                "associated_symptoms": "none",
            },
            "resolved": False,
            "followUpNotes": "Recommended physical therapy. Follow-up in 2 months.",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
    ]

    # Past complaints for Sarah Williams
    sarah_complaints = [
        {
            "type": "cough",
            "recordedAt": datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=120),
            "callId": f"call_{uuid.uuid4()}",
            "details": {
                "duration": "2 weeks",
                "cough_type": "productive",
                "triggers_patterns": "worse at night",
                "associated_symptoms": "mild fever, fatigue",
                "severity": "moderate, interrupts sleep",
            },
            "resolved": True,
            "followUpNotes": "Bronchitis. Resolved with antibiotics.",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
    ]

    logger.info("Creating complaint data...")

    # Create Michael's complaints
    michael_id = patient_ids["Michael Johnson"]
    recent_complaints = []
    for complaint in michael_complaints:
        complaint_ref = db.collection(
            f"{PATIENTS_COLLECTION}/{michael_id}/complaints"
        ).document()
        await complaint_ref.set(complaint)
        logger.info(
            f"Created {complaint['type']} complaint for Michael Johnson with ID: {complaint_ref.id}"
        )

        # Save the most recent complaint for the patient document
        recent_complaints.append(
            {
                "id": complaint_ref.id,
                "type": complaint["type"],
                "recordedAt": complaint["recordedAt"],
                "resolved": complaint["resolved"],
            }
        )

    # Update Michael's patient record with recent complaints
    await db.collection(PATIENTS_COLLECTION).document(michael_id).update(
        {
            "recentComplaints": sorted(
                recent_complaints, key=lambda x: x["recordedAt"], reverse=True
            )[:3],
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
    )

    # Create Sarah's complaints
    sarah_id = patient_ids["Sarah Williams"]
    recent_complaints = []
    for complaint in sarah_complaints:
        complaint_ref = db.collection(
            f"{PATIENTS_COLLECTION}/{sarah_id}/complaints"
        ).document()
        await complaint_ref.set(complaint)
        logger.info(
            f"Created {complaint['type']} complaint for Sarah Williams with ID: {complaint_ref.id}"
        )

        # Save for the patient document
        recent_complaints.append(
            {
                "id": complaint_ref.id,
                "type": complaint["type"],
                "recordedAt": complaint["recordedAt"],
                "resolved": complaint["resolved"],
            }
        )

    # Update Sarah's patient record with recent complaints
    await db.collection(PATIENTS_COLLECTION).document(sarah_id).update(
        {
            "recentComplaints": sorted(
                recent_complaints, key=lambda x: x["recordedAt"], reverse=True
            )[:3],
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }
    )


async def create_appointments(
    patient_ids: Dict[str, str], doctor_ids: Dict[str, str], clinic_ids: Dict[str, str]
) -> Dict[str, str]:
    """Create sample appointments and return mapping of descriptions to IDs."""
    db = await initialize_firestore()
    appointments = {}

    # Current time for reference
    now = datetime.datetime.now(datetime.timezone.utc)

    appointment_data = [
        {
            "patientId": patient_ids["Michael Johnson"],
            "doctorId": doctor_ids["John Smith"],
            "clinicId": clinic_ids["Surrey Medical Centre"],
            "appointmentType": "follow_up",
            "appointmentDate": now
            + datetime.timedelta(days=3, hours=10),  # 3 days from now, 10am
            "reason": "Follow-up for back pain",
            "autoCallEnabled": True,
            "status": "scheduled",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        {
            "patientId": patient_ids["Sarah Williams"],
            "doctorId": doctor_ids["Emily Jones"],
            "clinicId": clinic_ids["Vancouver Health Clinic"],
            "appointmentType": "annual_checkup",
            "appointmentDate": now
            + datetime.timedelta(days=7, hours=14),  # 7 days from now, 2pm
            "reason": "Annual physical examination",
            "autoCallEnabled": True,
            "status": "scheduled",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
    ]

    logger.info("Creating appointment data...")
    for i, appointment in enumerate(appointment_data):
        # For testing, use predictable IDs
        appt_id = f"appt{i+1}"
        await db.collection(APPOINTMENTS_COLLECTION).document(appt_id).set(appointment)

        description = f"{appointment['appointmentType']} for {appointment['patientId']}"
        appointments[description] = appt_id
        logger.info(f"Created appointment: {description} with ID: {appt_id}")

    return appointments


async def create_sample_calls(
    patient_ids: Dict[str, str], doctor_ids: Dict[str, str]
) -> None:
    """Create sample call records."""
    db = await initialize_firestore()

    # Sample completed calls
    call_data = [
        {
            "patient_id": patient_ids["Michael Johnson"],
            "doctor_id": doctor_ids["John Smith"],
            "call_start_time": datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=2, hours=1),
            "call_end_time": datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=2),
            "duration_minutes": 14.5,
            "call_status": "completed",
            "consent_given": True,
            "collectedComplaintId": None,  # Would refer to a complaint document
            "collectedMedicalHistoryId": None,  # Would refer to a medical history document
            "final_summary": "Patient reported continued back pain, especially after sitting for extended periods. Currently taking over-the-counter pain medication with some relief.",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
        {
            "patient_id": patient_ids["Sarah Williams"],
            "doctor_id": doctor_ids["Emily Jones"],
            "call_start_time": datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=5, hours=2),
            "call_end_time": datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(days=5, hours=1, minutes=45),
            "duration_minutes": 15.0,
            "call_status": "completed",
            "consent_given": True,
            "collectedComplaintId": None,
            "collectedMedicalHistoryId": None,
            "final_summary": "Pre-appointment check for annual physical. Patient reports feeling well, no new concerns.",
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
    ]

    logger.info("Creating call records...")
    for call in call_data:
        call_id = f"call_{uuid.uuid4()}"
        await db.collection(CALLS_COLLECTION).document(call_id).set(call)
        logger.info(f"Created call record with ID: {call_id}")


async def main(args):
    """Main function to seed the database."""
    try:
        # Clear existing collections if requested
        if args.clear:
            for collection in [
                CLINICS_COLLECTION,
                DOCTORS_COLLECTION,
                PATIENTS_COLLECTION,
                APPOINTMENTS_COLLECTION,
                CALLS_COLLECTION,
            ]:
                await clear_collection(collection)

            logger.info("All collections cleared successfully")

        # Create data in order of dependencies
        clinic_ids = await create_clinics()
        doctor_ids = await create_doctors(clinic_ids)
        patient_ids = await create_patients(doctor_ids, clinic_ids)

        # Add subcollection data
        await create_medical_history(patient_ids)
        await create_complaints(patient_ids)

        # Create appointments and calls that reference the above
        appointment_ids = await create_appointments(patient_ids, doctor_ids, clinic_ids)
        await create_sample_calls(patient_ids, doctor_ids)

        logger.info("Database seeded successfully!")

    except Exception as e:
        logger.exception(f"Error seeding database: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed the Firestore database with sample data"
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing data before seeding"
    )
    args = parser.parse_args()

    # Check if environment variables are set
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        logger.error("GOOGLE_APPLICATION_CREDENTIALS environment variable not set")
        exit(1)

    # Run the async main function
    asyncio.run(main(args))
