import logging
import os
import uuid
from datetime import datetime, timezone
from google.cloud import firestore
from typing import Optional, Dict, Any, List, Union

# Updated imports for the new types directory structure
from custom_types.patient_context import PatientContext
from custom_types.clinic import Clinic
from custom_types.doctor import Doctor
from custom_types.appointment import Appointment
from custom_types.complaint_details import (
    ComplaintDetails,
    ChestPainDetails,
    CoughDetails,
    BackPainDetails,
)
from custom_types.call_details import CallDetails

logger = logging.getLogger(__name__)

# Global variable to hold the initialized AsyncClient
_db_client: Optional[firestore.AsyncClient] = None
_initialized = False

# Define collection name constants centrally
DOCTORS_COLLECTION = "doctors"
PATIENTS_COLLECTION = "patients"
APPOINTMENTS_COLLECTION = "appointments"
CALLS_COLLECTION = "calls"
CALL_ATTEMPTS_COLLECTION = "call_attempts"
USERS_COLLECTION = "users"
CLINICS_COLLECTION = "clinics"


def _initialize_firestore_internal() -> bool:
    """
    Internal function to perform the actual Firestore initialization.
    Uses the FIRESTORE_SERVICE_ACCOUNT_KEY environment variable.

    Returns:
        True if initialization was successful, False otherwise.
    """
    global _db_client, _initialized  # Allow modification of globals

    # Check again in case called concurrently, though GIL helps in CPython
    if _initialized:
        return True

    key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

    if not key_path:
        logger.error(
            "Firestore service account key path env var GOOGLE_APPLICATION_CREDENTIALS not set."
        )
        return False

    if not os.path.exists(key_path):
        logger.error(f"Service account key file not found at: {key_path}")
        return False

    try:
        DATABASE_ID = os.getenv("FIRESTORE_DATABASE_ID", "medical-agent")
        _db_client = firestore.AsyncClient(database=DATABASE_ID)

        _initialized = True
        logger.info(
            f"Firestore AsyncClient obtained successfully (database='{DATABASE_ID}')."
        )
        return True
    except Exception as e:
        logger.exception(
            f"Failed to initialize Firestore AsyncClient: {e}", exc_info=True
        )
        _db_client = None
        _initialized = False
        return False


def get_firestore_client() -> firestore.AsyncClient:
    """
    Returns the initialized Firestore AsyncClient instance.
    Initializes the client on the first call if not already done.

    Raises:
        RuntimeError: If Firestore initialization fails during the first call, or if attempted usage occurs after a failed initialization.
    """
    global _initialized, _db_client  # Access globals

    if not _initialized:
        logger.info("Firestore client not initialized. Attempting initialization...")
        if not _initialize_firestore_internal():
            # Initialization failed, raise error to prevent usage
            raise RuntimeError("Firestore initialization failed. Cannot get client.")
    if _db_client is None:
        raise RuntimeError(
            "Firestore client is unexpectedly None after initialization check."
        )

    return _db_client


async def _fetch_document_by_id(
    collection: str, doc_id: str, model: type
) -> Optional[object]:
    """Helper to fetch a single document and parse it into a Pydantic model."""
    if not doc_id:
        logger.warning(f"No document ID provided for collection {collection}.")
        return None
    try:
        db = get_firestore_client()
        doc_ref = db.collection(collection).document(doc_id)
        doc_snapshot = await doc_ref.get()
        if doc_snapshot.exists:
            data_dict = doc_snapshot.to_dict()
            if data_dict:
                # Pass id explicitly as it's the document id
                return model(id=doc_id, **data_dict)
            else:
                logger.warning(
                    f"Document {doc_id} in {collection} exists but has no data."
                )
                return model(id=doc_id)  # Return model with just ID if doc is empty
        else:
            logger.warning(f"Document {doc_id} not found in collection {collection}.")
            return None
    except Exception as e:
        logger.exception(
            f"Error fetching document {doc_id} from {collection}: {e}", exc_info=True
        )
        return None


async def fetch_doctor(doctor_id: Optional[str]) -> Optional[Doctor]:
    """Fetches a doctor document by ID."""
    if not doctor_id:
        return None
    return await _fetch_document_by_id(DOCTORS_COLLECTION, doctor_id, Doctor)


async def fetch_clinic(clinic_id: Optional[str]) -> Optional[Clinic]:
    """Fetches a clinic document by ID."""
    if not clinic_id:
        return None
    return await _fetch_document_by_id(CLINICS_COLLECTION, clinic_id, Clinic)


async def fetch_appointment(appointment_id: Optional[str]) -> Optional[Appointment]:
    """Fetches an appointment document by ID."""
    if not appointment_id:
        return None
    return await _fetch_document_by_id(
        APPOINTMENTS_COLLECTION, appointment_id, Appointment
    )


async def fetch_and_prepare_patient_context(
    patient_id: Optional[str],
) -> PatientContext:
    """
    Fetches core patient data from Firestore based on ID
    and prepares the PatientContext object. Initializes embedded CallDetails
    with relevant IDs. Complaint/History are initialized as None.
    Returns a PatientContext instance.
    """
    patient_context: PatientContext

    if not patient_id:
        logger.warning("No patient ID provided. Initializing default PatientContext.")
        patient_context = PatientContext(id=None)
        return patient_context

    try:
        db = get_firestore_client()
        logger.info(f"Attempting to fetch core data for patient ID: {patient_id}")
        doc_ref = db.collection(PATIENTS_COLLECTION).document(patient_id)
        doc_snapshot = await doc_ref.get()

        if not doc_snapshot.exists:
            logger.warning(
                f"No patient document found for ID: {patient_id}. Initializing default PatientContext with ID."
            )
            patient_context = PatientContext(id=patient_id)
            return patient_context

        fetched_patient_data_dict = doc_snapshot.to_dict()
        if fetched_patient_data_dict is None:
            logger.error(
                f"Fetched data for patient ID {patient_id} is None. Document might be empty."
            )
            patient_context = PatientContext(
                id=patient_id, first_name="Error", last_name="Empty Doc"
            )
            return patient_context

        logger.info(f"Successfully fetched core data for patient ID: {patient_id}")

        try:
            # Map field names to match PatientContext model
            # Firestore typically uses camelCase, our models use snake_case
            if "firstName" in fetched_patient_data_dict:
                fetched_patient_data_dict["first_name"] = fetched_patient_data_dict.pop(
                    "firstName"
                )
            if "lastName" in fetched_patient_data_dict:
                fetched_patient_data_dict["last_name"] = fetched_patient_data_dict.pop(
                    "lastName"
                )
            if "doctorId" in fetched_patient_data_dict:
                doctor_id = fetched_patient_data_dict.pop("doctorId")
                # We'll link the doctor later, just store ID for now

            # Create the base PatientContext instance
            patient_context = PatientContext(id=patient_id, **fetched_patient_data_dict)

            # Set doctor_id in call_details if available
            if "doctor_id" in locals():
                patient_context.call_details.doctor_id = doctor_id

            # Always set patient_id in call_details
            patient_context.call_details.patient_id = patient_id

            logger.info(
                f"Initialized base PatientContext from Firestore for ID: {patient_context.id}"
            )
            return patient_context

        except Exception as parse_exc:
            logger.exception(
                f"Error parsing core fetched Firestore data into PatientContext for ID '{patient_id}': {parse_exc}. Using defaults.",
                exc_info=True,
            )
            patient_context = PatientContext(
                id=patient_id,
                first_name="Error Loading",
                last_name="Patient",
            )
            return patient_context

    except RuntimeError as db_err:
        logger.critical(f"Firestore client error during patient fetch: {db_err}")
        raise
    except Exception as fetch_exc:
        logger.exception(
            f"Error fetching patient data for patient ID '{patient_id}': {fetch_exc}",
            exc_info=True,
        )
        patient_context = PatientContext(
            id=patient_id,
            first_name="Error Loading",
            last_name="Patient (Fetch Err)",
        )
        return patient_context


async def initialize_context_from_appointment(appointment_id: str) -> PatientContext:
    """
    Initializes the full PatientContext by fetching the appointment,
    and then the related patient, doctor, and clinic documents.
    """
    logger.info(f"Initializing PatientContext from appointment ID: {appointment_id}")

    # 1. Fetch Appointment
    appointment = await fetch_appointment(appointment_id)
    if not appointment:
        logger.error(
            f"Failed to fetch appointment {appointment_id}. Returning empty context."
        )
        # Return a default context, perhaps with an indicator of the error source
        return PatientContext(
            id=None, first_name="Error", last_name="Appointment Not Found"
        )

    # 2. Extract IDs from Appointment
    patient_id = appointment.patientId
    doctor_id = appointment.doctorId
    clinic_id = appointment.clinicId

    # 3. Fetch Core Patient Data
    # Reuse the existing function to get the base patient details
    patient_context = await fetch_and_prepare_patient_context(patient_id)

    # If patient fetch failed, patient_context might have default/error values,
    # but we can still proceed to fetch doctor/clinic if IDs exist.
    if not patient_context.id and patient_id:
        logger.warning(
            f"Core patient data fetch failed for ID {patient_id}, but proceeding with appointment context."
        )
        # Ensure the ID from the appointment is set if patient fetch failed entirely
        patient_context.id = patient_id

    # 4. Fetch Doctor Data
    doctor = await fetch_doctor(doctor_id)
    if not doctor and doctor_id:
        logger.warning(
            f"Doctor {doctor_id} not found for appointment {appointment_id}."
        )
        # Optionally create a placeholder Doctor object if needed downstream
        # doctor = Doctor(id=doctor_id, firstName="Unknown", lastName="Doctor")

    # 5. Fetch Clinic Data
    clinic = await fetch_clinic(clinic_id)
    if not clinic and clinic_id:
        logger.warning(
            f"Clinic {clinic_id} not found for appointment {appointment_id}."
        )
        # Optionally create a placeholder Clinic object
        # clinic = Clinic(id=clinic_id, name="Unknown Clinic")

    # 6. Assemble PatientContext
    # Update the existing patient_context with the fetched related objects
    patient_context.appointment = appointment
    patient_context.doctor = doctor
    patient_context.clinic = clinic

    # Re-run __init__ logic implicitly or explicitly if needed for ID syncing
    # Pydantic's __init__ was already called, but we can manually sync IDs if needed
    # For example, ensure call_details has the doctor_id from the appointment context:
    if doctor and doctor.id:
        patient_context.call_details.doctor_id = doctor.id
    # Ensure call_details has the patient_id
    if patient_context.id:
        patient_context.call_details.patient_id = patient_context.id

    logger.info(
        f"Successfully initialized PatientContext for appointment {appointment_id} (Patient: {patient_context.id}, Doctor: {doctor.id if doctor else 'None'}, Clinic: {clinic.id if clinic else 'None'})"
    )
    return patient_context


async def fetch_patient_history(patient_id: str) -> Dict[str, Any]:
    """
    Fetches patient medical history and recent complaints for agent context.

    Returns structured dictionary containing:
    - medical_history: Most recent medical history record
    - recent_complaints: List of recent complaints (up to 3, sorted by date)
    """
    if not patient_id:
        logger.warning("No patient ID provided for fetching history")
        return {"medical_history": None, "recent_complaints": []}

    try:
        db = get_firestore_client()
        result = {"medical_history": None, "recent_complaints": []}

        # Fetch most recent medical history
        medical_history_query = (
            db.collection(f"{PATIENTS_COLLECTION}/{patient_id}/medicalHistory")
            .order_by("recordedAt", direction=firestore.Query.DESCENDING)
            .limit(1)
        )

        async for doc in medical_history_query.stream():
            history_data = doc.to_dict()
            if history_data:
                result["medical_history"] = {
                    "id": doc.id,
                    "recordedAt": history_data.get("recordedAt"),
                    "past_conditions": history_data.get("past_conditions", []),
                    "surgeries": history_data.get("surgeries", []),
                    "medications": history_data.get("medications", []),
                    "allergies": history_data.get("allergies", []),
                    "notes": history_data.get("notes"),
                }
                break

        # Fetch recent complaints (up to 3)
        complaints_query = (
            db.collection(f"{PATIENTS_COLLECTION}/{patient_id}/complaints")
            .order_by("recordedAt", direction=firestore.Query.DESCENDING)
            .limit(3)
        )

        async for doc in complaints_query.stream():
            complaint_data = doc.to_dict()
            if complaint_data:
                # Get basic complaint info
                complaint_type = complaint_data.get("type")
                details = complaint_data.get("details", {})

                formatted_complaint = {
                    "id": doc.id,
                    "type": complaint_type,
                    "recordedAt": complaint_data.get("recordedAt"),
                    "resolved": complaint_data.get("resolved", False),
                    "details": details,
                }

                result["recent_complaints"].append(formatted_complaint)

        return result

    except Exception as e:
        logger.exception(f"Error fetching patient history for {patient_id}: {e}")
        return {"medical_history": None, "recent_complaints": []}


async def prepare_initial_patient_context(appointment_id: str) -> PatientContext:
    """
    Prepares the full PatientContext for the agent by:
    1. Fetching appointment, patient, doctor, and clinic info
    2. Loading the patient's medical history
    3. Loading recent complaints
    4. Assembling CallDetails with current datetime

    This enriched context provides the agent with all necessary
    historical information to handle the call effectively.
    """
    try:
        # First get the basic context from appointment
        context = await initialize_context_from_appointment(appointment_id)

        # If we couldn't get a valid patient ID, return the basic context
        if not context.id:
            logger.warning(
                "No valid patient ID in context, skipping history enrichment"
            )
            return context

        # Fetch patient history and recent complaints
        patient_id = context.id
        history_data = await fetch_patient_history(patient_id)

        # Add history data to the context
        if history_data.get("medical_history"):
            # Create medical_history dict if it doesn't exist
            if not context.medical_history:
                context.medical_history = {}

            # Add historical medical data to context
            med_history = history_data["medical_history"]
            context.medical_history.update(
                {
                    "past_conditions": med_history.get("past_conditions", []),
                    "surgeries": med_history.get("surgeries", []),
                    "medications": med_history.get("medications", []),
                    "allergies": med_history.get("allergies", []),
                    "history_id": med_history.get("id"),
                    "last_updated": med_history.get("recordedAt"),
                }
            )

        # Add historical complaint data
        # We don't set complaint_details directly, as that would be the currently active complaint
        # Instead, we'll make the recent complaints available in the medical_history for reference
        recent_complaints = history_data.get("recent_complaints", [])
        if recent_complaints and context.medical_history:
            context.medical_history["recent_complaints"] = recent_complaints

        # Initialize call details with current time
        context.call_details.call_start_time = datetime.now(timezone.utc)
        context.call_details.call_status = "in_progress"

        # Generate a unique call_id if not already present
        if not context.call_details.call_id:
            context.call_details.call_id = f"call_{uuid.uuid4()}"

        # Set the appointment_id in call_details
        context.call_details.appointment_id = appointment_id

        logger.info(f"Successfully prepared enriched context for patient {patient_id}")
        return context

    except Exception as e:
        logger.exception(f"Error preparing initial patient context: {e}")
        # Return basic context or placeholder as fallback
        if "context" in locals() and context:
            return context
        return PatientContext(
            id=None, first_name="Error", last_name="Context Preparation Failed"
        )


async def save_complaint_data(
    patient_id: str, complaint_details: ComplaintDetails, call_id: str
):
    """Saves complaint details to a subcollection under the patient."""
    try:
        db = get_firestore_client()

        if not complaint_details or not hasattr(complaint_details, "type"):
            logger.warning(
                f"No valid complaint details to save for patient {patient_id}"
            )
            return None

        # Create new complaint document with a generated ID
        complaint_ref = db.collection(
            f"{PATIENTS_COLLECTION}/{patient_id}/complaints"
        ).document()

        # Convert complaint_details to dict, excluding the type field which we'll store separately
        if hasattr(complaint_details, "model_dump"):
            details_dict = complaint_details.model_dump(exclude={"type"})
        else:  # Fallback for older Pydantic versions
            details_dict = complaint_details.dict(exclude={"type"})

        # Prepare data for saving
        complaint_data = {
            "type": complaint_details.type,
            "recordedAt": datetime.now(timezone.utc),
            "callId": call_id,
            "details": details_dict,
            "resolved": False,
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }

        # Save the complaint document
        await complaint_ref.set(complaint_data)
        logger.info(
            f"Saved complaint {complaint_details.type} for patient {patient_id}"
        )

        # Update the patient's recentComplaints array
        patient_ref = db.collection(PATIENTS_COLLECTION).document(patient_id)

        # Create a summary for the recent complaints list
        complaint_summary = {
            "id": complaint_ref.id,
            "type": complaint_details.type,
            "recordedAt": complaint_data["recordedAt"],
            "resolved": False,
        }

        # Add to recentComplaints array, keeping only the most recent 3
        try:
            # Get current recentComplaints
            patient_doc = await patient_ref.get()
            if patient_doc.exists:
                current_complaints = patient_doc.get("recentComplaints") or []

                # Add new complaint at the beginning
                updated_complaints = [complaint_summary] + current_complaints

                # Keep only the 3 most recent
                if len(updated_complaints) > 3:
                    updated_complaints = updated_complaints[:3]

                # Update the patient document
                await patient_ref.update(
                    {
                        "recentComplaints": updated_complaints,
                        "updatedAt": firestore.SERVER_TIMESTAMP,
                    }
                )
                logger.info(f"Updated recentComplaints for patient {patient_id}")
        except Exception as e:
            logger.warning(
                f"Failed to update recentComplaints for patient {patient_id}: {e}"
            )

        # Return the new complaint ID for reference
        return complaint_ref.id

    except Exception as e:
        logger.exception(f"Failed to save complaint data for patient {patient_id}: {e}")
        return None


async def save_medical_history(patient_id: str, medical_history: dict, call_id: str):
    """Saves medical history to a subcollection under the patient."""
    try:
        db = get_firestore_client()

        if not medical_history:
            logger.warning(f"No medical history to save for patient {patient_id}")
            return None

        # Create new medical history document
        history_ref = db.collection(
            f"{PATIENTS_COLLECTION}/{patient_id}/medicalHistory"
        ).document()

        # Prepare data for saving
        history_data = {
            "recordedAt": datetime.now(timezone.utc),
            "source": "agent_call",
            "callId": call_id,
            "past_conditions": medical_history.get("past_conditions", []),
            "surgeries": medical_history.get("surgeries", []),
            "medications": medical_history.get("medications", []),
            "allergies": medical_history.get("allergies", []),
            "notes": medical_history.get("notes", ""),
            "createdAt": firestore.SERVER_TIMESTAMP,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        }

        # Save the medical history document
        await history_ref.set(history_data)
        logger.info(f"Saved medical history for patient {patient_id}")

        # Update the patient's medicalSummary for quick access
        patient_ref = db.collection(PATIENTS_COLLECTION).document(patient_id)

        # Create a summary for the patient document
        medical_summary = {
            "knownConditions": history_data["past_conditions"],
            "lastUpdated": history_data["recordedAt"],
        }

        try:
            # Update the patient document
            await patient_ref.update(
                {
                    "medicalSummary": medical_summary,
                    "updatedAt": firestore.SERVER_TIMESTAMP,
                }
            )
            logger.info(f"Updated medicalSummary for patient {patient_id}")
        except Exception as e:
            logger.warning(
                f"Failed to update medicalSummary for patient {patient_id}: {e}"
            )

        # Return the new history ID for reference
        return history_ref.id

    except Exception as e:
        logger.exception(
            f"Failed to save medical history for patient {patient_id}: {e}"
        )
        return None


async def save_call_details(call_details: CallDetails):
    """Saves the call details to the calls collection."""
    try:
        db = get_firestore_client()

        if not call_details.call_id:
            logger.warning("Missing call_id in call_details. Cannot save call details.")
            return False

        # Prepare call data for saving
        if hasattr(call_details, "model_dump"):
            call_data = call_details.model_dump(exclude={"call_id"})
        else:  # Fallback for older Pydantic versions
            call_data = call_details.dict(exclude={"call_id"})

        # Add timestamps
        call_data["createdAt"] = firestore.SERVER_TIMESTAMP
        call_data["updatedAt"] = firestore.SERVER_TIMESTAMP

        # Save to Firestore
        await db.collection(CALLS_COLLECTION).document(call_details.call_id).set(
            call_data
        )
        logger.info(
            f"Successfully saved call details for call ID: {call_details.call_id}"
        )
        return True

    except Exception as e:
        logger.exception(f"Failed to save call details: {e}")
        return False


async def update_call_with_collected_data(
    call_id: str,
    complaint_id: Optional[str] = None,
    history_id: Optional[str] = None,
    final_summary: Optional[str] = None,
):
    """Updates a call record with references to collected data."""
    try:
        db = get_firestore_client()

        update_data = {"updatedAt": firestore.SERVER_TIMESTAMP}

        if complaint_id:
            update_data["collectedComplaintId"] = complaint_id

        if history_id:
            update_data["collectedMedicalHistoryId"] = history_id

        if final_summary:
            update_data["final_summary"] = final_summary

        # Update call record
        await db.collection(CALLS_COLLECTION).document(call_id).update(update_data)
        logger.info(f"Updated call {call_id} with collected data references")
        return True

    except Exception as e:
        logger.exception(f"Failed to update call with collected data: {e}")
        return False
