# AI Medical Assistant Agent

## Project Description

This project implements an AI-powered medical assistant designed to interact with patients conversationally before their appointments. It gathers preliminary information, such as the reason for the visit and details about their symptoms, to streamline the consultation process for the medical staff. The agent uses various AI services for voice activity detection (VAD), speech-to-text (STT), language understanding (LLM), and text-to-speech (TTS).

## Architecture

The application is built using the `livekit-agents` Python framework, orchestrating interactions between different components:

1.  **Main Entrypoint (`src/main.py`):** Initializes the environment, sets up necessary plugins (VAD, STT, LLM, TTS), loads initial patient context from the database, creates the main `AgentSession`, and starts the initial agent (`EntryAgent`).
2.  **Agents (`src/agents/`):** Modular components responsible for specific parts of the conversation flow.
    - `BaseAgent`: Provides common functionalities or prompts.
    - `EntryAgent`: Handles the initial greeting and determines the primary reason for the call.
    - `IssueAgent` (and specific issue agents like `ChestPainAgent`, `CoughAgent`, `BackPainAgent`): Gather detailed information about the patient's specific complaint using LLM function calling.
    - `MedicalHistoryAgent`: Asks about the patient's medical history.
    - `EndCallAgent`: Concludes the conversation.
      Agents transition between each other based on the conversation context and LLM function calls.
3.  **Plugins (`livekit.plugins.*`):** Integrations with external AI services:
    - **VAD:** Silero VAD (`livekit.plugins.silero`) for detecting speech.
    - **STT:** Deepgram (`livekit.plugins.deepgram`) for transcribing speech to text.
    - **LLM:** Azure OpenAI (`livekit.plugins.openai`) for understanding patient responses and deciding the next steps (including function calls).
    - **TTS:** Deepgram (`livekit.plugins.deepgram`) for synthesizing the agent's voice. (Note: The code also includes `plugin_kokoro` for Kokoro TTS, though it wasn't actively used in `main.py`).
4.  **Database (`src/db/firestore_client.py`):** Uses Google Firestore to fetch initial patient, doctor, and clinic information (`PatientContext`) based on an appointment ID. It also likely stores the gathered information (though the saving part isn't fully shown in `main.py`).
5.  **Data Structures (`src/custom_types/`):** Pydantic models are used extensively (e.g., `PatientContext`, `ChestPainDetails`, `Clinic`, `Doctor`) to define the structure of data passed between components and stored in the database. This ensures type safety and clear data contracts.

## Setup and Installation

1.  **Create and Activate Virtual Environment:**

    ```bash
    python3.12 -m venv .venv
    source .venv/bin/activate
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Variables:**
    - Fill out the `example.env` file with the necessary API keys and configuration and rename it to `.env`
    - Ensure the `GOOGLE_APPLICATION_CREDENTIALS` environment variable is set correctly to point to your Google Cloud service account key file for Firestore access. This can be downloaded from the Google Cloud Console.

## Running the Application

Once the setup is complete and the virtual environment is active:

```bash
python src/main.py dev
```

or

```bash
python src/main.py console
```

## Development Notes

- **Data Models:** All data structures passed around (especially context and LLM function results) should be defined as Pydantic models in `src/custom_types/` for consistency.
