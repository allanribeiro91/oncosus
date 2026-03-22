import json
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "data/synthetic_data/paciente_sintetico.json"


def load_patient():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def find_patient_by_name(name: str):
    patients = load_patient()

    for patient in patients:
        if name.lower() in patient.get("nome", "").lower():
            return patient

    return None

def find_patient_by_id(patient_id: int):
    patients = load_patient()

    for patient in patients:
        if patient.get("id") == patient_id:
            return patient

    return None