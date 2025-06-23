# create_json.py
import json
import uuid
import random
from datetime import datetime, timedelta
from pathlib import Path
import shutil

CREATED_JSON_DIR = Path("created_json")
if CREATED_JSON_DIR.exists():
    shutil.rmtree(CREATED_JSON_DIR)
CREATED_JSON_DIR.mkdir(exist_ok=True)

def random_ip():
    return ".".join(str(random.randint(1,254)) for _ in range(4))

def random_timestamp():
    start_date = datetime(2024,1,1)
    end_date = datetime(2024,12,31)
    delta = end_date - start_date
    rand_date = start_date + timedelta(seconds=random.randint(0, int(delta.total_seconds())))
    duration = timedelta(minutes=random.randint(1, 60))
    return rand_date.isoformat(), (rand_date + duration).isoformat() + "Z"

def generate_device_metrics():
    return {
        "sentSignalLevel": random.randint(0,100),
        "sentNoiseLevel": random.randint(0,50),
        "inputClippingEventRatio": round(random.uniform(0,0.2),2),
        "deviceClippingEventRatio": round(random.uniform(0,0.2),2),
        "glitchRate": round(random.uniform(0,5),2),
        "speakerGlitchRate": round(random.uniform(0,5),2)
    }

def create_stream():
    start, end = random_timestamp()
    return {
        "streamId": str(uuid.uuid4()),
        "streamType": random.choice(["callerToCallee", "calleeToCaller"]),
        "averageJitter": round(random.uniform(0.0, 0.15), 2),
        "packetLossRate": round(random.uniform(0.0, 0.12), 3),
        "averageRoundTripTime": round(random.uniform(100.0,200.0),1),
        "startDateTime": start,
        "endDateTime": end,
        "deviceMetrics": generate_device_metrics()
    }

def create_network_info():
    return {
        "networkType": random.choice(["wired", "wireless"]),
        "averageBandwidthEstimate": random.randint(25000,90000),
        "averageReorderRatio": round(random.uniform(0,0.2),2),
        "ipAddress": random_ip()
    }

def create_participant():
    return {
        "participantId": str(uuid.uuid4()),
        "role": random.choice(["organizer","presenter","attendee"]),
        "networkInfo": create_network_info(),
        "streams": [create_stream() for _ in range(random.randint(1,2))]
    }

def create_session():
    num_participants = random.randint(2,5)
    participants = [create_participant() for _ in range(num_participants)]
    start, end = random_timestamp()
    return {
        "sessionId": str(uuid.uuid4()),
        "startDateTime": start,
        "endDateTime": end,
        "participants": participants
    }

def create_cdr_record():
    record = {
        "conferenceId": str(uuid.uuid4()),
        "callType": random.choice(["groupCall", "peerToPeer"]),
        "organizerUPN": random.choice([
            "adele.vance@contoso.com", "alex.wilber@contoso.com", "megan.bowen@contoso.com",
            "lynne.robbins@contoso.com", "diego.siciliani@contoso.com", "patti.ferguson@contoso.com"
        ]),
        "sessions": [create_session() for _ in range(random.randint(1,2))]
    }
    return record

if __name__ == "__main__":
    NUM_RECORDS = 50
    for i in range(NUM_RECORDS):
        rec = create_cdr_record()
        with open(CREATED_JSON_DIR / f"cdr_{i+1:03d}.json", "w") as f:
            json.dump(rec, f, indent=2)
    print(f"Created {NUM_RECORDS} sample CDR files in {CREATED_JSON_DIR}")
# flatten_json.py
import json
from pathlib import Path

CREATED_JSON_DIR = Path("created_json")
OUTPUT_JSONL = "flattened_cdrs.jsonl"

def flatten_record(record):
    """
    Flattens a nested CDR JSON into a dict for embedding/search.
    Focuses on organizer, participants, overall metrics, and notable stats.
    """
    organizer = record.get("organizerUPN")
    call_type = record.get("callType")
    conference_id = record.get("conferenceId")
    sessions = record.get("sessions", [])

    flat = {
        "organizerUPN": organizer,
        "callType": call_type,
        "conferenceId": conference_id,
        "sessionCount": len(sessions),
        "participantRoles": [],
        "notableMetrics": [],
    }

    for session in sessions:
        for participant in session.get("participants", []):
            flat["participantRoles"].append(participant.get("role"))
            for stream in participant.get("streams", []):
                avg_jitter = stream.get("averageJitter")
                packet_loss = stream.get("packetLossRate")
                avg_rtt = stream.get("averageRoundTripTime")
                glitch_rate = stream["deviceMetrics"].get("glitchRate") if "deviceMetrics" in stream else None

                desc = (f"role:{participant.get('role')} avgJitter:{avg_jitter} packetLoss:{packet_loss} "
                        f"avgRTT:{avg_rtt} glitchRate:{glitch_rate}")
                flat["notableMetrics"].append(desc)
    
    # For RAG: Combine into a single string
    summary = (
        f"Call by {organizer}, type: {call_type}. "
        f"Participants: {', '.join(flat['participantRoles'])}. "
        f"Session count: {flat['sessionCount']}. "
        f"Metrics: {' | '.join(flat['notableMetrics'])}."
    )

    return {
        "conferenceId": conference_id,
        "organizerUPN": organizer,
        "summary": summary
    }

if __name__ == "__main__":
    files = list(CREATED_JSON_DIR.glob("*.json"))
    with open(OUTPUT_JSONL, "w") as fout:
        for fp in files:
            with open(fp) as fin:
                rec = json.load(fin)
                flat = flatten_record(rec)
                fout.write(json.dumps(flat) + "\n")
    print(f"Flattened {len(files)} records into {OUTPUT_JSONL}")

