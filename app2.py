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
    # About 15% have bad log metrics: glitchRate > 2.5 or sentSignalLevel < 10
    is_log_issue = random.random() < 0.15
    if is_log_issue:
        glitch_rate = round(random.uniform(2.6, 5), 2)
        sent_signal = random.randint(0, 9)
    else:
        glitch_rate = round(random.uniform(0, 2.5), 2)
        sent_signal = random.randint(10, 100)
    return {
        "sentSignalLevel": sent_signal,
        "sentNoiseLevel": random.randint(0,50),
        "inputClippingEventRatio": round(random.uniform(0,0.2),2),
        "deviceClippingEventRatio": round(random.uniform(0,0.2),2),
        "glitchRate": glitch_rate,
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

VDI_PLATFORMS = ["vdi", "citrix", "vmware"]

def create_participant():
    # About 20% VDI, rest normal
    is_vdi = random.random() < 0.20
    if is_vdi:
        platform = random.choice(VDI_PLATFORMS)
    else:
        platform = random.choice(["windows", "macOS", "android", "iOS", "web"])
    return {
        "participantId": str(uuid.uuid4()),
        "role": random.choice(["organizer","presenter","attendee"]),
        "platform": platform,
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
    print(f"Created {NUM_RECORDS} sample CDR files in {CREATED_JSON_DIR} (with ~20% VDI, ~15% log issue)")
