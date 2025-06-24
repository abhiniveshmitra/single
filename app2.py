import json
from pathlib import Path

CREATED_JSON_DIR = Path("created_json")
OUTPUT_JSONL = "flattened_cdrs.jsonl"

def flatten_record(record):
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
            role = participant.get("role", "")
            platform = participant.get("platform", "")
            flat["participantRoles"].append(f"{role}({platform})")
            for stream in participant.get("streams", []):
                avg_jitter = stream.get("averageJitter")
                packet_loss = stream.get("packetLossRate")
                avg_rtt = stream.get("averageRoundTripTime")
                device_metrics = stream.get("deviceMetrics", {})
                glitch_rate = device_metrics.get("glitchRate")
                sent_signal = device_metrics.get("sentSignalLevel")

                desc = (
                    f"role:{role} platform:{platform} "
                    f"avgJitter:{avg_jitter} packetLoss:{packet_loss} "
                    f"avgRTT:{avg_rtt} glitchRate:{glitch_rate} sentSignalLevel:{sent_signal}"
                )
                flat["notableMetrics"].append(desc)

    # Build summary string that includes platform and all key metrics for agent routing
    summary = (
        f"Call by {organizer}, type: {call_type}. "
        f"Participants: {', '.join(flat['participantRoles'])}. "
        f"Session count: {flat['sessionCount']}. "
        f"Metrics: {' | '.join(flat['notableMetrics'])}."
    )

    return {
        "conferenceId": conference_id,
        "organizerUPN": organizer,
        "summary": summary,
        "notableMetrics": flat["notableMetrics"]
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
