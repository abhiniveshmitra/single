import json
import uuid
import random
from datetime import datetime, timedelta

def generate_call_records_to_file(num_records=50, output_filename="call_records.jsonl"):
    """
    Generates realistic, flattened Microsoft Teams call records and saves them
    to a specified output file in JSONL format.
    """
    
    sample_upns = [
        "adele.vance@contoso.com",
        "alex.wilber@contoso.com",
        "megan.bowen@contoso.com",
        "lynne.robbins@contoso.com",
        "diego.siciliani@contoso.com",
        "patti.ferguson@contoso.com"
    ]

    print(f"--- Generating {num_records} sample call records and saving to '{output_filename}' ---")

    # Use 'with open' to handle the file safely. It will automatically close the file.
    # We open the file in 'w' (write) mode.
    with open(output_filename, 'w') as f:
        for _ in range(num_records):
            # 1. GENERATE A REALISTIC, NESTED CALL RECORD OBJECT
            call_id = str(uuid.uuid4())
            call_type = random.choice(["groupCall", "peerToPeer"])
            start_time = datetime.utcnow() - timedelta(minutes=random.randint(5, 120))
            end_time = start_time + timedelta(minutes=random.randint(1, 45))
            
            organizer_upn = random.choice(sample_upns)
            participant_upn = random.choice([u for u in sample_upns if u != organizer_upn])
            
            possible_modalities = ["audio", "video", "videoBasedScreenSharing"]
            call_modalities = random.sample(possible_modalities, k=random.randint(1, len(possible_modalities)))

            jitter_value = random.uniform(0.005, 0.080) if random.random() > 0.2 else None
            average_jitter = f"PT{jitter_value:.3f}S" if jitter_value else None # ISO 8601 duration format

            avg_audio_degradation = round(random.uniform(0.1, 1.0), 2) if random.random() > 0.6 else None

            # 2. FLATTEN THE RECORD FOR ANALYSIS
            flattened_data = {
                "conferenceId": call_id,
                "callType": call_type,
                "startDateTime": start_time.isoformat() + "Z",
                "endDateTime": end_time.isoformat() + "Z",
                "modalities": call_modalities,
                "organizerUPN": organizer_upn,
                "participantUPN": participant_upn,
                "clientPlatform": random.choice(["windows", "macOS", "android"]),
                "averageJitter": average_jitter,
                "averageAudioDegradation": avg_audio_degradation,
                "joinWebUrl": f"https://teams.microsoft.com/l/meetup-join/19%3ameeting_{uuid.uuid4().hex}%40thread.v2/0" if call_type == "groupCall" else None,
            }

            # 3. WRITE THE FLATTENED JSON OBJECT TO THE FILE
            # json.dumps() converts the Python dictionary to a JSON string.
            # We add a newline character '\n' to ensure each JSON object is on its own line.
            f.write(json.dumps(flattened_data) + '\n')

    print(f"--- Successfully saved {num_records} records to '{output_filename}' ---")


# This block runs when you execute the script directly.
if __name__ == "__main__":
    # You can easily change the number of records or the filename here.
    generate_call_records_to_file(num_records=100, output_filename="sample_cdrs_for_analysis.jsonl")
