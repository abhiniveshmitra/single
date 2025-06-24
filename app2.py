import json
from agents.network_agent import analyze_network
from agents.vdi_agent import analyze_vdi
from agents.log_agent import analyze_logs

def route_record(record):
    fired = None
    reasons = []

    # 1. Network conditions (packet loss, jitter, RTT)
    for metric in record.get("notableMetrics", []):
        # Packet loss
        if "packetLoss:" in metric:
            try:
                packet_loss = float(metric.split("packetLoss:")[1].split()[0])
                if packet_loss > 0.05:
                    fired = "Network"
                    reasons.append(f"packetLoss={packet_loss}")
                    break
            except Exception as e:
                reasons.append(f"packet_loss parse fail: {e}")
        # Jitter
        if "avgJitter:" in metric:
            try:
                avg_jitter = float(metric.split("avgJitter:")[1].split()[0])
                if avg_jitter > 0.08:
                    fired = "Network"
                    reasons.append(f"avgJitter={avg_jitter}")
                    break
            except Exception as e:
                reasons.append(f"avg_jitter parse fail: {e}")
        # RTT
        if "avgRTT:" in metric:
            try:
                avg_rtt = float(metric.split("avgRTT:")[1].split()[0])
                if avg_rtt > 200:
                    fired = "Network"
                    reasons.append(f"avgRTT={avg_rtt}")
                    break
            except Exception as e:
                reasons.append(f"avg_rtt parse fail: {e}")
    if fired:
        return fired, reasons

    # 2. VDI (platform keywords in metrics or participant roles)
    summary = record.get("summary", "").lower()
    vdi_keywords = ["vdi", "virtual desktop", "citrix", "vmware", "remote"]
    if any(kw in summary for kw in vdi_keywords):
        fired = "VDI"
        reasons.append("VDI keyword found in summary")
        return fired, reasons

    # 3. Device/log issues (glitchRate, sentSignalLevel)
    for metric in record.get("notableMetrics", []):
        # GlitchRate
        if "glitchRate:" in metric:
            try:
                glitch_rate = float(metric.split("glitchRate:")[1].split()[0])
                if glitch_rate is not None and glitch_rate > 2.5:
                    fired = "Log"
                    reasons.append(f"glitchRate={glitch_rate}")
                    break
            except Exception as e:
                reasons.append(f"glitch_rate parse fail: {e}")
        # sentSignalLevel
        if "sentSignalLevel:" in metric:
            try:
                signal_level = float(metric.split("sentSignalLevel:")[1].split()[0])
                if signal_level is not None and signal_level < 10:
                    fired = "Log"
                    reasons.append(f"sentSignalLevel={signal_level}")
                    break
            except Exception as e:
                reasons.append(f"signal_level parse fail: {e}")
    if fired:
        return fired, reasons

    return None, reasons

if __name__ == "__main__":
    network_count, vdi_count, log_count, none_count = 0, 0, 0, 0
    with open("flattened_cdrs.jsonl") as fin:
        for line in fin:
            record = json.loads(line)
            agent, reasons = route_record(record)
            cid = record.get('conferenceId')
            if agent == "Network":
                print(f"Call {cid}: Network agent fired. Reason(s): {', '.join(reasons)}")
                analyze_network(record)
                network_count += 1
            elif agent == "VDI":
                print(f"Call {cid}: VDI agent fired. Reason(s): {', '.join(reasons)}")
                analyze_vdi(record)
                vdi_count += 1
            elif agent == "Log":
                print(f"Call {cid}: Log agent fired. Reason(s): {', '.join(reasons)}")
                analyze_logs(record)
                log_count += 1
            else:
                print(f"Call {cid}: No issue detected")
                none_count += 1
    print("\nSummary:")
    print(f"  Network agent fired: {network_count} times")
    print(f"  VDI agent fired:     {vdi_count} times")
    print(f"  Log agent fired:     {log_count} times")
    print(f"  No issue:            {none_count} times")
