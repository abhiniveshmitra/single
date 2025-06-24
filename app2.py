import json
from agents.network_agent import analyze_network
from agents.vdi_agent import analyze_vdi
from agents.log_agent import analyze_logs

def route_record(record):
    # 1. Network conditions (packet loss, jitter, RTT)
    for metric in record.get("notableMetrics", []):
        # Packet loss
        if "packetLoss:" in metric:
            try:
                packet_loss = float(metric.split("packetLoss:")[1].split()[0])
                if packet_loss > 0.05:
                    return "Network"
            except Exception:
                pass
        # Jitter
        if "avgJitter:" in metric:
            try:
                avg_jitter = float(metric.split("avgJitter:")[1].split()[0])
                if avg_jitter > 0.08:
                    return "Network"
            except Exception:
                pass
        # RTT
        if "avgRTT:" in metric:
            try:
                avg_rtt = float(metric.split("avgRTT:")[1].split()[0])
                if avg_rtt > 200:
                    return "Network"
            except Exception:
                pass

    # 2. VDI (platform keywords in metrics or participant roles)
    summary = record.get("summary", "").lower()
    vdi_keywords = ["vdi", "virtual desktop", "citrix", "vmware", "remote"]
    if any(kw in summary for kw in vdi_keywords):
        return "VDI"

    # 3. Device/log issues (glitchRate, sentSignalLevel)
    for metric in record.get("notableMetrics", []):
        # GlitchRate
        if "glitchRate:" in metric:
            try:
                glitch_rate = float(metric.split("glitchRate:")[1].split()[0])
                if glitch_rate is not None and glitch_rate > 2.5:
                    return "Log"
            except Exception:
                pass
        # sentSignalLevel
        if "sentSignalLevel:" in metric:
            try:
                signal_level = float(metric.split("sentSignalLevel:")[1].split()[0])
                if signal_level is not None and signal_level < 10:
                    return "Log"
            except Exception:
                pass

    return None

if __name__ == "__main__":
    with open("flattened_cdrs.jsonl") as fin:
        for line in fin:
            record = json.loads(line)
            route = route_record(record)
            if route == "Network":
                analyze_network(record)
            elif route == "VDI":
                analyze_vdi(record)
            elif route == "Log":
                analyze_logs(record)
            else:
                print(f"Call {record.get('conferenceId')}: No issue detected")
