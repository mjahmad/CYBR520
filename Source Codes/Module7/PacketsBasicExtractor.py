import pyshark
import pandas as pd
from datetime import datetime

# Path to your Wireshark capture file
#capture_file = 'tcp-3-way-handshake.pcap'
capture_file = 'we.pcapng'

# Create a capture object
capture = pyshark.FileCapture(capture_file, keep_packets=False)

# Variables to store session data
session_packets = 0
total_bytes_src_to_dst = 0
total_bytes_dst_to_src = 0
ack_count = 0
tcp_handshake = False
syn_count = 0
syn_ack_count = 0
protocols_used = set()
start_time = None
end_time = None

# Analyzing packets
for packet in capture:
    session_packets += 1
    protocols_used.add(packet.highest_layer)

    # Calculate session duration
    timestamp = datetime.fromtimestamp(float(packet.sniff_timestamp))
    if start_time is None:
        start_time = timestamp
    end_time = timestamp

    # Check for TCP packets and calculate bytes transferred
    if 'TCP' in packet:
        if packet.ip.src == packet.tcp.srcport:
            total_bytes_src_to_dst += int(packet.tcp.len)
        else:
            total_bytes_dst_to_src += int(packet.tcp.len)

        # Count ACK packets
        if 'ack' in packet.tcp.field_names:
            ack_count += 1

        # Check for SYN and SYN-ACK packets for TCP handshake
        if 'syn' in packet.tcp.field_names and 'ack' not in packet.tcp.field_names:
            syn_count += 1
        if 'syn' in packet.tcp.field_names and 'ack' in packet.tcp.field_names:
            syn_ack_count += 1

# Determine if a successful TCP handshake occurred
tcp_handshake = syn_count > 0 and syn_ack_count > 0

# Calculate session duration
session_duration = (end_time - start_time).total_seconds() if start_time and end_time else 0

# Determine network service (based on destination port)
# This is a simplified approach; for more accuracy, you might need a port-service mapping
network_service = 'Unknown'
if 'TCP' in packet and hasattr(packet.tcp, 'dstport'):
    network_service = packet.tcp.dstport

# Creating a DataFrame and writing to a CSV file
df = pd.DataFrame([{
    'Session Duration (seconds)': session_duration,
    'Total Session Packets': session_packets,
    'Protocols Used': ', '.join(protocols_used),
    'Total Bytes Src to Dst': total_bytes_src_to_dst,
    'Total Bytes Dst to Src': total_bytes_dst_to_src,
    'Successful TCP Handshake': tcp_handshake,
    'Network Service on Destination': network_service,
    'Number of ACK Packets': ack_count
}])

csv_file = 'session_analysis.csv'
df.to_csv(csv_file, index=False)

print(f"Session analysis saved to {csv_file}")
