import pyshark
import pandas as pd

# Path to your Wireshark capture file
#capture_file = 'tcp-3-way-handshake.pcap'
capture_file = 'we.pcapng'


# Create a capture object
capture = pyshark.FileCapture(capture_file)

# Function to extract data from a packet
def extract_packet_info(packet):
    packet_info = {
        'packet_number': packet.number,
        'timestamp': packet.sniff_time,
        'length': packet.length,
        'highest_layer': packet.highest_layer,
        'protocol': packet.transport_layer
    }

    # IP Layer
    if 'IP' in packet:
        packet_info['src_ip'] = packet.ip.src
        packet_info['dst_ip'] = packet.ip.dst
        packet_info['ip_version'] = packet.ip.version

    # TCP/UDP Layer
    if 'TCP' in packet or 'UDP' in packet:
        packet_info['src_port'] = packet[packet.transport_layer].srcport
        packet_info['dst_port'] = packet[packet.transport_layer].dstport

    # HTTP Layer
    if 'HTTP' in packet:
        packet_info['http_host'] = packet.http.get('host', '')
        packet_info['http_method'] = packet.http.get('request_method', '')
        packet_info['http_uri'] = packet.http.get('request_uri', '')

    # DNS Layer
    if 'DNS' in packet:
        packet_info['dns_query'] = packet.dns.get('qry_name', '')
        packet_info['dns_response'] = packet.dns.get('a', '')

    return packet_info

# Extracting packet data
packets_data = []
for pkt in capture:
    packets_data.append(extract_packet_info(pkt))

# Creating a DataFrame and writing to a CSV file
df = pd.DataFrame(packets_data)
csv_file = 'extracted_packet_data.csv'
df.to_csv(csv_file, index=False)

# Close the capture file
capture.close()

print(f"Data extracted and saved to {csv_file}")
