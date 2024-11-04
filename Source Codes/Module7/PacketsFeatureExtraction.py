import pyshark

# Path to your Wireshark capture file
capture_file = 'tcp-3-way-handshake.pcap'
capture_file = 'we.pcapng'

# Create a capture object
capture = pyshark.FileCapture(capture_file)

# Function to extract data from a packet
def extract_packet_info(packet):
    packet_info = {
        'packet_number': packet.number,
        'length': packet.length,
        'highest_layer': packet.highest_layer,
        'layers': [layer.layer_name for layer in packet.layers],
    }

    # Extract IP layer info if exists
    if 'IP' in packet:
        packet_info.update({
            'src_ip': packet.ip.src,
            'dst_ip': packet.ip.dst,
            'ip_version': packet.ip.version,
            'ip_ttl': packet.ip.ttl
        })

    # Extract TCP/UDP layer info if exists
    if 'TCP' in packet or 'UDP' in packet:
        packet_info.update({
            'src_port': packet[packet.transport_layer].srcport,
            'dst_port': packet[packet.transport_layer].dstport,
            'transport_layer': packet.transport_layer
        })

    # Extract DNS information if exists
    if 'DNS' in packet:
        dns_queries = getattr(packet.dns, 'qry_name', None)
        packet_info['dns_queries'] = dns_queries

    # Extract HTTP information if exists
    if 'HTTP' in packet:
        packet_info['http_request_method'] = getattr(packet.http, 'request_method', None)
        packet_info['http_request_uri'] = getattr(packet.http, 'request_full_uri', None)
        packet_info['http_user_agent'] = getattr(packet.http, 'user_agent', None)

    # Add more protocol-specific information as needed...

    return packet_info

# Process and print packet information
for pkt in capture:
    try:
        info = extract_packet_info(pkt)
        print(info)
    except AttributeError as e:
        # Handle packets that might not have the expected attributes
        print(f"Error in packet {pkt.number}: {e}")

# Close the capture file
capture.close()
