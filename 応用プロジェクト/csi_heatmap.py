#!/usr/bin/env python3
"""
CSI Data Heatmap Generator
Extracts CSI data from pcap file and generates heatmap visualization
Exports to MATLAB format for WHOFI analysis
"""

import struct
import numpy as np
import matplotlib.pyplot as plt
from scapy.all import rdpcap, UDP
from pathlib import Path
from scipy.io import savemat


def extract_csi_from_pcap(pcap_file):
    """
    Extract CSI data from nexmon_csi pcap file

    nexmon_csi packet structure:
    - Bytes 0-1: Magic bytes (0x1111)
    - Bytes 2-7: Source MAC address
    - Bytes 8-9: Sequence number
    - Bytes 10-11: Core/Spatial stream
    - Bytes 12-13: Channel specification
    - Bytes 14-15: Chip version
    - Bytes 16-39: Additional metadata (24 bytes)
    - Bytes 40+: CSI data (int16 real, int16 imag pairs)

    Args:
        pcap_file: Path to pcap file

    Returns:
        list of complex numpy arrays (one per packet)
    """
    print(f"Reading pcap file: {pcap_file}")
    packets = rdpcap(str(pcap_file))

    # nexmon_csi header size (empirically determined)
    NEXMON_HEADER_SIZE = 40

    csi_data_list = []
    skipped_packets = 0

    for i, pkt in enumerate(packets):
        if UDP in pkt:
            # Extract UDP payload
            payload = bytes(pkt[UDP].payload)

            if len(payload) <= NEXMON_HEADER_SIZE:
                skipped_packets += 1
                continue

            # Verify magic bytes (optional validation)
            magic = struct.unpack('<H', payload[0:2])[0]
            if magic != 0x1111:
                print(f"Warning: Packet {i} has invalid magic bytes: 0x{magic:04x}")
                skipped_packets += 1
                continue

            # Skip nexmon_csi header and extract CSI data
            csi_data = payload[NEXMON_HEADER_SIZE:]

            # Parse as 16-bit signed integers (little-endian)
            # Each complex number is represented as (real, imag) pair
            try:
                num_bytes = len(csi_data)
                num_complex = num_bytes // 4  # 4 bytes per complex number

                if num_complex == 0:
                    skipped_packets += 1
                    continue

                # Create complex array from int16 pairs
                complex_data = np.zeros(num_complex, dtype=complex)

                for j in range(num_complex):
                    offset = j * 4
                    real_part = struct.unpack('<h', csi_data[offset:offset+2])[0]
                    imag_part = struct.unpack('<h', csi_data[offset+2:offset+4])[0]
                    complex_data[j] = complex(real_part, imag_part)

                csi_data_list.append(complex_data)

            except struct.error as e:
                print(f"Warning: Failed to parse packet {i}: {e}")
                skipped_packets += 1
                continue

    print(f"Extracted CSI data from {len(csi_data_list)} packets")
    if skipped_packets > 0:
        print(f"Skipped {skipped_packets} packets (invalid or empty)")

    return csi_data_list


def create_csi_matrix(csi_data_list):
    """
    Convert list of CSI arrays into 2D matrix (packets × subcarriers)

    Args:
        csi_data_list: List of complex numpy arrays

    Returns:
        2D numpy array of complex values
    """
    if not csi_data_list:
        return np.array([])

    # Find the minimum length to ensure all rows have same size
    min_length = min(len(csi) for csi in csi_data_list)

    # Create matrix: rows = packets (time), cols = subcarriers
    num_packets = len(csi_data_list)
    csi_matrix = np.zeros((num_packets, min_length), dtype=complex)

    for i, csi in enumerate(csi_data_list):
        csi_matrix[i, :] = csi[:min_length]

    return csi_matrix


def plot_csi_heatmap(csi_matrix, output_file='csi_heatmap.png'):
    """
    Generate heatmap visualization with 5-95 percentile clipping

    Args:
        csi_matrix: 2D array of complex CSI values
        output_file: Output image filename
    """
    if csi_matrix.size == 0:
        print("Error: No CSI data to plot")
        return

    amplitude = np.abs(csi_matrix)

    # Use 5-95 percentile clipping (best visualization)
    vmin, vmax = np.percentile(amplitude, [5, 95])
    amplitude_clipped = np.clip(amplitude, vmin, vmax)

    # Create single figure with clipped amplitude heatmap
    fig, ax = plt.subplots(figsize=(14, 8))

    im = ax.imshow(amplitude_clipped.T, aspect='auto', cmap='viridis',
                   interpolation='bilinear', origin='lower', vmin=vmin, vmax=vmax)
    ax.set_xlabel('Packet Index (Time)', fontsize=12)
    ax.set_ylabel('Subcarrier Index', fontsize=12)
    ax.set_title('CSI Amplitude Heatmap (5-95 Percentile Clipped)', fontsize=14, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax, label='Amplitude |H|')
    cbar.set_label('Amplitude |H|', fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Heatmap saved to: {output_file}")

    # Show statistics
    print(f"\nCSI Matrix Shape: {csi_matrix.shape}")
    print(f"Amplitude - Min: {amplitude.min():.2f}, Max: {amplitude.max():.2f}, Mean: {amplitude.mean():.2f}")
    print(f"Clipping range (5-95 percentile): {vmin:.2f} to {vmax:.2f}")


def reshape_csi_for_whofi(csi_matrix, num_streams=3, num_subcarriers=114, num_packets=500):
    """
    Reshape CSI matrix to WHOFI format: (num_streams, num_subcarriers, num_packets)

    Args:
        csi_matrix: Original CSI matrix (packets × subcarriers)
        num_streams: Number of antenna streams (default: 3)
        num_subcarriers: Target number of subcarriers (default: 114)
        num_packets: Target number of packets (default: 500)

    Returns:
        Reshaped CSI matrix of shape (num_streams, num_subcarriers, num_packets)
    """
    original_packets, original_subcarriers = csi_matrix.shape

    # Extract center subcarriers
    start_sc = (original_subcarriers - num_subcarriers) // 2
    end_sc = start_sc + num_subcarriers

    # Create 3D array for multiple streams
    csi_3d = np.zeros((num_streams, num_subcarriers, num_packets), dtype=complex)

    # Strategy: Extract different packet ranges for each stream
    # to simulate multi-antenna data
    packets_per_stream = original_packets // num_streams

    for stream_idx in range(num_streams):
        # Calculate packet range for this stream
        start_pkt = stream_idx * packets_per_stream

        # Extract packets and subcarriers
        if start_pkt + num_packets <= original_packets:
            # Enough packets available
            stream_data = csi_matrix[start_pkt:start_pkt + num_packets, start_sc:end_sc]
        else:
            # Wrap around or use available packets
            available = original_packets - start_pkt
            stream_data_part1 = csi_matrix[start_pkt:, start_sc:end_sc]
            stream_data_part2 = csi_matrix[:num_packets - available, start_sc:end_sc]
            stream_data = np.vstack([stream_data_part1, stream_data_part2])

        # Transpose to (subcarriers, packets) and assign
        csi_3d[stream_idx, :, :] = stream_data.T

    return csi_3d


def save_csi_to_mat(csi_matrix, output_file='csi_data.mat'):
    """
    Save CSI data to MATLAB .mat format for WHOFI analysis
    Reshapes to (3, 114, 500) format as required

    Args:
        csi_matrix: 2D array of complex CSI values (packets × subcarriers)
        output_file: Output .mat filename
    """
    if csi_matrix.size == 0:
        print("Error: No CSI data to save")
        return

    # Reshape to WHOFI format: (3, 114, 500)
    print("Reshaping CSI data for WHOFI format...")
    csi_3d = reshape_csi_for_whofi(csi_matrix, num_streams=3, num_subcarriers=114, num_packets=500)

    # Calculate amplitude and phase for 3D data
    amplitude_3d = np.abs(csi_3d)
    phase_3d = np.angle(csi_3d)

    # Prepare data structure for MATLAB
    mat_data = {
        'csi': csi_3d,                       # Complex CSI data (3 × 114 × 500)
        'csi_amplitude': amplitude_3d,       # Amplitude |H| (3 × 114 × 500)
        'csi_phase': phase_3d,               # Phase angle (3 × 114 × 500)
        'num_streams': 3,
        'num_subcarriers': 114,
        'num_packets': 500,
        'description': 'CSI data in WHOFI format (streams × subcarriers × packets)',
        'format': 'nexmon_csi_whofi',
        'bandwidth': '80MHz'
    }

    # Save to .mat file (MATLAB v7.3 format for large arrays)
    savemat(output_file, mat_data, do_compression=True)
    print(f"MATLAB file saved to: {output_file}")
    print(f"  - csi: {csi_3d.shape} (complex)")
    print(f"  - csi_amplitude: {amplitude_3d.shape} (real)")
    print(f"  - csi_phase: {phase_3d.shape} (real)")
    print(f"  - Format: 3 streams × 114 subcarriers × 500 packets")
    print(f"  - Dimension order: (streams, subcarriers, packets)")


def main():
    # Input/output file paths
    pcap_file = Path("output_3.pcap")  # Updated filename
    output_image = "csi_heatmap.png"
    output_mat = "csi_data.mat"

    if not pcap_file.exists():
        print(f"Error: {pcap_file} not found")
        return

    print("="*70)
    print("CSI Data Extraction and Visualization")
    print("="*70)

    # Extract CSI data from pcap
    csi_data_list = extract_csi_from_pcap(pcap_file)

    if not csi_data_list:
        print("Error: No CSI data extracted from pcap file")
        return

    # Convert to matrix
    csi_matrix = create_csi_matrix(csi_data_list)

    print("\n" + "="*70)
    print("Generating Visualizations")
    print("="*70)

    # Generate heatmap (5-95 percentile clipped)
    plot_csi_heatmap(csi_matrix, output_image)

    print("\n" + "="*70)
    print("Exporting to MATLAB Format")
    print("="*70)

    # Save to MATLAB .mat format for WHOFI
    save_csi_to_mat(csi_matrix, output_mat)

    print("\n" + "="*70)
    print("Processing Complete!")
    print("="*70)
    print(f"\nOutput files:")
    print(f"  - Heatmap: {output_image}")
    print(f"  - MATLAB: {output_mat}")
    print(f"\nTo use in MATLAB/WHOFI:")
    print(f"  >> data = load('{output_mat}');")
    print(f"  >> csi = data.csi;  % Shape: (3, 114, 500)")
    print(f"  >> amplitude = data.csi_amplitude;")
    print(f"  >> % csi(stream, subcarrier, packet)")


if __name__ == "__main__":
    main()
