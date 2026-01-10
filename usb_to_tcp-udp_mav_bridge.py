#!/usr/bin/env python3
import socket
import serial
import threading

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/ttyACM0"  # change if needed
BAUD_RATE = 115200

TCP_IP = "0.0.0.0"
TCP_PORT = 5760          # TCP port for Mission Planner

UDP_IP = "<broadcast>"
UDP_PORT = 14550         # UDP port for Mission Planner
# ----------------------------------------

def serial_to_tcp(serial_port, tcp_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((TCP_IP, tcp_port))
    server_socket.listen(5)
    print(f"[TCP] Listening on {TCP_IP}:{tcp_port} ...")

    clients = []

    def handle_client(conn, addr):
        print(f"[TCP] Client connected: {addr}")
        try:
            while True:
                data = serial_port.read(serial_port.in_waiting or 1)
                if data:
                    conn.sendall(data)
        except Exception as e:
            print(f"[TCP] Client {addr} disconnected: {e}")
        finally:
            conn.close()
            if conn in clients:
                clients.remove(conn)

    def accept_clients():
        while True:
            conn, addr = server_socket.accept()
            clients.append(conn)
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

    threading.Thread(target=accept_clients, daemon=True).start()
    print("[TCP] Accepting clients in background thread.")

def serial_to_udp(serial_port, udp_port):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"[UDP] Broadcasting on {UDP_IP}:{udp_port} ...")
    while True:
        data = serial_port.read(serial_port.in_waiting or 1)
        if data:
            udp_socket.sendto(data, (UDP_IP, udp_port))

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
        print(f"[INFO] Connected to serial port {SERIAL_PORT} at {BAUD_RATE} baud")
    except Exception as e:
        print(f"[ERROR] Could not open serial port: {e}")
        return

    # Start TCP and UDP bridges
    threading.Thread(target=serial_to_tcp, args=(ser, TCP_PORT), daemon=True).start()
    threading.Thread(target=serial_to_udp, args=(ser, UDP_PORT), daemon=True).start()

    print("[INFO] MAVLink bridge running (TCP + UDP). Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n[INFO] Stopping MAVLink bridge.")
        ser.close()

if __name__ == "__main__":
    main()
