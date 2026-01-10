#!/usr/bin/env python3
import socket
import serial
import threading

# ---------------- CONFIG ----------------
SERIAL_PORT = "/dev/ttyACM0"
BAUD_RATE = 115200

TCP_IP = "0.0.0.0"
TCP_PORT = 5760

UDP_IP = "<broadcast>"
UDP_PORT = 14550
# ----------------------------------------

clients = []  # list of TCP clients

def accept_tcp_clients(tcp_port):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((TCP_IP, tcp_port))
    server_socket.listen(5)
    print(f"[TCP] Listening on {TCP_IP}:{tcp_port} ...")

    def handle_client(conn, addr):
        print(f"[TCP] Client connected: {addr}")
        clients.append(conn)
        try:
            while True:
                # TCP is only sending data from serial, no receive needed
                pass
        except:
            pass
        finally:
            conn.close()
            if conn in clients:
                clients.remove(conn)
            print(f"[TCP] Client disconnected: {addr}")

    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

def serial_broadcast(serial_port):
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    print(f"[UDP] Broadcasting on {UDP_IP}:{UDP_PORT} ...")

    while True:
        data = serial_port.read(serial_port.in_waiting or 1)
        if data:
            # send to all TCP clients
            for c in clients:
                try:
                    c.sendall(data)
                except:
                    pass
            # send via UDP
            udp_socket.sendto(data, (UDP_IP, UDP_PORT))

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0)
        print(f"[INFO] Connected to serial port {SERIAL_PORT} at {BAUD_RATE} baud")
    except Exception as e:
        print(f"[ERROR] Could not open serial port: {e}")
        return

    # Start TCP client acceptor
    threading.Thread(target=accept_tcp_clients, args=(TCP_PORT,), daemon=True).start()
    # Start serial broadcast (TCP + UDP)
    threading.Thread(target=serial_broadcast, args=(ser,), daemon=True).start()

    print("[INFO] MAVLink bridge running (TCP + UDP). Press Ctrl+C to stop.")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\n[INFO] Stopping MAVLink bridge.")
        ser.close()

if __name__ == "__main__":
    main()
