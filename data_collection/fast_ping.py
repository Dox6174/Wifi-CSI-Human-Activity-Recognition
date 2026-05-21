import socket
import time

# --- CONFIGURATION ---
# IMPORTANT: Change this to the exact IP address of your ESP32
TARGET_IP = "192.168.0.101"  
TARGET_PORT = 80  
INTERVAL_MS = 100  # 100ms interval = 10 Hz

print(f"Starting 10 Hz traffic to {TARGET_IP}...")
print("Press Ctrl+C to stop.")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
message = b"CSI_WAKEUP_PACKET"

try:
    while True:
        # Fire a packet at the ESP32
        sock.sendto(message, (TARGET_IP, TARGET_PORT))
        
        # Pause for exactly 100ms
        time.sleep(INTERVAL_MS / 1000.0)
        
except KeyboardInterrupt:
    print("\nFast ping stopped.")
    sock.close()