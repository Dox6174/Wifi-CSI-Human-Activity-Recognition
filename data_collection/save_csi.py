import serial
import csv
from datetime import datetime

# --- CONFIGURATION ---
COM_PORT = 'COM4'  
BAUD_RATE = 115200 # Default baud rate. 

print(f"Connecting to {COM_PORT}...")

try:
    # Open the serial port
    ser = serial.Serial(COM_PORT, BAUD_RATE)
    print("Connected! Listening for CSI data... (Press Ctrl+C to stop)")

    # Open the CSV file to write
    with open('csi_data_log.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header row
        writer.writerow(['PC_Timestamp', 'Raw_CSI_String'])

        while True:
            if ser.in_waiting > 0:
                # Read the line from the ESP32
                line = ser.readline().decode('utf-8', errors='ignore').strip()

                # Filter out the "ba-add" junk and only save CSI data
                if line.startswith('CSI_DATA'):
                    # Get the exact time down to the millisecond
                    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    
                    # Write to CSV

                    
                    writer.writerow([current_time, line])
                    
                    # Print to terminal so you know it's working
                    print(f"Captured packet at {current_time}")

except KeyboardInterrupt:
    print("\nData collection stopped. File saved as 'csi_data_log.csv'.")
except serial.SerialException as e:
    print(f"\nError: Could not open {COM_PORT}. Is the VS Code monitor still open?")