

import sys
import asyncio

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


import cv2
import mediapipe as mp
import numpy as np
import time
from bleak import BleakClient, BleakScanner



try:
    import anvil.server
    ANVIL_AVAILABLE = True
except ImportError:
    ANVIL_AVAILABLE = False
    print("⚠ Anvil not installed. Run: pip install anvil-uplink")

class EyeDetectorBLE:
    """
    Eye detector using Bluetooth Low Energy (BLE) for Arduino Nano 33 IoT.
    
    Status Codes:
    0 = Normal (eyes open)
    1 = Eyes closed >5s
    2 = Face lost
    3 = Camera error
    """
    
    STATUS_NORMAL = 0
    STATUS_EYES_CLOSED = 1
    STATUS_FACE_LOST = 2
    STATUS_CAMERA_ERROR = 3
    
    # BLE UUIDs (must match Arduino code)
    SERVICE_UUID = "19b10000-e8f2-537e-4f6c-d104768a1214"
    CHAR_UUID = "19b10001-e8f2-537e-4f6c-d104768a1214"
    
    def __init__(self, arduino_address=None, anvil_key=None):
        """
        Initialize detector with BLE.
        
        Args:
            arduino_address: BLE MAC address (e.g., "E4:65:B8:XX:XX:XX")
            anvil_key: Anvil uplink key (optional)
        """
        # MediaPipe setup
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        
        # Eye landmarks
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        
        # Detection parameters
        self.EYE_AR_THRESHOLD = 0.2
        self.CLOSED_DURATION_THRESHOLD = 5.0
        
        # State tracking
        self.eyes_closed_start_time = None
        self.is_detecting = False
        self.face_lost_count = 0
        self.FACE_LOST_THRESHOLD = 30
        self.last_status = self.STATUS_NORMAL
        
        # BLE connection
        self.arduino_address = arduino_address
        self.ble_client = None
        
        # Anvil connection
        self.anvil_connected = False
        if anvil_key and ANVIL_AVAILABLE:
            try:
                anvil.server.connect(anvil_key)
                self.anvil_connected = True
                print("✓ Connected to Anvil")
            except Exception as e:
                print(f"✗ Could not connect to Anvil: {e}")
    
    async def scan_for_arduino(self):
        """Scan for Arduino Nano 33 IoT devices."""
        print("\nScanning for Arduino devices...")
        devices = await BleakScanner.discover(timeout=10.0)
        
        arduino_devices = []
        for d in devices:
            if d.name and ("Arduino" in d.name or "Nano" in d.name):
                arduino_devices.append(d)
                print(f"  Found: {d.name} - {d.address}")
        
        return arduino_devices
    
    async def connect_ble(self):
        """Connect to Arduino via BLE."""
        if not self.arduino_address:
            return False
        
        try:
            print(f"\nConnecting to {self.arduino_address}...")
            self.ble_client = BleakClient(self.arduino_address)
            await self.ble_client.connect()
            print("✓ Connected to Arduino via BLE")
            return True
        except Exception as e:
            print(f"✗ BLE connection failed: {e}")
            return False
    
    async def send_to_arduino(self, status_code):
        """Send status code via BLE."""
        if self.ble_client and self.ble_client.is_connected:
            try:
                # Send as single byte
                await self.ble_client.write_gatt_char(
                    self.CHAR_UUID, 
                    bytes([status_code])
                )
            except Exception as e:
                print(f"Error sending BLE: {e}")
    
    def send_to_anvil(self, status_code, ear_value, closed_duration):
        """Send data to Anvil."""
        if self.anvil_connected:
            try:
                anvil.server.call('update_status',
                                  status=status_code,
                                  ear=round(ear_value, 3),
                                  closed_duration=round(closed_duration, 1),
                                  timestamp=time.time())
            except Exception as e:
                print(f"Error sending to Anvil: {e}")
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """Calculate EAR."""
        A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (A + B) / (2.0 * C)
    
    def get_eye_landmarks(self, face_landmarks, eye_indices, w, h):
        """Extract eye landmarks."""
        landmarks = []
        for idx in eye_indices:
            landmark = face_landmarks.landmark[idx]
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append([x, y])
        return np.array(landmarks)
    
    def get_status_info(self, status_code):
        """Get display info for status."""
        status_map = {
            self.STATUS_NORMAL: ("NORMAL - Eyes Open", (0, 255, 0)),
            self.STATUS_EYES_CLOSED: ("ALERT - Eyes Closed!", (0, 0, 255)),
            self.STATUS_FACE_LOST: ("WARNING - Face Lost", (0, 165, 255)),
            self.STATUS_CAMERA_ERROR: ("ERROR - Camera Lost", (0, 0, 255))
        }
        return status_map.get(status_code, ("UNKNOWN", (128, 128, 128)))
    
    async def detection_loop(self):
        """Main detection loop with BLE support."""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("✗ Camera error")
            if self.is_detecting and self.ble_client:
                await self.send_to_arduino(self.STATUS_CAMERA_ERROR)
            return
        
        print("\n" + "="*60)
        print("EYE DETECTION - BLE MODE")
        print("="*60)
        print("Press 'Q' to quit | 'SPACE' to start/pause")
        print("="*60 + "\n")
        
        consecutive_failures = 0
        
        while cap.isOpened():
            success, frame = cap.read()
            
            if not success:
                consecutive_failures += 1
                if consecutive_failures > 30:
                    if self.is_detecting and self.ble_client:
                        await self.send_to_arduino(self.STATUS_CAMERA_ERROR)
                time.sleep(0.1)
                continue
            
            consecutive_failures = 0
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            
            results = self.face_mesh.process(rgb_frame)
            
            current_status = self.STATUS_NORMAL
            ear_text = "N/A"
            avg_ear = 0
            closed_duration = 0
            
            if results.multi_face_landmarks:
                self.face_lost_count = 0
                face_landmarks = results.multi_face_landmarks[0]
                
                left_eye = self.get_eye_landmarks(face_landmarks, self.LEFT_EYE, w, h)
                right_eye = self.get_eye_landmarks(face_landmarks, self.RIGHT_EYE, w, h)
                
                left_ear = self.calculate_eye_aspect_ratio(left_eye)
                right_ear = self.calculate_eye_aspect_ratio(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0
                ear_text = f"{avg_ear:.3f}"
                
                # Draw landmarks
                for landmark in left_eye:
                    cv2.circle(frame, tuple(landmark), 2, (0, 255, 0), -1)
                for landmark in right_eye:
                    cv2.circle(frame, tuple(landmark), 2, (0, 255, 0), -1)
                
                if self.is_detecting:
                    if avg_ear < self.EYE_AR_THRESHOLD:
                        if self.eyes_closed_start_time is None:
                            self.eyes_closed_start_time = time.time()
                        else:
                            closed_duration = time.time() - self.eyes_closed_start_time
                            if closed_duration >= self.CLOSED_DURATION_THRESHOLD:
                                current_status = self.STATUS_EYES_CLOSED
                    else:
                        self.eyes_closed_start_time = None
            else:
                self.face_lost_count += 1
                if self.face_lost_count > self.FACE_LOST_THRESHOLD:
                    if self.is_detecting:
                        current_status = self.STATUS_FACE_LOST
                    self.eyes_closed_start_time = None
            
            # Send updates
            if self.is_detecting:
                if current_status != self.last_status:
                    print(f"Status: {current_status}")
                    self.last_status = current_status
                
                if self.ble_client:
                    await self.send_to_arduino(current_status)
                self.send_to_anvil(current_status, avg_ear, closed_duration)
            
            # Display
            status_text, status_color = self.get_status_info(current_status)
            detection_color = (0, 255, 0) if self.is_detecting else (0, 165, 255)
            detection_text = "DETECTING" if self.is_detecting else "PAUSED"
            
            cv2.putText(frame, f"Mode: {detection_text}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, detection_color, 2)
            cv2.putText(frame, f"Status: {status_text}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(frame, f"EAR: {ear_text}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Code: {current_status}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if self.ble_client and self.ble_client.is_connected:
                cv2.putText(frame, "BLE: Connected", (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            if self.is_detecting and self.eyes_closed_start_time:
                closed_time = time.time() - self.eyes_closed_start_time
                cv2.putText(frame, f"Closed: {closed_time:.1f}s", (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow('Eye Detection (BLE)', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord(' '):
                self.is_detecting = not self.is_detecting
                print(f"Detection {'STARTED' if self.is_detecting else 'PAUSED'}")
            
            # Small delay for async operations
            await asyncio.sleep(0.01)
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        if self.ble_client and self.ble_client.is_connected:
            await self.ble_client.disconnect()
        print("\nDetection stopped")


async def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("EYE DETECTION - ARDUINO NANO 33 IoT (BLE)")
    print("="*60)
    
    detector = EyeDetectorBLE()
    
    # Scan for Arduino
    use_scan = input("\nScan for Arduino devices? (y/n): ").strip().lower()
    arduino_address = None
    
    if use_scan == 'y':
        devices = await detector.scan_for_arduino()
        if devices:
            print("\nSelect device:")
            for i, d in enumerate(devices):
                print(f"  {i+1}. {d.name} - {d.address}")
            
            try:
                choice = int(input("Enter number: ")) - 1
                arduino_address = devices[choice].address
            except:
                print("Invalid choice")
    else:
        arduino_address = input("Enter Arduino BLE address (or skip): ").strip()
    
    if arduino_address:
        detector.arduino_address = arduino_address
        connected = await detector.connect_ble()
        if not connected:
            print("Running without Arduino...")
    
    # Anvil setup
    anvil_key = None
    if ANVIL_AVAILABLE:
        use_anvil = input("\nConnect to Anvil? (y/n): ").strip().lower()
        if use_anvil == 'y':
            anvil_key = input("Enter Anvil uplink key: ").strip()
            detector.anvil_connected = True
            try:
                anvil.server.connect(anvil_key)
            except:
                print("Anvil connection failed")
    
    # Start detection
    await detector.detection_loop()


import threading
import asyncio

def run_ble():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())

if __name__ == "__main__":
    t = threading.Thread(target=run_ble)
    t.start()
    t.join()
