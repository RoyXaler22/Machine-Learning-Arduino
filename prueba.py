import serial
import time

arduino = serial.Serial("COM5", 9600)
time.sleep(2)

while True:

    n = input("Número: ")

    arduino.write(f"{n}\n".encode())