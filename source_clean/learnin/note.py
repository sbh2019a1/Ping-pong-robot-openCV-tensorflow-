import socket
import random
from time import sleep


def socket_sender(IP, portnum, msg):
    sock=socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((IP,portnum))
    sock.send(msg.encode())
    print("Send:", msg)

while True:
    data = ""
    for j in range(3):
        data = data + str(random.random()) + "\t"

    socket_sender('127.0.0.1', 8888, data)
    sleep(1)