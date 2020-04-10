import numpy as np
import datetime
import socket
import threading
import time
from time import sleep

def socket_listener(IP, portnum):
    sever_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sever_socket.bind((IP, portnum))
    sever_socket.listen(0)
    print("Socket is ready")
    while True:
        client_socket, addr = sever_socket.accept()
        data = client_socket.recv(65535)
        data = data.decode()
    sever_socket.close()


t = threading.Thread(target=socket_listener, args=('127.0.0.1', 8888))

t.start()