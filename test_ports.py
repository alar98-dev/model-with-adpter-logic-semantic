import socket

ports = [22, 5432, 8000, 8022, 8080, 8800]
for p in ports:
    s = socket.socket()
    s.settimeout(1)
    res = s.connect_ex(('192.168.10.107', p))
    s.close()
    st = "ABERTA 🟢" if res == 0 else "FECHADA 🔴"
    print(f"Porta {p} em 192.168.10.107: {st}")
