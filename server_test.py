import random
import socket

host='localhost'
port=random.randrange(9000,9020)
addr=(host,port)
s=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
s.bind(addr)
s.listen(5)
s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
while True:
	print('Server waiting for connection  '+ socket.gethostbyname(socket.gethostname())  +'  '+str(port))
	c,a=s.accept()
	print('client connected ',a)
	while True:
		try:
			data=c.recv(1024)
		except Exception as e:
			print(str(e))   
		c.send(bytes('Connected to server : vecho','utf-8'))
		if not data or data.decode('utf-8')=='END':
			break

		#print('Recieved from client : %s'%data.decode('utf-8'))
		
		print(data.decode('utf-8'))
		x = data.decode('utf-8')
		if x=="'a'":
			print('in') 
		
		#except KeyboardInterrupt:
		#   print('Exited by user')
	c.close()
s.close()