import threading
from Decoder import RUN

class myThread(threading.Thread):
	def __init__(self,threadID,name):
		threading.Thread.__init__(self)
		self.threadID=threadID # std value.
		self.name=name
		self.counter=counter
	def run(self):
		#threadLock.acquire()
		RUN(self.threadID,self.name)
		#threadLock.release()
