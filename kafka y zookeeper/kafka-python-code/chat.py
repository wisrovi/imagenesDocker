from kafka import KafkaProducer, KafkaConsumer
import threading
import sys

l = []

SERVIDOR_KAFKA = "localhost:9092"

name = sys.argv[1]

consumer = KafkaConsumer('messages', 
                    bootstrap_servers=SERVIDOR_KAFKA, 
                    group_id=None, 
                    auto_offset_reset='earliest')

def checkForMessages(consumer):
    while True:
        global l
        msg = next(consumer)
        l.append(msg)
thr = threading.Thread(target=checkForMessages, args=(consumer,))
thr.start()




from tkinter import *

class App:
    primer_envio = False
    def __init__(self, master):
        frame = Frame(master)
        frame.pack()
        S = Scrollbar(master)
        T = Text(master, height=30, width=100)
        S.pack(side=RIGHT, fill=Y)
        T.pack(side=LEFT, fill=Y)
        S.config(command=T.yview)
        T.config(yscrollcommand=S.set)
        quote = """Bienvenido al chat usando kafka.  """
        T.insert(END, quote)
        self.T = T
        self.entry = Entry(master)
        self.entry.bind("<Return>", self.evaluate)
        self.entry.pack()
        self.producer = KafkaProducer(bootstrap_servers=SERVIDOR_KAFKA)

    def send(self, msg):
        msg = msg.encode('utf-8')
        self.producer.send('messages', msg)
        self.primer_envio = True

    def evaluate(self, event):
        msg = name + ": " + self.entry.get()
        self.send(msg)
        self.entry.delete(0,END)

    def add(self, msg):   
        if self.primer_envio:   
            self.T.insert(END, "\n " + msg.value.decode())

    def checkmsg(self):
        global l
        while len(l) > 0:
            msg = l.pop()
            self.add(msg)      

root = Tk()
root.title(name)
app = App(root)

while True:
    root.update_idletasks()
    root.update()
    app.checkmsg()