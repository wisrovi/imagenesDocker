import redis
import datetime

today = datetime.date.today()
today = today.isoformat() # '2017-01-01'

# crear conexion
SERVER = "10.1.31.2" # "localhost"

try:
    r = redis.StrictRedis(host=SERVER, port=6379, db=0)
    r.ping()
    print("Conectado a Redis al servidor: {}".format(SERVER))
except:
    print("No se pudo conectar a Redis en %s" % SERVER)
    exit()




# leer
print("keys:", r.keys())
print('foo',r.get('foo'))
print("Bahamas", r.get("Bahamas"))
print("smembers", r.smembers(today))
print("card", r.scard(today))



# escribir
r.mset(
    {
        "Croatia": "Zagreb", 
        "Bahamas": "Nassau"
    })
r.set('foo', 'bar')
visitors = {"dan", "jon", "alex"}
r.sadd(today, *visitors)

