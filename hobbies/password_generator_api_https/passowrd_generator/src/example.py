from libraries.Mongo_Password_Generator import Buscar_generar_clave, Users

USER = "wisrovi_4"
ORIGIN = Users.api
rta = Buscar_generar_clave(USER, ORIGIN)
print(rta)