from local_aplication.libraries.Mongo_Password_Generator import Buscar_generar_clave, Users

USER = "wisrovi_3"
ORIGIN = Users.api
rta = Buscar_generar_clave(USER, ORIGIN)
print(rta)