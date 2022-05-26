from libraries.Mongo_conection import collection, validar_existe_author, validar_existe_password
from Dto.Dto import *
from libraries.Password_generador import Generator_password

# ********************************   DATOS   ************************************
def Buscar_generar_clave(user, origin):
    user = user.upper()
    rta_user = validar_existe_author(user)
    if rta_user is not None:
        # print("Usuario ya existe:", rta_user)
        return rta_user['password']
    else:
        generator = Generator_password()
        new_password = generator.get_password()
        PASSWORD = new_password['basic']
        for _ in range(10):
            OK = False
            if validar_existe_password(PASSWORD) is None:
                new_password = Password(bas=new_password['basic'], sha=new_password['sha256'])
                data_save = Data(author=user, password=vars(new_password), origin=origin)
                data_save = vars(data_save) # data_save.__dict__  
                try:
                    post_id = collection.insert_one(data_save).inserted_id
                    # print("save:", data_save, "id:", post_id)
                    return vars(new_password)
                except Exception as e:
                    error_info = e.__dict__['_OperationFailure__details']['keyValue']
                    error_detail = e.__dict__['_OperationFailure__details']['errmsg'].split(":")[0]
                    print(error_info, "->", error_detail)
            if OK:
                break

if __name__ == "__main__":
    USER = "wisrovi"
    ORIGIN = Users.api
    rta = Buscar_generar_clave(USER, ORIGIN)
    print(rta)

        
