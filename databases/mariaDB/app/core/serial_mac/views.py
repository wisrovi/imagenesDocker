import datetime

from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.authtoken.models import Token
from django.contrib.auth.hashers import check_password

# Create your views here.


from django.http import HttpResponse

from config.config import formato
from config.mensajes.mensajes import licencia_nueva, no_hay_configuracion_servidor
from libraries.util import Util
from core.serial_mac.models import FileSincronizacion, Serial

util = Util()


@api_view(['POST'])
@authentication_classes([])
@permission_classes([])
def login(request):
    username = request.POST.get("username")
    password = request.POST.get("password")

    # 1 logeo con Smarti
    # 2 si Smarti dice que OK, entonces actualizo las credenciales del user en el sistema

    answer_smarti = True
    if not answer_smarti:
        return Response(dict(error="Credenciales invalidas"))

    try:
        user = User.objects.get(username=username)
    except User.DoesNotExist:
        # usuario no existe, pero como si existe en Smarti, entonces lo creo aca en el sistema
        User.objects.get_or_create(username=username, password=password)
        user = User.objects.get(username=username)

    if not check_password(password, user.password):
        # solo si la clave es distinta a la almacena, entonces la actualizo
        user.set_password(password)

    # creo el token asignado
    token, _ = Token.objects.get_or_create(user=user)

    response = dict(token=token.key)
    return Response(response)


@api_view(['POST'])
def activation(request):
    file = FileSincronizacion.objects.filter(active=True)
    old = None
    for f in file:
        if old is None:
            old = f
        else:
            if f.vence > old.vence:
                old = f

    activacion = dict(
        server=old.server,
        project=old.project,
        vence=old.vence.strftime(formato),
        user=old.user,
        password=old.password
    )

    info_save = util.dict_to_aes128(activacion)
    print(util.aes_to_json(info_save))

    return Response(dict(data=info_save))


@api_view(['POST'])
def licence(request):
    today = datetime.datetime.now()
    mac = request.POST.get("mac")
    project = request.POST.get("project")
    path = request.POST.get("path")
    computername = request.POST.get("computername")
    md5 = request.POST.get("md5")
    hash = request.POST.get("hash")
    checksum = request.POST.get("checksum")

    print(today, mac, project, path, computername, md5, hash, checksum)

    msg = ""
    try:
        licencia = Serial.objects.get(mac=mac)
        status = "old"
    except Serial.DoesNotExist:
        print("[licencia]: licencia no existe, creo una pero con fecha vencida")
        Serial.objects.get_or_create(
            mac=mac,
            project=project,
            path_instalation=path,
            created=datetime.datetime.now(),
            end=today,
            computername=computername,
            h_h=hash,
            h_m5=md5,
            h_c=checksum
        )
        licencia = Serial.objects.get(mac=mac)
        print(licencia.__dict__)
        status = "new"
        msg = licencia_nueva

    file = FileSincronizacion.objects.filter(active=True)
    old = None
    for f in file:
        if old is None:
            old = f
        else:
            if f.vence > old.vence:
                old = f

    if old is None:
        return Response(dict(error=no_hay_configuracion_servidor))

    answer = dict(
        COMPUTERNAME=licencia.computername,
        hash_md5=dict(
            checksum=licencia.h_c,
            hash=licencia.h_h,
            md5=licencia.h_m5
        ),
        mac=licencia.mac,
        path=licencia.path_instalation,
        project=licencia.project,
        server=old.server,
        create=licencia.created.strftime(formato),
        vence=licencia.end.strftime(formato),
        status=status,
        msg=msg
    )

    info_save = util.dict_to_aes128(answer)
    print(info_save)

    return Response(dict(data=info_save))


def index(request):
    return HttpResponse("Hello world!")


def home(request):
    return HttpResponse("Login required")


# @login_required(login_url='/home/')
def search(request):
    user_asking = ""
    if request.method == "POST":
        user_asking = User(request.POST)

    return HttpResponse("good request!" + user_asking.username)
