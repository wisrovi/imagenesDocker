from rest_framework.response import Response
from django.contrib.auth.models import User
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.authtoken.models import Token
from django.contrib.auth.hashers import check_password

# Create your views here.


from django.http import HttpResponse


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
