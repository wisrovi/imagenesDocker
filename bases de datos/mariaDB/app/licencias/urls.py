"""licencias URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path, include
from core.serial_mac.api.router import router_serial
from rest_framework.authtoken.views import obtain_auth_token

from core.serial_mac.views import login, activation, licence

urlpatterns = [
    path('jet/', include('jet.urls', 'jet')),
    path('jet/dashboard/', include('jet.dashboard.urls', 'jet-dashboard')),

    path('admin/', admin.site.urls),

    path('serial_mac/', include("core.serial_mac.urls")),
    path('api/', include(router_serial.urls)),

    path('api-token-auth/', obtain_auth_token),

    path("login/", login),
    path("activation/", activation),
    path("licence/", licence)
]
