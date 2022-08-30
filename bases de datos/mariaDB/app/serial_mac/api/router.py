from rest_framework.routers import DefaultRouter
from serial_mac.api.views import SerialApiViewSet

router_serial = DefaultRouter()

router_serial.register(prefix='post', basename='post', viewset=SerialApiViewSet)
