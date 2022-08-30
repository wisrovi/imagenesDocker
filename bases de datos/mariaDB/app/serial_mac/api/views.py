from rest_framework.viewsets import ModelViewSet
from serial_mac.api.serializers import SerialSerializer
from serial_mac.models import Serial


class SerialApiViewSet(ModelViewSet):
    serializer_class = SerialSerializer
    queryset = Serial.objects.all()
