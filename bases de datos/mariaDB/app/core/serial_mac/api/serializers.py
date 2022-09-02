from rest_framework.serializers import ModelSerializer
from core.serial_mac.models import Serial


class SerialSerializer(ModelSerializer):
    class Meta:
        model = Serial
        fields = '__all__'
