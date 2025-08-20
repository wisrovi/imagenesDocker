from tasks import task_suma, task_por2, task_parent
import time

if __name__ == "__main__":
    for _ in range(10):
        time.sleep(1)

        # Ejecuta task_suma y espera el resultado
        result_a = task_suma.delay(5, 7)  # Llama a task_suma con 5 y 7
        result_a_value = result_a.get(timeout=10)  # Espera el resultado de task_suma

        # Usa el resultado de task_suma como entrada para task_por2
        result_b = task_por2.delay(result_a_value)
        result_b_value = result_b.get(timeout=10)  # Espera el resultado de task_por2

        print(f"Resultado Final de task_por2: {result_b_value}")

        # Ejecuta la tarea principal que llama a otras tareas
        result = task_parent.delay(5)
        result_value = result.get(timeout=10)
        print(f"Resultado final de la tarea principal: {result_value}")
