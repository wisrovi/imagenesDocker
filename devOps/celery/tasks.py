"""
pip install celery redis flower
"""

from celery import Celery, chain

# Configuración de Celery con Redis como broker y backend
app = Celery('tasks', broker='redis://localhost:6379/0', backend='redis://localhost:6379/0')


@app.task
def task_suma(x, y):
    """
    Suma dos números.
    """
    return x + y


@app.task
def task_por2(x):
    """
    Multiplica un número por 2.
    """
    return x * 2


@app.task
def task_parent(value):
    """
    Tarea principal que ejecuta una cadena de dos tareas hijas.
    """
    print(f"Executing Parent Task with value: {value}")

    # Encadenar las tareas hijas utilizando chain, sin bloquear con get().
    task_chain = chain(task_suma.s(value, value), task_por2.s())
    result = task_chain()

    # El resultado será manejado asincrónicamente sin bloquear.
    return result


if __name__ == "__main__":
    print(task_suma(3, 2))
    print(task_por2(3))



"""
# poner disponibles las tareas

celery -A tasks worker --loglevel=info


# iniciar flower
celery -A tasks flower --port=5555

"""