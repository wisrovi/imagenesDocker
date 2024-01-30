// Obtener el valor de se_adjunto_archivo del atributo de datos del body
const seAdjuntoArchivo = document.body.getAttribute('data-se-adjunto-archivo');

// Obtener todos los contenedores de las tarjetas
const tarjetas = document.querySelectorAll('.card-body');

// Iterar sobre cada tarjeta
tarjetas.forEach(tarjeta => {
    // Obtener elementos dentro de la tarjeta actual
    const guardarBtn = tarjeta.querySelector('.guardarBtn');
    const mensaje = tarjeta.querySelector('.mensaje');

    // Agregar evento de clic al botón
    guardarBtn.addEventListener('click', function() {
        // Verificar si se ha adjuntado un archivo
        if (!seAdjuntoArchivo) {
            mensaje.style.display = 'block'; // Mostrar el mensaje
            return; // Salir de la función si no se ha adjuntado un archivo
        }
    });

    // Agregar evento de mouseover al botón
    guardarBtn.addEventListener('mouseover', function() {
        // Verificar si se ha adjuntado un archivo
        if (seAdjuntoArchivo) {
            guardarBtn.innerHTML = '<img src="https://images.freeimages.com/clg/images/46/469537/play-audio-button-set-clip-art_f.jpg" alt="Guardar" width="200" height="200"><span>Entrenar</span>';
        }
    });

    // Agregar evento de mouseout al botón
    guardarBtn.addEventListener('mouseout', function() {
        // Verificar si se ha adjuntado un archivo
        if (seAdjuntoArchivo) {
            guardarBtn.innerHTML = '<img src="https://images.freeimages.com/clg/images/46/469537/play-audio-button-set-clip-art_f.jpg" alt="Guardar" width="100" height="100">';
        }
    });
});
