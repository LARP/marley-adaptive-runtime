# Directivas de Proyecto: Marley Runtime

## Protocolo de Comando Git (`-git`)
Cuando el usuario incluya el comando o bandera `-git` en su mensaje (por ejemplo: `agrega feature X -git`, `-git: mensaje`, o simplemente `-git`):

1. **Inspeccionar estado:** Ejecutar `git status` y `git diff` para revisar los cambios pendientes.
2. **Generar descripción detallada:** Elaborar un mensaje de commit claro y explicativo que detalle qué se modificó y por qué (o incorporar la descripción proporcionada por el usuario si la hay).
3. **Preparar y confirmar cambios:** Ejecutar `git add .` y `git commit -m "<título>" -m "<descripción de los cambios>"`.
4. **Subir a GitHub:** Ejecutar `git push origin <rama_actual>` para sincronizar con el repositorio remoto.
5. **Confirmar resultado:** Reportar al usuario el hash del commit, el mensaje utilizado y los archivos enviados.
