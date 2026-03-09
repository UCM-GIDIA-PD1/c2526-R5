# Explicación de las variables escogidas para el DataSet conjunto

Este documento detalla la selección de variables (feature selection) para la fase de modelado del proyecto. Tras realizar el Análisis Exploratorio de Datos (EDA) en la Fase 2, hemos filtrado las variables originales para mantener aquellas con mayor capacidad predictiva y eliminar el ruido, asegurando así modelos más eficientes y ligeros para la Fase 3.

---
## Variables de trenes (GTFS)
Las observaciones para este tipo de datos se definen como la llegada de un tren a una parada específica (match_key + stop_id).

### Variables que se quedan:
- **`route_id`**: Crucial para diferenciar el comportamiento por línea (exprés vs. local).
- **`lagged_delay_1` / `lagged_delay_2`**: Los predictores más potentes para el corto plazo (inercia del retraso).
- **`route_rolling_delay`**: Indispensable para predicciones a medio plazo (30-60 min).
- **`actual_headway_seconds`**: Clave para detectar anomalías y predecir avisos.
- **`is_unscheduled`**: Señal de refuerzos operativos de la MTA.
- **`hour_sin` / `hour_cos`**: Codificación cíclica que sustituye a las horas absolutas para el aprendizaje del modelo.
- **`dow` / `is_weekend`**: Capturan la diferencia de comportamiento en fines de semana.

### Variables objetivo (Targets):
Se mantienen todos los horizontes de predicción para permitir diferentes enfoques de modelo:
- **`target_delay_10m` hasta `target_delay_60m`**: Predicción del retraso absoluto en paradas futuras.
- **`delta_delay_10m` hasta `delta_delay_60m`**: Predicción de la variación (incremento o recuperación) del retraso respecto al momento actual.

### Variables que se van:
- **`trip_uid`**: Eliminada por redundancia con match_key
- **`scheduled_time` / `actual_time`**: Estas variables se usan para el cruce de datasets (merge con clima y eventos), pero se eliminan del dataset final de entrenamiento ya que su información queda contenida en las variables cíclicas de hora.
- **`stops_to_end` / `trip_progress`**: Se descartan debido a la inconsistencia entre líneas; el número de paradas restantes tiene significados distintos dependiendo de la longitud de la ruta, y nuestro análisis demuestra que la congestión de la red (`rolling_delay`) es un predictor mucho más robusto.
* **`scheduled_seconds` / `actual_seconds`**: Descartadas por redundancia.
* **`delay_minutes`**: Eliminada por redundancia con la variable en segundos.

---

## Variables climáticas (OpenMeteo)

### Variables que se incorporan:
- **`temp_extreme`**

### Variables que se quitan:
- **`Cloud Cover`**
- **`Precipitation`**
- **`precip_3h_accum`**
- **`Snow`**
- **`is_high_wind`**
- **`is_freezing`**
- **`apparent_temp`** 
- **`Wind Speed`** 
- **`Temperature`**

Tras el análisis, se ve claramente que estas tres variables son las que realmente tienen un efecto sobre el comportamiento de los trenes. En orden de importancia, la sensación térmica aporta un 37% aproximado del peso relativo sobre la variable objetivo principal (el retraso del metro); la velocidad del viento, un 24%; y la temperatura un 22%.
Con esto, podemos hacer una conclusión adicional. Parece que afecta más el comportamiento de los usuarios a los trenes que el clima directamente, reflejado en la importancia de la sensación térmica frente a la temperatura real. 


Descartamos la visibilidad (cloud cover), lluvia, precipitación, precipitaciones acumuladas, y las booleanas de is_freezing e is_high_wind por lo poco que afectan a los trenes (visto en todas las gráficas del análisis). Entre todas apenas suman un 17% del peso relativo sobre la variable objetivo.

---
## Variables de eventos

-Hubo evento en el día (hubo_evento_en_el_dia)
-Cantidad de eventos (n_eventos)
-Tipo de evento principal (tipo_evento_prioritario)
-Fases de impacto del evento (durante_entrada, durante_evento, despues_evento)
-Evento nocturno (evento_nocturno)

Tras el analisis de como afectan los eventos a la red de transporte, hemos sacado las siguientes variables que nos permiten añadir información valiosa al dataset principal, sin expandirlo de tamaño. Hubo evento en el dia nos indica si algún evento va a priori influir a la red, la cantidad de eventos nos indica cuantos han habido en el dia, el tipo de evento principal se elige el que mayor score tenga, que va a ser el que más afectación tenga sobre la red. Las fases de impacto nos diferencian el momento del evento que es clave por ejemplo en eventos con salidas masivas de espectadores y por último si es un evento nocturno debido a que a altas horas de la noche hay menos trenes de circulación, junto a una fuerte propagación del retraso, por lo cual puede ser una métrica util para los modelos.

Descartamos: las paradas afectadas y sus lineas, que van a ser usadas para el cruce de datasets, el score que nos va a servir para elegir el evento prioritario, pero después no aporta nada. El resto de columnas son compartidas con el dataset principal, como por ejemplo el dia, hora.

---
## Variables de alertas

- Categoría de la alerta (category)
- Número de actualizaciones que ha tenido la alerta (num_updates)
- Líneas afectadas (lines)
- Fecha y hora exacta de la publicación de la alerta (timestamp_start)
- Minutos / segundos desde la última publicación de alerta (seconds_since_last_alert)

Tras el análisis exploratorio , consideramos suficientes estas variables para las alertas. Hay que hacer un explode de la columna 'lines' en el dataframe de alertas antes del join, para tener una fila por línea afectada, y entonces poder cruzarlo correctamente con GTFS. La variable 'seconds_since_last_alert' muestra para cada observación , el tiempo en segundos que ha transcurrido desde la última publicación de una alerta. Nos hemos desprendido de la variable 'event_id' ya que es un identificador que no nos aporta información útil para modelar. Descartamos 'timestamp_end', ya que puede ser malinterpretada, porque no determina el fin de una alerta sino el momento de publicación de la última actualización (si tiene) de una alerta, por lo que si una alerta no tiene actualizaciones el timestamp_start y el end es el mismo, lo que puede llevar a confusión.'Text_snippet' y 'description' también se van porque son texto libre y no hemos hallado nada suficientemente útil procesándolas.










