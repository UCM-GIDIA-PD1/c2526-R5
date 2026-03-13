# Explicación de las variables escogidas para el DataSet conjunto

Este documento detalla la selección de variables (feature selection) para la fase de modelado del proyecto. Tras realizar el Análisis Exploratorio de Datos (EDA) en la Fase 2, hemos filtrado las variables originales para mantener aquellas con mayor capacidad predictiva y eliminar el ruido, asegurando así modelos más eficientes y ligeros para la Fase 3.

---
## Variables de trenes (GTFS)
Las observaciones para este tipo de datos se definen como la llegada de un tren a una parada específica (match_key + stop_id).

### Variables que se quedan:
- **`date`** / **`match_key`** / **`stop_id`** / **`route_id`** / **`direction`**: Claves referenciales y dimensiones cruzadas para localizar tiempo/espacio en el modelo de forma inequívoca.
- **`delay_seconds`**: Retraso del tren en la parada.
- **`lagged_delay_1` / `lagged_delay_2`**: Los predictores más potentes para el corto plazo (inercia temporal del retraso).
- **`route_rolling_delay`**: Indispensable para predicciones a medio plazo sobre la congestión de la red.
- **`actual_headway_seconds`**: Clave para detectar anomalías de frecuencia entre trenes.
- **`is_unscheduled`**: Señal binaria de refuerzos operativos en tiempo real de la MTA.
- **`hour_sin` / `hour_cos`**: Codificación cíclica temporal, transformando horas absolutas de forma útil para los modelos.
- **`dow` / `is_weekend`**: Capturan el comportamiento y tráfico de pasajeros drásticamente diferente en días libres.
- **`scheduled_time_to_end`** / **`stops_to_end`**: Variables enfocadas hacia el tramo final, permitiendo discriminar dónde se encuentra un tren en la ruta generalizada.
- **`station_delay_10m` a `station_delay_30m`**: Agrupación zonal referenciando tendencias focalizadas en demoras de la estación.
- **`merge_time`**: Referencia general del cruce temporal.

### Variables objetivo (Targets):
Se mantienen todos los horizontes de predicción para permitir diferentes enfoques estructurados:
- **`target_delay_10m` hasta `target_delay_60m` y `target_delay_end`**: Predicción del retraso absoluto en paradas venideras.
- **`delta_delay_10m` hasta `delta_delay_60m` y `delta_delay_end`**: Predicción escalonada de la variación (recuperación o empeoramiento) del retraso respecto al instante actual.

### Variables que se van:
- **`trip_uid`**: Eliminada enteramente por redundancia ante `match_key`.
- **`scheduled_time` / `actual_time`**: Sirven originariamente para el cruce asíncrono, pero se descartan una vez la información es asimilada en las cíclicas y en `merge_time`.
- **`trip_progress`**: Descartada por complejidad entre sub-rutas dinámicas que difieren grandemente.
- **`scheduled_seconds` / `actual_seconds`**: Descartadas por redundancia semántica tras normalizarse en intervalos más explicativos.
- **`delay_minutes`**: Eliminada a favor de mantener todo el flujo temporal consolidado en la métrica continua base (`*_seconds`).

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

---
## Variables de eventos

### Variables que se incorporan:
- **`n_eventos_afectando`**: Cantidad de eventos masivos activos y solapados a lo largo del paso del tren.
- **`tipo_referente`**: Categorización focalizada en el tipo de evento principal (aquel de mayor afectación y relevancia implícita o 'score').
- **`afecta_previo`**, **`afecta_durante`**, **`afecta_despues`**: Indicadores bandera determinando el instante cronológico de afectación de los viandantes que interactúan con el metro en torno a dichos eventos.

Tras el análisis de cómo perturban los eventos a la red de transporte, logramos estructurar estas agregaciones. Permiten sumar información contextual imprescindible (diferenciando flujos en embudo previos vs masificación paralela) limitando de manera eficiente la dimensionalidad sin recargar el motor algorítmico posterior.

### Variables que se descartan:
- **`paradas_afectadas`**, **`parada_nombre`**, **`parada_lineas`**: Se utilizan transversalmente y únicamente en la estrategia inicial del merge geo-espacial, descartándose con posterioridad.
- **`score`**: Valor originario en crudo que dirime transitoriamente el cálculo de "tipo_referente". Descartado por ser información instrumental.
- **Resto de metadatos tabulados de evento**: Desechados ya que heredan las representaciones temporales y estacionales generadas para base en GTFS (haciéndose redundantes o residuales).

---
## Variables de alertas

### Variables que se incorporan:
- **`category`**: Tipología categórica general de la alerta registrada.
- **`num_updates`**: Número de iteraciones que ha sufrido la incidencia en sistema.
- **`timestamp_start`**: Sello temporal inicial base de la publicación oficial.
- **`seconds_since_last_alert`**: Ventana cronometrada en segundos transcurridos desde la publicación subyacente de notificación a una alerta. 
- **`is_alert_just_published`**: Booleano reaccionario si la difusión acaba de sucederse de forma inminente (<= 60 segundos).
- **`seconds_to_next_alert`**: Margen temporal en segundos para manifestarse la retransmisión paralela o de relevo colindante (forecast a futuro útil para modelos predictivos).
- **`alert_in_next_15m`** / **`alert_in_next_30m`**: Banderas proyectivas que alertan sobre la probabilidad de proximidad a perturbaciones inminentes en el sistema o reportes sistémicos.

### Variables que se descartan:
- **`lines`**: Aprovechada inicial y fundamentalmente para la segmentación vía 'explode' enlazando múltiples rutas conectadas, y omitida ulteriormente de acuerdo a la granularidad preprocesada.
- **`event_id`**: Identificador arbitrario instrumental sin rastro predictivo real.
- **`timestamp_end`**: Estimulaba ruido confusional, dado que no expone el final resolutivo de una incidencia, limitándose a proyectar la estimación de validez u última constancia de modificación oficial. 
- **`text_snippet` / `description`**: Cadenas descriptivas textuales libres descartadas proactivamente al ya delegar su representatividad abstracta en `category`. Evitar columnas no estructuradas optimiza drásticamente todo el procesamiento algorítmico tabular posterior.










