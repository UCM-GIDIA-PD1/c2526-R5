# Explicación de las variables escogidas para el DataSet conjunto
---
## Variables climáticas
- Sensación térmica
- Velocidad del viento: 
- Temperatura
Tras el análisis, se ve claramente que estas tres variables son las que realmente tienen un efecto sobre el comportamiento de los trenes. En orden de importancia, la sensación térmica aporta un 37% aproximado del peso relativo sobre la variable objetivo principal (el retraso del metro); la velocidad del viento, un 24%; y la temperatura un 22%.
Con esto, podemos hacer una conclusión adicional. Parece que afecta más el comportamiento de los usuarios a los trenes que el clima directamente, reflejado en la importancia de la sensación térmica frente a la temperatura real. 

Descartamos: la visibilidad (cloud cover), lluvia, precipitación, precipitaciones acumuladas, y las booleanas de is_freezing e is_high_wind por lo poco que afectan a los trenes (visto en todas las gráficas del análisis). Entre todas apenas suman un 17% del peso relativo sobre la variable objetivo.
