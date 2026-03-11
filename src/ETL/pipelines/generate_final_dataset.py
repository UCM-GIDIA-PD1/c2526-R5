"""
Construye el dataset final de entrenamiento a partir de las capas limpias en MinIO.

Fuentes mezcladas en un único dataset:
1) GTFS cleaned (scheduled + unscheduled)
2) Clima cleaned
3) Eventos cleaned
4) Alertas oficiales cleaned

Uso:
  uv run python -m src.ETL.pipelines.generate_final_dataset \
	--start 2025-01-01 --end 2025-01-31
"""

import argparse
import gc
import os
from calendar import monthrange
from datetime import date, datetime, timedelta
from typing import Iterable
import numpy as np
import pandas as pd

from src.common import minio_client


class MinIODataClient:
	"""Cliente para operaciones de lectura/escritura de DataFrames Parquet en MinIO."""

	def __init__(self, access_key: str, secret_key: str, endpoint: str | None = None, bucket: str | None = None) -> None:
		self.access_key = access_key
		self.secret_key = secret_key
		self.endpoint = endpoint or minio_client.DEFAULT_ENDPOINT
		self.bucket = bucket or minio_client.DEFAULT_BUCKET

	def download_df_parquet(self, object_name: str) -> pd.DataFrame:
		"""Descarga un objeto Parquet de MinIO y lo devuelve como DataFrame."""
		return minio_client.download_df_parquet(
			access_key=self.access_key,
			secret_key=self.secret_key,
			object_name=object_name,
			endpoint=self.endpoint,
			bucket=self.bucket,
		)

	def upload_df_parquet(self, object_name: str, df: pd.DataFrame) -> None:
		"""Sube un DataFrame a MinIO serializándolo como Parquet."""
		minio_client.upload_df_parquet(
			access_key=self.access_key,
			secret_key=self.secret_key,
			object_name=object_name,
			df=df,
			endpoint=self.endpoint,
			bucket=self.bucket,
		)

	def upload_json(self, object_name: str, data: object) -> None:
		"""Sube un objeto JSON serializable a MinIO."""
		minio_client.upload_json(
			access_key=self.access_key,
			secret_key=self.secret_key,
			object_name=object_name,
			data=data,
			endpoint=self.endpoint,
			bucket=self.bucket,
		)


OUTPUT_OBJECT = "grupo5/final/dataset_final_entrenamiento.parquet"


def iterate_dates(start: date, end: date) -> Iterable[str]:
	"""Genera fechas en formato YYYY-MM-DD para un rango inclusivo."""
	cur = start
	while cur <= end:
		yield cur.strftime("%Y-%m-%d")
		cur += timedelta(days=1)


def safe_download(client: MinIODataClient, object_name: str) -> pd.DataFrame:
	"""Descarga segura: ante error o nulo, devuelve DataFrame vacío."""
	try:
		df = client.download_df_parquet(object_name)
		if df is None:
			return pd.DataFrame()
		return df
	except Exception:
		return pd.DataFrame()


def load_gtfs(client: MinIODataClient, days: list[str]) -> pd.DataFrame:
	"""Carga GTFS cleaned (scheduled + unscheduled) y concatena por rango de fechas."""
	frames: list[pd.DataFrame] = []
	for day in days:
		obj_sched = f"grupo5/cleaned/gtfs_clean_scheduled/date={day}/gtfs_scheduled_{day}.parquet"
		obj_uns = f"grupo5/cleaned/gtfs_clean_unscheduled/date={day}/gtfs_unscheduled_{day}.parquet"
		df_sched = safe_download(client, obj_sched)
		df_uns = safe_download(client, obj_uns)
		if not df_sched.empty:
			frames.append(df_sched)
		if not df_uns.empty:
			frames.append(df_uns)
	if not frames:
		raise ValueError("No GTFS cleaned data found in MinIO for requested range.")
	out = pd.concat(frames, ignore_index=True)
	return out


def load_weather(client: MinIODataClient, days: list[str]) -> pd.DataFrame:
	"""Carga clima exclusivamente desde cleaned/clima_clean."""
	frames: list[pd.DataFrame] = []
	for day in days:
		obj_cleaned = f"grupo5/cleaned/clima_clean/date={day}/clima_{day}.parquet"
		df = safe_download(client, obj_cleaned)
		if not df.empty:
			frames.append(df)
	if not frames:
		return pd.DataFrame()
	return pd.concat(frames, ignore_index=True)


def load_events(client: MinIODataClient, days: list[str]) -> pd.DataFrame:
	"""Carga eventos exclusivamente desde cleaned/eventos_nyc."""
	frames: list[pd.DataFrame] = []
	for day in days:
		obj_cleaned = f"grupo5/cleaned/eventos_nyc/date={day}/eventos_{day}.parquet"
		df = safe_download(client, obj_cleaned)
		if not df.empty:
			frames.append(df)
	if not frames:
		return pd.DataFrame()
	return pd.concat(frames, ignore_index=True)


def load_alerts(client: MinIODataClient, days: list[str]) -> pd.DataFrame:
	"""Carga alertas oficiales exclusivamente desde cleaned/official_alerts."""
	frames: list[pd.DataFrame] = []
	for day in days:
		obj_cleaned = f"grupo5/cleaned/official_alerts/date={day}/alerts.parquet"
		df = safe_download(client, obj_cleaned)
		if not df.empty:
			frames.append(df)
	if not frames:
		return pd.DataFrame()
	return pd.concat(frames, ignore_index=True)


def normalize_route_id(series: pd.Series) -> pd.Series:
	"""Normaliza códigos de línea/ruta (trim + mayúsculas)."""
	return series.astype("string").str.strip().str.upper()


def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
	"""
	Reduce uso de RAM mediante:
	- object/string -> category (si cardinalidad relativa < 0.5)
	- numéricas -> int32/float32 cuando es seguro
	"""
	if df.empty:
		return df

	for col in df.columns:
		col_data = df[col]
		dtype = col_data.dtype

		# object/string: convertir a category si hay baja cardinalidad.
		if pd.api.types.is_object_dtype(dtype) or pd.api.types.is_string_dtype(dtype):
			total = len(col_data)
			if total == 0:
				continue
			try:
				unique = col_data.nunique(dropna=True)
			except TypeError:
				# Evita fallar con objetos no hasheables (listas/dicts).
				continue
			ratio = unique / total
			if ratio < 0.5:
				df[col] = col_data.astype("category")
			continue

		# Evitar tocar datetimes/bools/categories
		if (
			pd.api.types.is_datetime64_any_dtype(dtype)
			or pd.api.types.is_bool_dtype(dtype)
			or isinstance(dtype, pd.CategoricalDtype)
		):
			continue

		# Numéricas: downcast seguro
		if pd.api.types.is_numeric_dtype(dtype):
			c_min = col_data.min(skipna=True)
			c_max = col_data.max(skipna=True)
			if pd.isna(c_min) or pd.isna(c_max):
				continue

			if pd.api.types.is_integer_dtype(dtype):
				if c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
					if pd.api.types.is_extension_array_dtype(dtype):
						df[col] = col_data.astype("Int32")
					else:
						df[col] = col_data.astype(np.int32)
			else:
				if c_min >= np.finfo(np.float32).min and c_max <= np.finfo(np.float32).max:
					df[col] = col_data.astype(np.float32)

	return df


def prepare_gtfs(df_gtfs: pd.DataFrame) -> pd.DataFrame:
	"""
	Prepara GTFS para los cruces:
	- Normaliza fecha de servicio a `date`.
	- Construye `merge_time` de forma segura con fallback de hora real a programada.
	- Extrae `hour` para join con clima.
	"""
	df = df_gtfs.copy()

	if "service_date" in df.columns and "date" not in df.columns:
		df["date"] = pd.to_datetime(df["service_date"], errors="coerce").dt.date
	elif "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
	else:
		raise ValueError("GTFS requires either 'service_date' or 'date' to merge.")

	if "scheduled_time" not in df.columns and "actual_time" not in df.columns:
		raise ValueError("GTFS requires 'scheduled_time' or 'actual_time' for temporal merges.")

	if "actual_time" in df.columns and "scheduled_time" in df.columns:
		# Requisito: crear tiempo base seguro rellenando NaN de actual_time con scheduled_time.
		base_time = df["actual_time"].fillna(df["scheduled_time"])
	elif "actual_time" in df.columns:
		base_time = df["actual_time"]
	else:
		base_time = df["scheduled_time"]

	df["merge_time"] = pd.to_datetime(
		df["date"].astype("string") + " " + base_time.astype("string"),
		errors="coerce",
	)

	# Mantener compatibilidad con posibles usos históricos de train_timestamp.
	df["train_timestamp"] = df["merge_time"]
	df["hour"] = df["merge_time"].dt.hour
	if "route_id" in df.columns:
		df["route_id"] = normalize_route_id(df["route_id"])
	if "stop_id" in df.columns:
		df["stop_id"] = df["stop_id"].astype("string")
	return df


def prepare_weather(df_weather: pd.DataFrame) -> pd.DataFrame:
	"""
	Prepara clima para merge por `date` + `hour`.
	Asume que `temp_extreme` ya existe en cleaned/clima_clean.
	"""
	df = df_weather.copy()
	if df.empty:
		return df

	if "Date" in df.columns:
		dt = pd.to_datetime(df["Date"], errors="coerce")
		df["date"] = dt.dt.date
		if "hour" not in df.columns:
			df["hour"] = dt.dt.hour
	elif "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
		if "hour" not in df.columns:
			raise ValueError("Weather data must include 'hour' when 'Date' is absent.")
	else:
		raise ValueError("Weather data requires 'Date' or 'date'.")

	if "temp_extreme" not in df.columns:
		raise ValueError("Se esperaba la columna 'temp_extreme' en cleaned/clima_clean.")

	return df


def _looks_like_stop_id(value: object) -> bool:
	"""Heurística mínima para detectar un stop_id típico del metro (sufijo N/S)."""
	if not isinstance(value, str):
		return False
	val = value.strip().upper()
	if len(val) < 2:
		return False
	return val.endswith(("N", "S"))


def _extract_stop_ids_from_paradas(value: object) -> list[str]:
	"""Extrae stop_ids desde estructuras anidadas de `paradas_afectadas`."""
	if value is None or (isinstance(value, float) and pd.isna(value)):
		return []
	if not isinstance(value, (list, tuple)):
		return []

	out: list[str] = []
	entries = list(value)
	for entry in entries:
		if isinstance(entry, str) and _looks_like_stop_id(entry):
			out.append(entry.strip().upper())
			continue

		if isinstance(entry, (list, tuple)):
			for item in entry:
				if isinstance(item, str) and _looks_like_stop_id(item):
					out.append(item.strip().upper())
				elif isinstance(item, (list, tuple)):
					for nested in item:
						if isinstance(nested, str) and _looks_like_stop_id(nested):
							out.append(nested.strip().upper())

	return sorted(set(out))


def _split_lines(value: object) -> list[str]:
	"""Convierte el campo `lines` (string/lista) en lista limpia de route_id."""
	if value is None or (isinstance(value, float) and pd.isna(value)):
		return []
	if isinstance(value, list):
		out = [str(x).strip().upper() for x in value if str(x).strip()]
		return out
	if isinstance(value, str):
		raw = value.replace("|", ",")
		parts = [p.strip().upper() for p in raw.split(",") if p.strip()]
		return parts
	return []


def prepare_alerts(df_alerts: pd.DataFrame) -> pd.DataFrame:
	"""
	Preprocesa alertas oficiales:
	- Estandariza timestamp y claves temporales.
	- Hace explode de líneas afectadas.
	- Normaliza columnas para el merge posterior con GTFS.
	"""
	df = df_alerts.copy()
	if df.empty:
		return df

	if "timestamp_start" not in df.columns and "date" in df.columns:
		df = df.rename(columns={"date": "timestamp_start"})
	if "timestamp_start" not in df.columns:
		raise ValueError("Alerts data requires 'timestamp_start' (or source 'date').")

	df["timestamp_start"] = pd.to_datetime(df["timestamp_start"], errors="coerce")
	df = df.dropna(subset=["timestamp_start"]).copy()
	df["date"] = df["timestamp_start"].dt.date
	df["hour"] = df["timestamp_start"].dt.hour

	if "lines" not in df.columns:
		# Si no hay líneas afectadas, dejamos listas vacías para mantener el contrato.
		df["lines"] = [[] for _ in range(len(df))]
	else:
		df["lines"] = df["lines"].apply(_split_lines)

	df = df.explode("lines", ignore_index=True)
	df = df.dropna(subset=["lines"]).copy()
	df["route_id"] = normalize_route_id(df["lines"])

	if "num_updates" not in df.columns:
		# Si no viene calculado, se estima por repeticiones del event_id.
		if "event_id" in df.columns:
			counts = df.groupby("event_id")["timestamp_start"].transform("count")
			df["num_updates"] = (counts - 1).clip(lower=0)
		else:
			df["num_updates"] = 0
	df["num_updates"] = pd.to_numeric(df["num_updates"], errors="coerce").fillna(0)

	# Orden estable para merge temporal posterior.
	df = df.sort_values(["route_id", "timestamp_start"])

	return df


def prepare_events(df_events: pd.DataFrame) -> pd.DataFrame:
	"""
	Preprocesa eventos:
	- Estandariza fechas y score.
	- Garantiza claves para merge (`date`, `stop_id`).
	- Intenta reconstruir `stop_id` desde `paradas_afectadas` cuando falta.
	"""
	df = df_events.copy()
	if df.empty:
		return df

	if "fecha_inicio" in df.columns:
		df["fecha_inicio"] = pd.to_datetime(df["fecha_inicio"], errors="coerce")
	if "fecha_final" in df.columns:
		df["fecha_final"] = pd.to_datetime(df["fecha_final"], errors="coerce")
	if "fecha_final" not in df.columns:
		df["fecha_final"] = df.get("fecha_inicio")
	df["fecha_final"] = df["fecha_final"].fillna(df["fecha_inicio"])

	if "score" not in df.columns:
		df["score"] = 0.0
	df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0)

	# Clave diaria para cruzar con GTFS.
	if "fecha_inicio" in df.columns:
		df["date"] = df["fecha_inicio"].dt.date
	elif "date" in df.columns:
		df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
	else:
		raise ValueError("Events data requires 'fecha_inicio' or 'date'.")

	if "tipo" not in df.columns:
		df["tipo"] = "Desconocido"
	if "nombre_evento" not in df.columns:
		df["nombre_evento"] = "evento_sin_nombre"

	if "stop_id" not in df.columns and "paradas_afectadas" in df.columns:
		# En algunos datasets de eventos la parada viene embebida en listas/tuplas.
		df["stop_ids"] = df["paradas_afectadas"].apply(_extract_stop_ids_from_paradas)
		df = df.explode("stop_ids", ignore_index=True)
		df = df.rename(columns={"stop_ids": "stop_id"})

	if "stop_id" in df.columns:
		df["stop_id"] = df["stop_id"].astype("string")

	return df


def merge_gtfs_weather(df_gtfs: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
	"""LEFT JOIN GTFS + Clima por `date` y `hour` (GTFS como base)."""
	if df_weather.empty:
		return df_gtfs.copy()
	out = df_gtfs.merge(df_weather, how="left", on=["date", "hour"], suffixes=("", "_weather"))
	return out


def merge_gtfs_events(df_base: pd.DataFrame, df_events: pd.DataFrame) -> pd.DataFrame:
	"""
	LEFT JOIN GTFS + Eventos por `date` y `stop_id`.
	Explota `paradas_afectadas` y conserva el evento prioritario por tren/parada.
	"""
	if df_events.empty:
		out = df_base.copy()
		out["hubo_evento_en_el_dia"] = 0
		out["n_eventos"] = 0
		out["tipo_evento_prioritario"] = "Ninguno"
		out["durante_entrada"] = 0
		out["durante_evento"] = 0
		out["despues_evento"] = 0
		out["evento_nocturno"] = 0
		return out

	base = df_base.copy()
	evt = df_events.copy()
	if "stop_id" in base.columns:
		base["stop_id"] = base["stop_id"].astype("string")

	if "date" in evt.columns:
		evt["date"] = pd.to_datetime(evt["date"], errors="coerce").dt.date
	elif "fecha_inicio" in evt.columns:
		evt["date"] = pd.to_datetime(evt["fecha_inicio"], errors="coerce").dt.date
	else:
		raise ValueError("Events data requires 'date' or 'fecha_inicio'.")

	if "paradas_afectadas" in evt.columns:
		evt = evt.explode("paradas_afectadas", ignore_index=True)
		evt = evt.rename(columns={"paradas_afectadas": "stop_id"})
	elif "stop_id" not in evt.columns:
		raise ValueError("Events data requires 'paradas_afectadas' or 'stop_id' for spatial merge.")

	evt["stop_id"] = evt["stop_id"].astype("string")

	tipo_col = "tipo" if "tipo" in evt.columns else "tipo_evento"
	if tipo_col not in evt.columns:
		evt[tipo_col] = "Desconocido"

	if "nombre_evento" not in evt.columns:
		evt["nombre_evento"] = "evento_sin_nombre"
	if "score" not in evt.columns:
		evt["score"] = 0.0
	evt["score"] = pd.to_numeric(evt["score"], errors="coerce").fillna(0.0)
	if "hora_inicio" not in evt.columns:
		evt["hora_inicio"] = "00:00:00"
	if "hora_salida_estimada" not in evt.columns:
		evt["hora_salida_estimada"] = "23:59:59"

	if "merge_time" not in base.columns:
		raise ValueError("GTFS base requires 'merge_time' for temporal event features.")
	base["merge_time"] = pd.to_datetime(base["merge_time"], errors="coerce")
	base["date"] = pd.to_datetime(base["date"], errors="coerce").dt.date

	evt["inicio_dt"] = pd.to_datetime(
		evt["date"].astype("string") + " " + evt["hora_inicio"].astype("string"),
		errors="coerce",
	)
	if "fecha_final" in evt.columns:
		evt["fecha_final"] = pd.to_datetime(evt["fecha_final"], errors="coerce").dt.date
	else:
		evt["fecha_final"] = evt["date"]
	evt["fin_dt"] = pd.to_datetime(
		evt["fecha_final"].astype("string") + " " + evt["hora_salida_estimada"].astype("string"),
		errors="coerce",
	)
	evt["fin_dt"] = evt["fin_dt"].fillna(evt["inicio_dt"])

	# Requisito: cruce espacial + temporal con LEFT JOIN por fecha y parada.
	df_temp = base.merge(
		evt[["date", "stop_id", "nombre_evento", tipo_col, "score", "inicio_dt", "fin_dt"]],
		how="left",
		on=["date", "stop_id"],
	)
	df_temp["event_count"] = df_temp.groupby(["match_key", "stop_id"])["nombre_evento"].transform("nunique")
	df_temp = df_temp.sort_values(by=["score"], ascending=False)
	df_temp = df_temp.drop_duplicates(subset=["date","match_key", "stop_id"], keep="first")

	ventana_entrada_inicio = df_temp["inicio_dt"] - pd.Timedelta(hours=1.5)
	ventana_salida_fin = df_temp["fin_dt"] + pd.Timedelta(hours=1.5)

	df_temp["durante_entrada"] = ((df_temp["merge_time"] >= ventana_entrada_inicio) & (df_temp["merge_time"] < df_temp["inicio_dt"])).astype(int)
	df_temp["durante_evento"] = ((df_temp["merge_time"] >= df_temp["inicio_dt"]) & (df_temp["merge_time"] <= df_temp["fin_dt"])).astype(int)
	df_temp["despues_evento"] = ((df_temp["merge_time"] > df_temp["fin_dt"]) & (df_temp["merge_time"] <= ventana_salida_fin)).astype(int)
	df_temp["evento_nocturno"] = (df_temp["fin_dt"].dt.hour >= 22).astype(int)

	out = df_temp.copy()
	out["hubo_evento_en_el_dia"] = out["nombre_evento"].notna().astype(int)
	out["n_eventos"] = out["event_count"].fillna(0).astype(int)
	out["tipo_evento_prioritario"] = out[tipo_col].fillna("Ninguno")
	out["hubo_evento_en_el_dia"] = out["hubo_evento_en_el_dia"].fillna(0).astype(int)
	out["n_eventos"] = out["n_eventos"].fillna(0).astype(int)
	out["durante_entrada"] = out["durante_entrada"].fillna(0).astype(int)
	out["durante_evento"] = out["durante_evento"].fillna(0).astype(int)
	out["despues_evento"] = out["despues_evento"].fillna(0).astype(int)
	out["evento_nocturno"] = out["evento_nocturno"].fillna(0).astype(int)
	out["tipo_evento_prioritario"] = out["tipo_evento_prioritario"].fillna("Ninguno")

	out = out.drop(
		columns=[
			"event_count",
			"inicio_dt",
			"fin_dt",
		],
		errors="ignore",
	)
	return out


def merge_gtfs_alerts(df_base: pd.DataFrame, df_alerts: pd.DataFrame) -> pd.DataFrame:
	"""
	LEFT JOIN temporal GTFS + Alertas por `route_id` usando `merge_asof` hacia atrás.
	Asocia a cada tren la alerta más reciente activa en su línea y calcula
	`seconds_since_last_alert` en el instante del evento GTFS.
	"""
	if df_alerts.empty:
		out = df_base.copy()
		out["category"] = pd.NA
		out["num_updates"] = 0
		out["timestamp_start"] = pd.NaT
		out["seconds_since_last_alert"] = 2592000
		return out

	left = df_base.copy()
	# Reducir alertas al mínimo imprescindible antes del cruce.
	right = df_alerts[
		["route_id", "timestamp_start", "category", "num_updates"]
	].copy()

	left["route_id"] = normalize_route_id(left["route_id"])
	right["route_id"] = normalize_route_id(right["route_id"])

	left["merge_time"] = pd.to_datetime(left["merge_time"], errors="coerce")
	right["timestamp_start"] = pd.to_datetime(right["timestamp_start"], errors="coerce")

	# Preservar todas las filas de la tabla izquierda; solo separamos válidas/inválidas.
	mask_valid = left["route_id"].notna() & left["merge_time"].notna()
	left_valid = left[mask_valid].copy()
	left_invalid = left[~mask_valid].copy()
	left_invalid["category"] = pd.NA
	left_invalid["num_updates"] = 0
	left_invalid["timestamp_start"] = pd.NaT
	left_invalid["seconds_since_last_alert"] = 2592000

	right = right.dropna(subset=["route_id", "timestamp_start"])

	# Reducir RAM en claves de join.
	left_valid["route_id"] = left_valid["route_id"].astype("category")
	right["route_id"] = right["route_id"].astype("category")

	chunks: list[pd.DataFrame] = []
	for route in left_valid["route_id"].unique():
		left_chunk = left_valid[left_valid["route_id"] == route].sort_values("merge_time").copy()
		right_chunk = right[right["route_id"] == route].sort_values("timestamp_start").copy()

		if right_chunk.empty:
			left_chunk["category"] = pd.NA
			left_chunk["num_updates"] = 0
			left_chunk["timestamp_start"] = pd.NaT
			left_chunk["seconds_since_last_alert"] = 2592000
			chunks.append(left_chunk)
			continue

		# Estamos en una sola route_id, por tanto no usamos `by`.
		right_chunk = right_chunk.drop(columns=["route_id"], errors="ignore")
		merged_chunk = pd.merge_asof(
			left_chunk,
			right_chunk,
			left_on="merge_time",
			right_on="timestamp_start",
			direction="backward",
			suffixes=("", "_alert"),
		)
		merged_chunk["num_updates"] = pd.to_numeric(merged_chunk["num_updates"], errors="coerce").fillna(0)
		merged_chunk["seconds_since_last_alert"] = (
			merged_chunk["merge_time"] - merged_chunk["timestamp_start"]
		).dt.total_seconds()
		merged_chunk["seconds_since_last_alert"] = pd.to_numeric(
			merged_chunk["seconds_since_last_alert"], errors="coerce"
		).fillna(2592000)
		chunks.append(merged_chunk)

	if not chunks:
		return left_invalid.sort_values("merge_time").reset_index(drop=True)

	merged = pd.concat(chunks + [left_invalid], ignore_index=True)
	merged = merged.sort_values("merge_time").reset_index(drop=True)
	return merged


def apply_final_column_policy(df: pd.DataFrame) -> pd.DataFrame:
	"""Aplica la política estricta de columnas: keep list + drop explícito."""
	gtfs_keep = [
		"match_key",
		"stop_id",
		"route_id",
		"delay_seconds",
		"lagged_delay_1",
		"lagged_delay_2",
		"route_rolling_delay",
		"actual_headway_seconds",
		"is_unscheduled",
		"hour_sin",
		"hour_cos",
		"dow",
		"is_weekend",
		"target_delay_10m",
		"target_delay_20m",
		"target_delay_30m",
		"target_delay_45m",
		"target_delay_60m",
		"delta_delay_10m",
		"delta_delay_20m",
		"delta_delay_30m",
		"delta_delay_45m",
		"delta_delay_60m",
		"scheduled_time_to_end"
	]

	weather_keep = ["temp_extreme"]

	events_keep = [
		"hubo_evento_en_el_dia",
		"n_eventos",
		"tipo_evento_prioritario",
		"durante_entrada",
		"durante_evento",
		"despues_evento",
		"evento_nocturno",
	]

	alerts_keep = ["category", "num_updates", "timestamp_start", "seconds_since_last_alert"]

	explicit_drop = [
		"trip_uid",
		"scheduled_time",
		"actual_time",
		"stops_to_end",
		"trip_progress",
		"scheduled_seconds",
		"actual_seconds",
		"delay_minutes",
		"Cloud Cover",
		"Precipitation",
		"precip_3h_accum",
		"Snow",
		"is_high_wind",
		"is_freezing",
		"apparent_temp",
		"Wind Speed",
		"Temperature",
		"paradas_afectadas",
		"parada_nombre",
		"parada_lineas",
		"lines",
		"score",
		"event_id",
		"timestamp_end",
		"text_snippet",
		"description",
		"merge_time",
	]

	wanted = gtfs_keep + weather_keep + events_keep + alerts_keep
	# Seleccionar columnas válidas evitando columnas explícitamente prohibidas.
	existing = [c for c in wanted if c in df.columns and c not in explicit_drop]
	return df.loc[:, existing].copy()


def build_final_dataset(start: str, end: str, output_base: str = "grupo5/final") -> None:
	"""
	Orquesta la generación del dataset final por bloques mensuales con continuidad temporal.
	- GTFS/Clima: mes estricto.
	- Eventos/Alertas: lookback de 3 días + mes.
	- Salida particionada por year/month en MinIO.
	"""
	access_key = os.getenv("MINIO_ACCESS_KEY")
	secret_key = os.getenv("MINIO_SECRET_KEY")
	if not access_key or not secret_key:
		raise ValueError("MINIO_ACCESS_KEY and MINIO_SECRET_KEY must be defined.")

	start_dt = datetime.strptime(start, "%Y-%m-%d").date()
	end_dt = datetime.strptime(end, "%Y-%m-%d").date()
	if end_dt < start_dt:
		raise ValueError("'end' must be greater than or equal to 'start'.")

	client = MinIODataClient(access_key=access_key, secret_key=secret_key)

	cursor = date(start_dt.year, start_dt.month, 1)
	while cursor <= end_dt:
		year = cursor.year
		month = cursor.month
		month_first = date(year, month, 1)
		month_last = date(year, month, monthrange(year, month)[1])

		month_start = max(month_first, start_dt)
		month_end = min(month_last, end_dt)
		lookback_start = month_start - timedelta(days=3)

		days_gtfs = list(iterate_dates(month_start, month_end))
		days_lookback = list(iterate_dates(lookback_start, month_end))

		output_object = (
			f"{output_base}/year={year}/month={month:02d}/dataset_final.parquet"
		)

		# Carga/preparación por bloque mensual con continuidad temporal.
		gtfs_raw = load_gtfs(client, days_gtfs)

		gtfs = prepare_gtfs(gtfs_raw)
		del gtfs_raw
		gc.collect()

		gtfs = reduce_mem_usage(gtfs)
		weather = prepare_weather(load_weather(client, days_gtfs))

		# Orden de cruces: GTFS+Clima -> GTFS+Eventos -> GTFS+Alertas.
		merged = merge_gtfs_weather(gtfs, weather)
		del gtfs, weather
		gc.collect()

		events = prepare_events(load_events(client, days_lookback))
		merged = merge_gtfs_events(merged, events)
		del events
		gc.collect()

		alerts = prepare_alerts(load_alerts(client, days_lookback))
		merged = merge_gtfs_alerts(merged, alerts)
		del alerts
		gc.collect()

		merged = reduce_mem_usage(merged)
		final_df = apply_final_column_policy(merged)
		del merged
		gc.collect()

		client.upload_df_parquet(output_object, final_df)

		print(
			"[generate_final_dataset] OK "
			f"month={year}-{month:02d} rows={len(final_df)} cols={len(final_df.columns)} "
			f"output={output_object}"
		)

		del final_df
		gc.collect()

		if month == 12:
			cursor = date(year + 1, 1, 1)
		else:
			cursor = date(year, month + 1, 1)


def parse_args() -> argparse.Namespace:
	"""Parsea argumentos CLI para ejecutar el pipeline por rango de fechas."""
	parser = argparse.ArgumentParser(description="Generate final training dataset from MinIO layers.")
	parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
	parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
	parser.add_argument(
		"--output-object",
		default=OUTPUT_OBJECT,
		help="MinIO object path for final parquet",
	)
	return parser.parse_args()


def main() -> int:
	"""Punto de entrada del script para ejecución por línea de comandos."""
	args = parse_args()
	output_base = args.output_object
	if output_base == OUTPUT_OBJECT:
		output_base = "grupo5/final"

	build_final_dataset(start=args.start, end=args.end, output_base=output_base)
	print("[generate_final_dataset] DONE monthly processing completed")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
