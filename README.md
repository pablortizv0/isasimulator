# Bcas CAP API

API para calcular el CAP óptimo de un ISA según el importe financiado y la duración del curso.

## Endpoint principal

### POST /cap

Calcula el CAP óptimo en base a los inputs del simulador ISA.

**Body:**
```json
{
  "importe_financiado": 7000,
  "duracion_formacion": 8
}
```

**Response:**
```json
{
  "cap_opt": 8400.0,
  "tir": 0.181
}
```

## Requisitos

- Python 3.9+
- FastAPI
- numpy, pandas, scipy, numpy-financial

## Ejecución local

```bash
uvicorn main:app --reload
```