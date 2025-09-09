
from dataclasses import dataclass, field
from typing import List, Tuple, Literal, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, conlist
import numpy as np
import numpy_financial as npf
from scipy.optimize import minimize_scalar, root_scalar

# ---------- Empleabilidad por categoría (24 meses) ----------
EMPLEABILIDAD_POR_CAT = {
    'A+': [0.05,0.12,0.20,0.28,0.38,0.50,0.60,0.70,0.78,0.84,0.88,0.90,0.91,0.92,0.93,0.93,0.94,0.95,0.95,0.95,0.95,0.95,0.95,0.95],
    'A' : [0.04,0.09,0.16,0.23,0.30,0.40,0.50,0.60,0.68,0.74,0.77,0.80,0.82,0.84,0.86,0.88,0.90,0.92,0.93,0.94,0.95,0.95,0.95,0.95],
    'B' : [0.03,0.07,0.12,0.17,0.22,0.30,0.38,0.46,0.54,0.60,0.65,0.70,0.72,0.74,0.76,0.78,0.80,0.82,0.84,0.86,0.87,0.87,0.89,0.90],
    'C' : [0.00,0.00,0.0,0.0,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95],
    'D' : [0.00,0.00,0.0,0.0,0,0,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95,0.95],
}

# ---------- Config y helpers ----------
@dataclass
class ModeloConfig:
    num_estudiantes: int = 100
    importe_financiado: float = 18000.0
    comision: float = 0.20
    duracion_formacion: int = 12
    porcentaje_sueldo: float = 0.132
    salario_mensual: float = 2500.0
    incremento_salarial_anual: float = 0.10
    tir_objetivo_anual: float = 0.18
    pagos_porcentajes: Tuple[float, float, float, float, float] = (0.8, 0.0, 0.0, 0.0, 0.0)
    pagos_meses: Tuple[int, int, int, int, int] = (1, 2, 3, 4, 5)
    categoria: str = 'D'

    @property
    def empleabilidad_mensual(self) -> List[float]:
        return EMPLEABILIDAD_POR_CAT.get(self.categoria, [0.0] * 24)

    @property
    def desembolso_total(self) -> float:
        return self.importe_financiado * (1 - self.comision)

    @property
    def porcentaje_restante(self) -> float:
        return 1 - sum(self.pagos_porcentajes)

def round_to_10(n: float) -> float:
    return round(n / 10.0) * 10.0

def construir_perfil(config: ModeloConfig, cap: float) -> np.ndarray:
    pagos = []
    restante = cap
    mes = 0
    while restante > 0:
        año = mes // 12
        sueldo = config.salario_mensual * (1 + config.incremento_salarial_anual) ** año
        pago = min(config.porcentaje_sueldo * sueldo, restante)
        pagos.append(pago)
        restante -= pago
        mes += 1
    return np.array(pagos)

def construir_flujos_optimizado(config: ModeloConfig, cap: float):
    N = config.num_estudiantes
    desembolso = config.desembolso_total

    prop = np.array(config.empleabilidad_mensual)
    prop = np.maximum.accumulate(prop)
    emp_tot = prop * N
    nuevos = np.round(np.diff(np.concatenate(([0], emp_tot)))).astype(int)
    nuevos = np.clip(nuevos, 0, None)

    perfil = construir_perfil(config, cap)
    offset = config.duracion_formacion + 1

    total_len = offset + len(nuevos) + len(perfil)
    meses = np.arange(total_len + 1)

    outflows_fijos = np.zeros_like(meses, dtype=float)
    outflows_empleo = np.zeros_like(meses, dtype=float)
    inflows = np.zeros_like(meses, dtype=float)

    for pct, m in zip(config.pagos_porcentajes, config.pagos_meses):
        m = int(m)
        if m <= total_len:
            outflows_fijos[m] = - N * (desembolso * pct)

    pr = config.porcentaje_restante
    for i, n in enumerate(nuevos):
        mes_emp = offset + i
        if mes_emp <= total_len:
            outflows_empleo[mes_emp] -= n * (desembolso * pr)

    conv = np.convolve(nuevos, perfil)
    for j, val in enumerate(conv):
        idx = offset + j
        if idx <= total_len:
            inflows[idx] += val

    neto = outflows_fijos + outflows_empleo + inflows
    return {'meses': meses, 'neto': neto}

def calcular_tir(cashflows: np.ndarray) -> float:
    irr_m = npf.irr(cashflows)
    if irr_m is None or np.isnan(irr_m):
        return float('-inf')
    return (1 + irr_m) ** 12 - 1

def objetivo(cap: float, config: ModeloConfig) -> float:
    comp = construir_flujos_optimizado(config, cap)
    return (calcular_tir(comp['neto']) - config.tir_objetivo_anual) ** 2

def obtener_resultados(config: ModeloConfig, cap_max: float = 50000):
    res = minimize_scalar(lambda c: objetivo(c, config),
                          bounds=(config.importe_financiado, cap_max),
                          method='bounded')
    cap_opt = float(res.x)
    comp = construir_flujos_optimizado(config, cap_opt)
    tir = calcular_tir(comp['neto'])
    return cap_opt, tir

def _cashflows_anticipado(config: ModeloConfig, cap: float, ventana: int):
    dt = config.duracion_formacion
    desembolso = config.desembolso_total
    pr = config.porcentaje_restante
    t_finish = dt + ventana
    cf = np.zeros(t_finish + 1)
    for pct, m in zip(config.pagos_porcentajes, config.pagos_meses):
        m = int(m)
        if m <= t_finish:
            cf[m] -= desembolso * pct
    cf[t_finish] -= desembolso * pr
    cf[t_finish] += cap
    return cf

def calcular_cap_ventana(config: ModeloConfig, ventana: int, min_cap: Optional[float] = None, tir_target: Optional[float] = None):
    tir_target = tir_target or config.tir_objetivo_anual
    low = min_cap if min_cap is not None else config.importe_financiado

    def f(cap):
        cf = _cashflows_anticipado(config, cap, ventana)
        return calcular_tir(cf) - tir_target

    if f(low) >= 0:
        return low

    high = low * 2
    while f(high) < 0:
        high *= 2
    sol = root_scalar(f, bracket=[low, high], method='brentq')
    return float(sol.root)

# ---------- FastAPI I/O ----------
class Payload(BaseModel):
    categoria: Literal["A+","A","B","C","D"]
    importe_financiado: float = Field(..., ge=0)
    duracion_formacion: int = Field(..., ge=0, le=24)
    comision: float = Field(..., ge=0.0, le=1.0)
    porcentaje_sueldo: float = Field(..., ge=0.0, le=1.0)
    salario_mensual: float = Field(..., ge=0.0)
    pagos_porcentajes: conlist(float, min_length=5, max_length=5)
    pagos_meses: conlist(int, min_length=5, max_length=5)

class ISAResult(BaseModel):
    cap_final: float
    ventana1: float
    ventana2: float
    ventana3: float

class Proposal(BaseModel):
    isa: ISAResult

app = FastAPI(title="Bcas Calc API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True}

def build_config(p: Payload) -> ModeloConfig:
    return ModeloConfig(
        categoria=p.categoria,
        importe_financiado=p.importe_financiado,
        comision=p.comision,
        duracion_formacion=p.duracion_formacion,
        porcentaje_sueldo=p.porcentaje_sueldo,
        salario_mensual=p.salario_mensual,
        pagos_porcentajes=tuple(p.pagos_porcentajes),
        pagos_meses=tuple(p.pagos_meses)
    )

@app.post("/propuesta", response_model=Proposal)
def propuesta(p: Payload):
    cfg = build_config(p)

    cap_opt, _tir = obtener_resultados(cfg)
    cap_final = round_to_10(cap_opt)

    cap1 = calcular_cap_ventana(cfg, ventana=6)
    cap2 = calcular_cap_ventana(cfg, ventana=12, min_cap=cap1 * 1.05)
    cap3 = calcular_cap_ventana(cfg, ventana=18, min_cap=cap2 * 1.05)

    res = ISAResult(
        cap_final=round_to_10(cap_final),
        ventana1=round_to_10(cap1),
        ventana2=round_to_10(cap2),
        ventana3=round_to_10(cap3),
    )
    return Proposal(isa=res)
