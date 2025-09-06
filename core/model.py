from dataclasses import dataclass, field
from scipy.optimize import root_scalar
import numpy as np
import numpy_financial as npf

EMPLEABILIDAD_POR_CAT = {
    'A+': [0.05, 0.12, 0.20, 0.28, 0.38, 0.50, 0.60, 0.70, 0.78, 0.84, 0.88, 0.90,
           0.91, 0.92, 0.93, 0.93, 0.94, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95, 0.95],
    'A': [0.04, 0.09, 0.16, 0.23, 0.30, 0.40, 0.50, 0.60, 0.68, 0.74, 0.77, 0.80,
          0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.93, 0.94, 0.95, 0.95, 0.95, 0.95],
    'B': [0.03, 0.07, 0.12, 0.17, 0.22, 0.30, 0.38, 0.46, 0.54, 0.60, 0.65, 0.70,
          0.72, 0.74, 0.76, 0.78, 0.80, 0.82, 0.82, 0.86, 0.87, 0.88, 0.91, 0.92],
    'C': [0.02, 0.06, 0.09, 0.13, 0.18, 0.25, 0.31, 0.36, 0.42, 0.48, 0.53, 0.60,
          0.63, 0.66, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.79, 0.79, 0.80, 0.80],
    'D': [0.00, 0.00, 0.0, 0.0, 0, 0,
          0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
          0.95, 0.95, 0.95, 0.95, 0.95, 0.95,
          0.95, 0.95, 0.95, 0.95, 0.95, 0.95]
}

@dataclass
class ModeloConfig:
    importe_financiado: float
    duracion_formacion: int
    categoria: str = 'D'
    comision: float = 0.2
    porcentaje_sueldo: float = 0.132
    salario_mensual: float = 1500
    tir_objetivo_anual: float = 0.18

    @property
    def empleabilidad(self):
        return EMPLEABILIDAD_POR_CAT.get(self.categoria, [0.0] * 24)

    @property
    def desembolso(self):
        return self.importe_financiado * (1 - self.comision)

def _cashflows(config: ModeloConfig, cap: float, ventana: int = 24):
    cf = np.zeros(config.duracion_formacion + ventana + 1)
    cf[0] -= config.desembolso
    cf[config.duracion_formacion + ventana] += cap
    return cf

def calcular_tir(cashflows: np.ndarray) -> float:
    tir_m = npf.irr(cashflows)
    return (1 + tir_m) ** 12 - 1 if tir_m is not None else float('-inf')

def calcular_cap_isa(importe_financiado: float, duracion_formacion: int, categoria: str = 'D') -> float:
    config = ModeloConfig(
        importe_financiado=importe_financiado,
        duracion_formacion=duracion_formacion,
        categoria=categoria
    )
    def f(cap):
        cf = _cashflows(config, cap)
        return calcular_tir(cf) - config.tir_objetivo_anual

    if f(importe_financiado) >= 0:
        return importe_financiado

    high = importe_financiado * 2
    while f(high) < 0:
        high *= 2

    sol = root_scalar(f, bracket=[importe_financiado, high], method='brentq')
    return sol.root