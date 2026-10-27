"""
Ablation study — treina múltiplas configurações do YOLOv8 e salva
os resultados em experimentos_resultados.csv para comparação.

Como usar:
  1. Edite a lista EXPERIMENTOS abaixo com os cenários que quiser testar.
  2. Execute: python experimentos.py
  3. Acompanhe o progresso no terminal.
  4. Ao final, abra experimentos_resultados.csv para comparar.

Métricas salvas por experimento:
  mAP50, mAP50-95, Precision, Recall, tempo de treino (min)
"""

import time
import csv
import traceback
from pathlib import Path
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
#  EXPERIMENTOS — edite aqui
#  Cada dicionário é um experimento independente.
#  Campos obrigatórios: model, imgsz, epochs, batch
#  name é gerado automaticamente se omitido.
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENTOS = [
    # ── Baseline ──────────────────────────────────────────────────────────────
    {"model": "yolov8n.pt", "imgsz": 512,  "epochs": 10, "batch": 4},

    # ── Variar tamanho de imagem ───────────────────────────────────────────────
    {"model": "yolov8n.pt", "imgsz": 320,  "epochs": 10, "batch": 4},
    {"model": "yolov8n.pt", "imgsz": 512,  "epochs": 10, "batch": 4},
    {"model": "yolov8n.pt", "imgsz": 640,  "epochs": 10, "batch": 4},

    # ── Variar epochs ──────────────────────────────────────────────────────────
    {"model": "yolov8n.pt", "imgsz": 512,  "epochs": 20, "batch": 4},
    {"model": "yolov8n.pt", "imgsz": 512,  "epochs": 35, "batch": 4},
    {"model": "yolov8n.pt", "imgsz": 512,  "epochs": 50, "batch": 4},

    # ── Variar versão do modelo ────────────────────────────────────────────────
    {"model": "yolov8s.pt", "imgsz": 512,  "epochs": 10, "batch": 4},
    {"model": "yolov8m.pt", "imgsz": 512,  "epochs": 10, "batch": 4},

    # ── Variar versão do batch ────────────────────────────────────────────────
    {"model": "yolov8n.pt", "imgsz": 512,  "epochs": 10, "batch": 8},
    {"model": "yolov8n.pt", "imgsz": 512,  "epochs": 10, "batch": 16},
    {"model": "yolov8n.pt", "imgsz": 512,  "epochs": 10, "batch": 32},
]

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURAÇÕES FIXAS
# ══════════════════════════════════════════════════════════════════════════════
DATA_YAML = "epi.yaml"
WORKERS   = 0          # 0 evita problemas no Windows
CSV_OUT   = Path(__file__).parent / "experimentos_resultados.csv"

CSV_FIELDS = [
    "nome", "model", "imgsz", "epochs", "batch",
    "mAP50", "mAP50-95", "precision", "recall", "f1_score",  # ← f1_score adicionado
    "tempo_min", "status", "pasta_runs"
]


# ══════════════════════════════════════════════════════════════════════════════
#  FUNÇÕES AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════
def nome_experimento(cfg: dict) -> str:
    modelo = Path(cfg["model"]).stem          # yolov8n
    return f"{modelo}_img{cfg['imgsz']}_ep{cfg['epochs']}_b{cfg['batch']}"


def extrair_metricas(metrics) -> dict:
    """Extrai as métricas principais do objeto retornado por model.val()."""
    try:
        p = float(metrics.box.mp)
        r = float(metrics.box.mr)
        f1 = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0  # ← F1 calculado
        return {
            "mAP50":     round(float(metrics.box.map50), 4),
            "mAP50-95":  round(float(metrics.box.map),   4),
            "precision": round(p, 4),
            "recall":    round(r, 4),
            "f1_score":  f1,                                           # ← f1_score adicionado
        }
    except Exception:
        return {"mAP50": "", "mAP50-95": "", "precision": "", "recall": "", "f1_score": ""}  # ← f1_score adicionado


def salvar_linha(row: dict):
    """Acrescenta uma linha no CSV (cria o arquivo com cabeçalho se necessário)."""
    novo = not CSV_OUT.exists()
    with open(CSV_OUT, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if novo:
            writer.writeheader()
        writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════════
#  LOOP PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════
def main():
    total = len(EXPERIMENTOS)
    print(f"\n{'='*60}")
    print(f"  INICIANDO {total} EXPERIMENTO(S)")
    print(f"{'='*60}\n")

    for i, cfg in enumerate(EXPERIMENTOS, 1):
        nome = cfg.get("name") or nome_experimento(cfg)
        print(f"\n[{i}/{total}] {nome}")
        print(f"  modelo={cfg['model']}  imgsz={cfg['imgsz']}  "
              f"epochs={cfg['epochs']}  batch={cfg['batch']}")
        print("-" * 50)

        row = {
            "nome":       nome,
            "model":      cfg["model"],
            "imgsz":      cfg["imgsz"],
            "epochs":     cfg["epochs"],
            "batch":      cfg["batch"],
            "mAP50":      "",
            "mAP50-95":   "",
            "precision":  "",
            "recall":     "",
            "f1_score":   "",  # ← f1_score adicionado
            "tempo_min":  "",
            "status":     "erro",
            "pasta_runs": "",
        }

        t0 = time.time()
        try:
            model = YOLO(cfg["model"])

            results = model.train(
                data    = DATA_YAML,
                epochs  = cfg["epochs"],
                imgsz   = cfg["imgsz"],
                batch   = cfg["batch"],
                workers = WORKERS,
                name    = nome,          # pasta em runs/detect/<nome>
                exist_ok= True,          # não cria subpasta numerada se já existe
            )

            # caminho onde o YOLOv8 salvou este treino
            run_dir = Path(model.trainer.save_dir)
            row["pasta_runs"] = str(run_dir)

            # avaliar no split de validação
            metrics = model.val()
            row.update(extrair_metricas(metrics))

            row["status"] = "ok"
            print(f"  ✓ mAP50={row['mAP50']}  mAP50-95={row['mAP50-95']}  "
                  f"P={row['precision']}  R={row['recall']}  F1={row['f1_score']}")  # ← f1_score no print

        except Exception as e:
            print(f"  ✗ ERRO: {e}")
            traceback.print_exc()

        finally:
            elapsed = round((time.time() - t0) / 60, 2)
            row["tempo_min"] = elapsed
            salvar_linha(row)
            print(f"  Tempo: {elapsed} min  |  Resultado salvo em {CSV_OUT.name}")

    # ── resumo final ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  CONCLUÍDO — resultados em: {CSV_OUT}")
    print(f"{'='*60}\n")

    # imprime tabela no terminal
    with open(CSV_OUT, encoding="utf-8") as f:
        linhas = f.readlines()
    print("".join(linhas))


if __name__ == "__main__":
    main()
