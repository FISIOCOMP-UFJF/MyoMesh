#!/usr/bin/env bash

LOG_DIR="logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Rodando pacientes 1..7 (um comando por paciente)"
echo "Logs em: $LOG_DIR"
echo "=============================================="

for i in 1 2 3 5 6 7; do
  IN="../../../media/filipe/FILIPE/Patient_${i}.mat"
  OUT="$LOG_DIR/Patient_${i}.out.log"
  ERR="$LOG_DIR/Patient_${i}.err.log"

  echo "----------------------------------------------"
  echo "Paciente $i -> $IN"
  echo "stdout: $OUT"
  echo "stderr: $ERR"
  echo "----------------------------------------------"

  # Roda e salva logs. Se der erro, registra o exit code e segue.
  python3 execAll.py -i "$IN" >"$OUT" 2>"$ERR"
  CODE=$?

  if [ $CODE -ne 0 ]; then
    echo "ERRO no paciente $i (exit code $CODE)" | tee -a "$LOG_DIR/summary.log"
    echo "Comando: python3 execAll.py -i $IN" | tee -a "$LOG_DIR/summary.log"
    echo "stderr: $ERR" | tee -a "$LOG_DIR/summary.log"
    echo "" | tee -a "$LOG_DIR/summary.log"
  else
    echo "OK paciente $i" | tee -a "$LOG_DIR/summary.log"
  fi
done

echo "=============================================="
echo "Finalizado. Resumo: $LOG_DIR/summary.log"
echo "=============================================="