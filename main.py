#z czata
# main.py

import datetime
import csv
import psutil
import subprocess



# # Ścieżki wejścia/wyjścia dla procesu inferencji
# INPUT_UNET = 'input_unet'
# OUTPUT_PATH = 'output_contours'

def get_cpu_mem():
    """Zwraca (cpu_percent, memory_percent)."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    return cpu, mem

def get_gpu_stats():
    """
    Próbuje odczytać z nvidia-smi: (utilization [%], memory_used [MiB]).
    Jeśli brak nvidia-smi lub błąd – zwraca (None, None).
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            encoding='utf-8'
        )
        # przykładowa linia: "12, 432"
        line = out.strip().splitlines()[0]
        util, mem = [float(x.strip()) for x in line.split(",")]
        return util, mem
    except Exception:
        return None, None

def run_stage(name, func, *args, **kwargs):
    """
    Uruchamia func(*args, **kwargs), loguje czasy i zasoby.
    Zwraca słownik z metrykami.
    """
    metrics = {'stage': name}

    # snapshot startowy
    t0 = datetime.datetime.now()
    cpu0, mem0 = get_cpu_mem()
    gpu_u0, gpu_m0 = get_gpu_stats()

    # właściwy krok
    func(*args, **kwargs)

    # snapshot końcowy
    t1 = datetime.datetime.now()
    cpu1, mem1 = get_cpu_mem()
    gpu_u1, gpu_m1 = get_gpu_stats()

    # wypełnij metryki
    metrics.update({
        'start_time': t0.isoformat(),
        'end_time':   t1.isoformat(),
        'duration_s': (t1 - t0).total_seconds(),
        'cpu_start_%': cpu0,
        'cpu_end_%':   cpu1,
        'mem_start_%': mem0,
        'mem_end_%':   mem1,
        'gpu_util_start_%': gpu_u0,
        'gpu_util_end_%':   gpu_u1,
        'gpu_mem_start_MiB': gpu_m0,
        'gpu_mem_end_MiB':   gpu_m1,
    })
    return metrics

def main():
    logs = []

    # 1) Czyszczenie
    from clear import clear
    logs.append(run_stage(
        'clear',
        clear,
        # flagi clear: usuwamy .keras i output_contours
        # dry_run=False, żeby faktycznie poszło usuwanie
        dict(keras=True, output=True, dry_run=True)
    ) if False else run_stage('clear', clear, keras=True, output=True, dry_run=True)) #dziwny zapis. czat tłumaczy się że to dla wygody
    del clear

    # 2) Trening

    from train import train

    logs.append(run_stage('train', train))

    del train

    # 3) Inferencja / proces
    from process import process
    logs.append(run_stage(
        'process',
        process#,
        # INPUT_UNET,
        # OUTPUT_PATH
    ))

    # Zapis do CSV
    fieldnames = [
        'stage',
        'start_time', 'end_time', 'duration_s',
        'cpu_start_%', 'cpu_end_%',
        'mem_start_%', 'mem_end_%',
        'gpu_util_start_%', 'gpu_util_end_%',
        'gpu_mem_start_MiB', 'gpu_mem_end_MiB',
    ]
    with open('resource_log.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in logs:
            writer.writerow(row)

    print("Zakończono. Metryki zapisane w resource_log.csv")

if __name__ == '__main__':
    main()
