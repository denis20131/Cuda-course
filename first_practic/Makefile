# Makefile для компиляции и запуска Jacobi3D (OpenMP и CUDA версии)

# Компиляторы и флаги
CC = g++
CFLAGS = -O3 -fopenmp -std=c++11
NVCC = nvcc
CUDAFLAGS = -O3 -arch=sm_60 -std=c++11 -Wno-deprecated-gpu-targets

# Имена исполняемых файлов
EXEC_CPU = jacobi3d_cpu
EXEC_GPU = jacobi3d_gpu

# Стандартные параметры запуска
L ?= 900
ITMAX ?= 20


# Временные файлы для хранения результатов
CPU_TIME_FILE = cpu_time.txt
GPU_TIME_FILE = gpu_time.txt

.PHONY: all cpu gpu clean run run_cpu run_gpu

all: cpu gpu

# CPU версия (OpenMP)
cpu: $(EXEC_CPU)

$(EXEC_CPU): main.cpp common.hpp
	$(CC) $(CFLAGS) -o $@ $<

# GPU версия (CUDA)
gpu: $(EXEC_GPU)

$(EXEC_GPU): main.cu common.hpp
	$(NVCC) $(CUDAFLAGS) -o $@ $<

# Очистка
clean:
	rm -f $(EXEC_CPU) $(EXEC_GPU) $(CPU_TIME_FILE) $(GPU_TIME_FILE)

# Запуск CPU версии и сохранение времени
run_cpu: $(EXEC_CPU)
	@echo "Запуск CPU версии..."
	@if [ -x "$(EXEC_CPU)" ]; then \
		./$(EXEC_CPU) $(L) $(ITMAX)  > $(CPU_TIME_FILE); \
		echo "CPU версия завершена."; \
	else \
		echo "Ошибка: $(EXEC_CPU) не существует или не является исполняемым файлом."; \
		exit 1; \
	fi

# Запуск GPU версии и сохранение времени
run_gpu: $(EXEC_GPU)
	@echo "Запуск GPU версии..."
	@if [ -x "$(EXEC_GPU)" ]; then \
		./$(EXEC_GPU) $(L) $(ITMAX)  > $(GPU_TIME_FILE); \
		echo "GPU версия завершена."; \
	else \
		echo "Ошибка: $(EXEC_GPU) не существует или не является исполняемым файлом."; \
		exit 1; \
	fi

# Сравнение производительности
run: run_cpu run_gpu
	@echo ""
	@echo "Сравнение производительности:"
	@echo "----------------------------"
	@CPU_TIME=$$(grep 'Time in seconds' $(CPU_TIME_FILE) | awk '{print $$5}'); \
	GPU_TIME=$$(grep 'Time in seconds' $(GPU_TIME_FILE) | awk '{print $$5}'); \
	if [ -z "$$CPU_TIME" ] || [ -z "$$GPU_TIME" ]; then \
		echo "Ошибка: не удалось извлечь время выполнения."; \
		echo "Содержимое $(CPU_TIME_FILE):"; cat $(CPU_TIME_FILE); \
		echo "Содержимое $(GPU_TIME_FILE):"; cat $(GPU_TIME_FILE); \
	else \
		SPEEDUP=$$(echo "$$CPU_TIME / $$GPU_TIME" | bc -l); \
		printf "CPU время: %20.2f сек\n" $$CPU_TIME; \
		printf "GPU время: %20.2f сек\n" $$GPU_TIME; \
		printf "Ускорение (speedup): %16.2f раз\n" $$SPEEDUP; \
	fi
	@echo "----------------------------"
	@echo "Тестирование завершено"