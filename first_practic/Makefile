# Makefile для компиляции и запуска Jacobi3D (OpenMP и CUDA версии)

# Компиляторы и флаги
CC = g++
CFLAGS = -O3 -fopenmp -std=c++11
NVCC = nvcc
CUDAFLAGS = -O3 -arch=sm_60 -std=c++11 -Wno-deprecated-gpu-targets

# Имена исполняемых файлов
EXEC_CPU = jacobi3d_cpu
EXEC_GPU = jacobi3d_gpu

# Временные файлы для хранения результатов
CPU_TIME_FILE = cpu_time.txt
GPU_TIME_FILE = gpu_time.txt

.PHONY: all cpu gpu clean run run_cpu run_gpu

all: cpu gpu

# CPU версия (OpenMP)
cpu: $(EXEC_CPU)

$(EXEC_CPU): main.cpp 
	$(CC) $(CFLAGS) -o $@ $<

# GPU версия (CUDA)
gpu: $(EXEC_GPU)

$(EXEC_GPU): main.cu 
	$(NVCC) $(CUDAFLAGS) -o $@ $<

# Очистка
clean:
	rm -f $(EXEC_CPU) $(EXEC_GPU) $(CPU_TIME_FILE) $(GPU_TIME_FILE)

# Запуск CPU версии и сохранение времени
run_cpu: $(EXEC_CPU)
	@echo "Запуск CPU версии..."
	@if [ -x "$(EXEC_CPU)" ]; then \
		./$(EXEC_CPU)  > $(CPU_TIME_FILE); \
		echo "CPU версия завершена."; \
	else \
		echo "Ошибка: $(EXEC_CPU) не существует или не является исполняемым файлом."; \
		exit 1; \
	fi

# Запуск GPU версии и сохранение времени
run_gpu: $(EXEC_GPU)
	@echo "Запуск GPU версии..."
	@if [ -x "$(EXEC_GPU)" ]; then \
		./$(EXEC_GPU)  > $(GPU_TIME_FILE); \
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
	@echo "CPU результаты:"
	@cat $(CPU_TIME_FILE)
	@echo ""
	@echo "GPU результаты:"
	@cat $(GPU_TIME_FILE)
	@echo ""
	@CPU_TIME=$$(grep 'Time in seconds' $(CPU_TIME_FILE) | awk '{print $$5}'); \
	GPU_TIME=$$(grep 'Total computation time' $(GPU_TIME_FILE) | awk '{print $$4}'); \
	if [ -n "$$CPU_TIME" ] && [ -n "$$GPU_TIME" ]; then \
		RATIO=$$(echo "scale=4; $$CPU_TIME / $$GPU_TIME" | bc -l); \
		LC_NUMERIC=C printf "Ускорение на GPU: %.2f раз\n" $$RATIO; \
	else \
		echo "Не удалось извлечь время выполнения"; \
	fi