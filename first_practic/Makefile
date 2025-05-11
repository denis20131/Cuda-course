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
	@[ -f $(CPU_TIME_FILE) ] && cat $(CPU_TIME_FILE) || echo "CPU результаты недоступны"
	@echo ""
	@echo "GPU результаты:"
	@[ -f $(GPU_TIME_FILE) ] && cat $(GPU_TIME_FILE) || echo "GPU результаты недоступны"
	@echo ""
	@if [ -f $(CPU_TIME_FILE) ] && [ -f $(GPU_TIME_FILE) ]; then \
		CPU_TIME=$$(grep 'Time in seconds' $(CPU_TIME_FILE) | grep -oE '[0-9]+\.[0-9]+'); \
		GPU_TIME=$$(grep 'Elapsed Time' $(GPU_TIME_FILE) | grep -oE '[0-9]+\.[0-9]+'); \
		if [ -n "$$CPU_TIME" ] && [ -n "$$GPU_TIME" ] && [ "$$GPU_TIME" != "0" ]; then \
			RATIO=$$(LC_ALL=C echo "scale=2; $$CPU_TIME / $$GPU_TIME" | bc -l); \
			LC_ALL=C printf "Ускорение на GPU: в %.2f раз\n" $$RATIO; \
			LC_ALL=C printf "CPU время: %.2f сек | GPU время: %.2f сек\n" $$CPU_TIME $$GPU_TIME; \
		else \
			echo "Не удалось извлечь корректное время выполнения"; \
			[ -z "$$CPU_TIME" ] && echo "Не найдено CPU время"; \
			[ -z "$$GPU_TIME" ] && echo "Не найдено GPU время"; \
		fi \
	else \
		echo "Файлы результатов не найдены"; \
		[ ! -f $(CPU_TIME_FILE) ] && echo "Отсутствует $(CPU_TIME_FILE)"; \
		[ ! -f $(GPU_TIME_FILE) ] && echo "Отсутствует $(GPU_TIME_FILE)"; \
	fi
