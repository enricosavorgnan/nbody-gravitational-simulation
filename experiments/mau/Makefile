CC        ?= cc
STD       ?= -std=c11
CPPFLAGS  ?= -I$(HOME)/opt/papi/include
LDFLAGS   ?= -L$(HOME)/opt/papi/lib -Wl,-rpath,$(HOME)/opt/papi/lib
CFLAGS    ?= -O3 -march=native -Wall -Wextra -Wpedantic -flto -ffast-math -D_POSIX_C_SOURCE=199309L -DBLOCK_SIZE=128
LDLIBS    ?= -lm

PRECISION ?= double
USE_PAPI  ?= 1
N_RSQRT_LOOP ?= 3

PROGRAMS := main generate_initial_conditions
HEADERS  := 	./headers/common.h \
				./headers/integration.h \
				./headers/particles.h \
				./headers/profiling.h \
			 	./headers/utils.h

ifeq ($(PRECISION),float)
PRECISION_CPPFLAGS := -DNBODY_USE_FLOAT
else ifeq ($(PRECISION),double)
PRECISION_CPPFLAGS := -DNBODY_USE_DOUBLE
else
$(error PRECISION must be either double or float)
endif

ifeq ($(USE_PAPI), 1)
CFLAGS += -DUSE_PAPI
LDLIBS += -lpapi
endif

NBODY_OBJS := main.o integration.o particles.o profiling.o utils.o

.PHONY: all clean run-smoke

all: $(PROGRAMS)


main: $(NBODY_OBJS)
	$(CC) $(STD) $(CFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

generate_initial_conditions: generate_initial_conditions.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) $(LDFLAGS) -o $@ $< $(LDLIBS)

generate_ic: generate_initial_conditions

%.o: %.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -c $< -o $@

test-mau: all
	mkdir -p ./bins
	mkdir -p ./profilings
	./generate_initial_conditions --model 0 --n 100 --seed 42 --output ./bins/test-mau.bin
	for i in 1; do \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/m2.out --kernel "m2" --profiler 1 --papi 1 --profiler-path ./profilings/m2.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/m4.out --kernel "m4" --profiler 1 --papi 1 --profiler-path ./profilings/m4.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/m8.out --kernel "m8" --profiler 1 --papi 1 --profiler-path ./profilings/m8.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/m16.out --kernel "m16" --profiler 1 --papi 1 --profiler-path ./profilings/m16.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/rm2.out --kernel "rm2" --profiler 1 --papi 1 --profiler-path ./profilings/rm2.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/rm4.out --kernel "rm4" --profiler 1 --papi 1 --profiler-path ./profilings/rm4.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/rm8.out --kernel "rm8" --profiler 1 --papi 1 --profiler-path ./profilings/rm8.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/rm16.out --kernel "rm16" --profiler 1 --papi 1 --profiler-path ./profilings/rm16.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/brm2.out --kernel "brm2" --profiler 1 --papi 1 --profiler-path ./profilings/brm2.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/brm4.out --kernel "brm4" --profiler 1 --papi 1 --profiler-path ./profilings/brm4.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/brm8.out --kernel "brm8" --profiler 1 --papi 1 --profiler-path ./profilings/brm8.txt; \
		./main --input ./bins/test-mau.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/brm16.out --kernel "brm16" --profiler 1 --papi 1 --profiler-path ./profilings/brm16.txt; \
	done

clean:
	rm -f $(PROGRAMS) *.o *.bin


clean-folders:
	rm -rf ./bins
	rm -rf ./profilings