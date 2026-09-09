CC        ?= cc
STD       ?= -std=c11
CPPFLAGS  ?= -I$(HOME)/opt/papi/include
LDFLAGS   ?= -L$(HOME)/opt/papi/lib -Wl,-rpath,$(HOME)/opt/papi/lib
CFLAGS    ?= -O3 -march=native -Wall -Wextra -Wpedantic -flto -D_POSIX_C_SOURCE=199309L -DBLOCK_SIZE=128

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


test-kernel : all
	mkdir -p ./bins
	mkdir -p ./profilings
	./generate_initial_conditions --model 0 --n 100000 --seed 42 --output ./bins/test-kernel.bin
	for i in 1; do \
		./main --input ./bins/test-kernel.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/n.out --kernel "n" --profiler 1 --papi 1 --profiler-path ./profilings/n.txt ; \
		./main --input ./bins/test-kernel.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/r.out --kernel "r" --profiler 1 --papi 1 --profiler-path ./profilings/r.txt ; \
		./main --input ./bins/test-kernel.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/t.out --kernel "t" --profiler 1 --papi 1 --profiler-path ./profilings/t.txt ; \
		./main --input ./bins/test-kernel.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/b.out --kernel "b" --profiler 1 --papi 1 --profiler-path ./profilings/b.txt ; \
		./main --input ./bins/test-kernel.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/rt.out --kernel "rt" --profiler 1 --papi 1 --profiler-path ./profilings/rt.txt ; \
		./main --input ./bins/test-kernel.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/bt.out --kernel "bt" --profiler 1 --papi 1 --profiler-path ./profilings/bt.txt ; \
		./main --input ./bins/test-kernel.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/br.out --kernel "br" --profiler 1 --papi 1 --profiler-path ./profilings/br.txt ; \
		./main --input ./bins/test-kernel.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/brt.out --kernel "brt" --profiler 1 --papi 1 --profiler-path ./profilings/brt.txt ; \
	done


clean:
	rm -f $(PROGRAMS) *.o *.bin

clean-folders:
	rm -rf ./bins
	rm -rf ./profilings


