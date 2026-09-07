CC        ?= cc
STD       ?= -std=c11
CPPFLAGS  ?= -I$(HOME)/opt/papi/include
LDFLAGS   ?= -L$(HOME)/opt/papi/lib -Wl,-rpath,$(HOME)/opt/papi/lib
CFLAGS    ?= -O3 -Wall -Wextra -Wpedantic -flto -D_POSIX_C_SOURCE=199309L -DBLOCK_SIZE=128

LDLIBS    ?= -lm
PRECISION ?= double
USE_PAPI  ?= 0

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
	mkdir -p ./bins/test-mau/
	mkdir -p ./profilings
	mkdir -p ./profilings/test-mau/
	./generate_initial_conditions --model 0 --n 100 --seed 42 --output ./bins/test-mau.bin
	for i in 1 2 3 4 5; do \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/m2.out --kernel "m2" --profiler 1  --profiler-path ./profilings/test-mau/m2.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/m4.out --kernel "m4" --profiler 1  --profiler-path ./profilings/test-mau/m4.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/m8.out --kernel "m8" --profiler 1  --profiler-path ./profilings/test-mau/m8.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/m16.out --kernel "m16" --profiler 1  --profiler-path ./profilings/test-mau/m16.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/rm2.out --kernel "rm2" --profiler 1  --profiler-path ./profilings/test-mau/rm2.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/rm4.out --kernel "rm4" --profiler 1  --profiler-path ./profilings/test-mau/rm4.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/rm8.out --kernel "rm8" --profiler 1  --profiler-path ./profilings/test-mau/rm8.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/rm16.out --kernel "rm16" --profiler 1  --profiler-path ./profilings/test-mau/rm16.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/brm2.out --kernel "brm2" --profiler 1  --profiler-path ./profilings/test-mau/brm2.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/brm4.out --kernel "brm4" --profiler 1  --profiler-path ./profilings/test-mau/brm4.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/brm8.out --kernel "brm8" --profiler 1  --profiler-path ./profilings/test-mau/brm8.txt; \
		./main --input ./bins/test-mau.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-mau/brm16.out --kernel "brm16" --profiler 1  --profiler-path ./profilings/test-mau/brm16.txt; \
	done

clean:
	rm -f $(PROGRAMS) *.o *.bin


clean-folders:
	rm -rf ./bins/test-mau/
	rm -rf ./profilings/test-mau/