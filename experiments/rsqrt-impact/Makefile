CC        ?= cc
STD       ?= -std=c11
CPPFLAGS  ?= -I$(HOME)/opt/papi/include
LDFLAGS   ?= -L$(HOME)/opt/papi/lib -Wl,-rpath,$(HOME)/opt/papi/lib
CFLAGS    ?= -O3 -march=native -Wall -Wextra -Wpedantic -flto -ffast-math -D_POSIX_C_SOURCE=199309L -DBLOCK_SIZE=128 -DN_RSQRT_LOOP=$(N_RSQRT_LOOP)

LDLIBS    ?= -lm
PRECISION ?= double
USE_PAPI  ?= 0
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

test-rsqrt: generate_initial_conditions
	mkdir -p ./bins/test-rsqrt/
	mkdir -p ./profilings/test-rsqrt/
	./generate_initial_conditions --model 0 --n 100000 --seed 42 --output ./bins/test-rsqrt.bin
	for i in 1 2 3 4 5; do \
		$(MAKE) clean all N_RSQRT_LOOP=$$i; \
		for j in 1; do \
			./main --input ./bins/test-rsqrt.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/test-rsqrt/n_$$i.out --kernel n  --profiler 1 --papi 1 --profiler-path ./profilings/test-rsqrt/n_$$i.txt; \
			./main --input ./bins/test-rsqrt.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/test-rsqrt/r_$$i.out --kernel r  --profiler 1 --papi 1 --profiler-path ./profilings/test-rsqrt/r_$$i.txt; \
			./main --input ./bins/test-rsqrt.bin --nsteps 10 --dt 1e-4 --eps 0.05 --energy-every 10 --output ./bins/test-rsqrt/rt_$$i.out --kernel rt --profiler 1 --papi 1 --profiler-path ./profilings/test-rsqrt/rt_$$i.txt; \
		done \
	done

clean:
	rm -f $(PROGRAMS) *.o *.bin


clean-folders:
	rm -r ./bins/
	rm -r ./profilings/
