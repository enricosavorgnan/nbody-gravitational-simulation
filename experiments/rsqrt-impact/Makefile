CC        ?= cc
STD       ?= -std=c11
CPPFLAGS  ?=
CFLAGS    ?= -O3 -Wall -Wextra -Wpedantic -flto -D_POSIX_C_SOURCE=199309L -DBLOCK_SIZE=128 -DN_RSQRT_LOOPS=$(N_RSQRT_LOOP)

LDLIBS    ?= -lm
PRECISION ?= double

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

NBODY_OBJS := main.o integration.o particles.o profiling.o utils.o

.PHONY: all clean run-smoke

all: $(PROGRAMS)


main: $(NBODY_OBJS)
	$(CC) $(STD) $(CFLAGS) -o $@ $^ $(LDLIBS)

generate_initial_conditions: generate_initial_conditions.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

%.o: %.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -c $< -o $@

test-rsqrt: all
	mkdir -p ./bins/test-rsqrt/
	mkdir -p ./profilings/test-rsqrt/
	./generate_initial_conditions --model 0 --n 100 --seed 42 --output ./bins/test-rsqrt.bin
	for i in 1 2 3 4; do \
		$(MAKE) clean all N_RSQRT_LOOP=$$i; \
		for j in 1 2 3 4 5; do \
			./main --input ./bins/test-rsqrt.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-rsqrt/n_$$i.out --kernel n  --profiler 1 --profiler-path ./profilings/test-rsqrt/n_$$i.txt; \
			./main --input ./bins/test-rsqrt.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-rsqrt/r_$$i.out --kernel r  --profiler 1 --profiler-path ./profilings/test-rsqrt/r_$$i.txt; \
			./main --input ./bins/test-rsqrt.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-rsqrt/rt_$$i.out --kernel rt --profiler 1 --profiler-path ./profilings/test-rsqrt/rt_$$i.txt; \
		done \
	done

clean:
	rm -f $(PROGRAMS) *.o *.bin


clean-folders:
	rm -r ./bins/
	rm -r ./profilings/