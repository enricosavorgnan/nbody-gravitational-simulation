CC        ?= cc
STD       ?= -std=c11
UNROLL	  ?= 0
CPPFLAGS  ?=
CFLAGS    ?= -O3 -Wall -Wextra -Wpedantic -flto -D_POSIX_C_SOURCE=199309L -DBLOCK_SIZE=128 -DDO_UNROLL=$(UNROLL)

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

# all, but with setting DO_UNROLL NUMBER:
all-n-unroll: UNROLL=8
all-auto-unroll: UNROLL=-1
all-no-unroll: UNROLL=0

all-n-unroll all-auto-unroll all-no-unroll: clean
	$(MAKE) all UNROLL=$(UNROLL)


main: $(NBODY_OBJS)
	$(CC) $(STD) $(CFLAGS) -o $@ $^ $(LDLIBS)

generate_ic: generate_initial_conditions.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

%.o: %.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -c $< -o $@


test-no-unroll: all-no-unroll
	mkdir -p ./bins
	mkdir -p ./bins/test-no-unroll/
	mkdir -p ./profilings
	mkdir -p ./profilings/test-no-unroll/
	./generate_initial_conditions --model 0 --n 100000 --seed 42 --output ./bins/test-no-unroll.bin
	for i in 1 2 3 4 5; do \
		./main --input ./bins/test-no-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-no-unroll/n.out --kernel "n" --profiler 1  --profiler-path ./profilings/test-no-unroll/n.txt ; \
		./main --input ./bins/test-no-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-no-unroll/n.out --kernel "t" --profiler 1  --profiler-path ./profilings/test-no-unroll/t.txt ; \
		./main --input ./bins/test-no-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-no-unroll/n.out --kernel "r" --profiler 1  --profiler-path ./profilings/test-no-unroll/r.txt ; \
		./main --input ./bins/test-no-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-no-unroll/n.out --kernel "rt" --profiler 1  --profiler-path ./profilings/test-no-unroll/rt.txt ; \
	done

test-auto-unroll: all-auto-unroll
	mkdir -p ./bins
	mkdir -p ./bins/test-auto-unroll/
	mkdir -p ./profilings
	mkdir -p ./profilings/test-auto-unroll/
	./generate_initial_conditions --model 0 --n 100000 --seed 42 --output ./bins/test-auto-unroll.bin
	for i in 1 2 3 4 5; do \
		./main --input ./bins/test-auto-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-auto-unroll/n.out --kernel "n" --profiler 1  --profiler-path ./profilings/test-auto-unroll/n.txt ; \
		./main --input ./bins/test-auto-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-auto-unroll/n.out --kernel "t" --profiler 1  --profiler-path ./profilings/test-auto-unroll/t.txt ; \
		./main --input ./bins/test-auto-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-auto-unroll/n.out --kernel "r" --profiler 1  --profiler-path ./profilings/test-auto-unroll/r.txt ; \
		./main --input ./bins/test-auto-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-auto-unroll/n.out --kernel "rt" --profiler 1  --profiler-path ./profilings/test-auto-unroll/rt.txt ; \
	done


test-n-unroll: all-n-unroll
	mkdir -p ./bins
	mkdir -p ./bins/test-n-unroll/
	mkdir -p ./profilings
	mkdir -p ./profilings/test-n-unroll/
	./generate_initial_conditions --model 0 --n 100000 --seed 42 --output ./bins/test-n-unroll.bin
	for i in 1 2 3 4 5; do \
		./main --input ./bins/test-n-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-n-unroll/n.out --kernel "n" --profiler 1  --profiler-path ./profilings/test-n-unroll/n.txt ; \
		./main --input ./bins/test-n-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-n-unroll/n.out --kernel "t" --profiler 1  --profiler-path ./profilings/test-n-unroll/t.txt ; \
		./main --input ./bins/test-n-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-n-unroll/n.out --kernel "r" --profiler 1  --profiler-path ./profilings/test-n-unroll/r.txt ; \
		./main --input ./bins/test-n-unroll.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-n-unroll/n.out --kernel "rt" --profiler 1  --profiler-path ./profilings/test-n-unroll/rt.txt ; \
	done


clean:
	rm -f $(PROGRAMS) *.o *.bin

clean-folders:
	rm -rf ./bins/test-auto-unroll/
	rm -rf ./bins/test-n-unroll/
	rm -rf ./bins/test-no-unroll/
	rm -rf ./profilings/test-auto-unroll/
	rm -rf ./profilings/test-n-unroll/
	rm -rf ./profilings/test-no-unroll/


