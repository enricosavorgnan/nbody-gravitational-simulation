CC        ?= cc
STD       ?= -std=c11
CPPFLAGS  ?=
CFLAGS    ?= -O3 -Wall -Wextra -Wpedantic -flto -D_POSIX_C_SOURCE=199309L -DBLOCK_SIZE=128

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

generate_ic: generate_initial_conditions.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

%.o: %.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -c $< -o $@


run-smoke: all
	mkdir -p ./bins
	./generate_initial_conditions --model 0 --n 128 --seed 42 --output ./bins/plummer_128.bin
	./main --input ./bins/plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/plummer_128_final.bin --quiet
	./generate_initial_conditions --model 1 --n 128 --seed 43 --output ./bins/ball_128.bin
	./main --input ./bins/ball_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/ball_128_final.bin --quiet

test-serial : all
	mkdir -p ./bins
	mkdir -p ./bins/test-serial/
	mkdir -p ./profilings
	mkdir -p ./profilings/test-serial/
	./generate_initial_conditions --model 0 --n 50000 --seed 42 --output ./bins/test-serial.bin
	for i in 1 2 3 4; do \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/n.out --kernel "n" --profiler 1  --profiler-path ./profilings/test-serial/n.txt ; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/r.out --kernel "r" --profiler 1  --profiler-path ./profilings/test-serial/r.txt ; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/t.out --kernel "t" --profiler 1  --profiler-path ./profilings/test-serial/t.txt ; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/b.out --kernel "b" --profiler 1  --profiler-path ./profilings/test-serial/b.txt ; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/rt.out --kernel "rt" --profiler 1  --profiler-path ./profilings/test-serial/rt.txt ; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/bt.out --kernel "bt" --profiler 1  --profiler-path ./profilings/test-serial/bt.txt ; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/br.out --kernel "br" --profiler 1  --profiler-path ./profilings/test-serial/br.txt ; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/brt.out --kernel "brt" --profiler 1  --profiler-path ./profilings/test-serial/brt.txt ; \
	done

test-serial-more-blocks: all
	mkdir -p ./bins
	mkdir -p ./bins/test-serial/
	mkdir -p ./profilings
	mkdir -p ./profilings/test-serial/
	./generate_initial_conditions --model 0 --n 50000 --seed 42 --output ./bins/test-serial.bin
	for i in 1 2 3 4; do \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/n.out --kernel "n" --profiler 1  --profiler-path ./profilings/test-serial/n.txt; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/r.out --kernel "r" --profiler 1  --profiler-path ./profilings/test-serial/r.txt; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/t.out --kernel "t" --profiler 1  --profiler-path ./profilings/test-serial/t.txt; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/b.out --kernel "b" --profiler 1  --profiler-path ./profilings/test-serial/b.txt; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/rt.out --kernel "rt" --profiler 1  --profiler-path ./profilings/test-serial/rt.txt; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/bt.out --kernel "bt" --profiler 1  --profiler-path ./profilings/test-serial/bt.txt; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/br.out --kernel "br" --profiler 1  --profiler-path ./profilings/test-serial/br.txt; \
		./main --input ./bins/test-serial.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 3 --output ./bins/test-serial/brt.out --kernel "brt" --profiler 1  --profiler-path ./profilings/test-serial/brt.txt; \
	done

clean:
	rm -f $(PROGRAMS) *.o *.bin
