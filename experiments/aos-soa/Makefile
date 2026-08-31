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


test-aos: all
	mkdir -p ./bins/test-aos/
	mkdir -p ./profilings/test-aos/
	./generate_initial_conditions --model 0 --n 1000000 --seed 42 --output ./bins/test-aos.bin
	for i in 1; do \
		./main --input ./bins/test-aos.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-aos/n.out --kernel "n" --profiler 1  --profiler-path ./profilings/test-aos/n.txt; \
		./main --input ./bins/test-aos.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-aos/r.out --kernel "t" --profiler 1  --profiler-path ./profilings/test-aos/t.txt; \
	done


clean:
	rm -f $(PROGRAMS) *.o *.bin

clean-folders:
	rm -rf ./bins/test-aos/
	rm -rf ./profilings/test-aos/
