CC        ?= cc
STD       ?= -std=c11
CPPFLAGS  ?=
CFLAGS    ?= -O3 -Wall -Wextra -Wpedantic -flto -D_POSIX_C_SOURCE=199309L -DBLOCK_SIZE=128 -DUSE_PAPI=1

LDLIBS    ?= -lm -lpapi
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
	$(CC) $(STD) $(CFLAGS) -o $@ $^ $(LDLIBS)

generate_initial_conditions: generate_initial_conditions.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

%.o: %.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -c $< -o $@


test-aos: all
	mkdir -p ./bins/test-aos/
	mkdir -p ./profilings/test-aos/
	./generate_initial_conditions --model 0 --n 100000 --seed 42 --output ./bins/test-aos.bin
	for i in 1; do \
		./main --input ./bins/test-aos.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-aos/n.out --layout "aos" --kernel "n" --profiler 1  --profiler-path ./profilings/test-aos/n-aos.txt --papi 1; \
		./main --input ./bins/test-aos.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-aos/n.out --layout "soa" --kernel "n" --profiler 1  --profiler-path ./profilings/test-aos/n-soa.txt --papi 1; \
		./main --input ./bins/test-aos.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-aos/r.out --layout "aos" --kernel "t" --profiler 1  --profiler-path ./profilings/test-aos/t-aos.txt --papi 1; \
		./main --input ./bins/test-aos.bin --nsteps 25 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/test-aos/r.out --layout "soa" --kernel "t" --profiler 1  --profiler-path ./profilings/test-aos/t-soa.txt --papi 1; \
	done


clean:
	rm -f $(PROGRAMS) *.o *.bin

clean-folders:
	rm -rf ./bins/test-aos/
	rm -rf ./profilings/test-aos/
