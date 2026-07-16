CC        ?= cc
STD       ?= -std=c11
CPPFLAGS  ?=
CFLAGS    ?= -O2 -Wall -Wextra -Wpedantic
LDLIBS    ?= -lm
PRECISION ?= double

PROGRAMS := nbody_direct generate_initial_conditions
HEADERS  := ./headers/nbody_common.h, ./headers/nbody_time.h

ifeq ($(PRECISION),float)
PRECISION_CPPFLAGS := -DNBODY_USE_FLOAT
else ifeq ($(PRECISION),double)
PRECISION_CPPFLAGS := -DNBODY_USE_DOUBLE
else
$(error PRECISION must be either double or float)
endif

.PHONY: all clean run-smoke

all: $(PROGRAMS)

nbody_direct_serial: nbody_direct.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)

generate_ic: generate_initial_conditions.c $(HEADERS)
	$(CC) $(STD) $(CPPFLAGS) $(PRECISION_CPPFLAGS) $(CFLAGS) -o $@ $< $(LDLIBS)


run-smoke: all
	./generate_initial_conditions --model 0 --n 128 --seed 42 --output ./bins/plummer_128.bin
	./nbody_direct --input ./bins/plummer_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/plummer_128_final.bin --quiet
	./generate_initial_conditions --model 1 --n 128 --seed 43 --output ./bins/ball_128.bin
	./nbody_direct --input ./bins/ball_128.bin --nsteps 5 --dt 1e-4 --eps 0.05 --energy-every 1 --output ./bins/ball_128_final.bin --quiet

clean:
	rm -f $(PROGRAMS) *.o *.bin
