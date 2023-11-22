SSE_ARCH_FLAGS=-msse3
CC=g++
ARCH=-m64 $(SSE_ARCH_FLAGS)
CFLAGS= $(ARCH) -g -Wall -O3 -ftree-vectorize
LDFLAGS=$(ARCH)
LIBS=-lrt
BLASFLAG=-I /home/resiliente/cs410software/openblas_0_3_21_install/include -L /home/resiliente/cs410software/openblas_0_3_21_install/lib -lopenblas
INCLUDES= -I $(PWD)
SRC_DIR = $(PWD)
SRCS=$(SRC_DIR)/matvec.cpp $(SRC_DIR)/timeutil.cpp
OBJS=$(SRCS:.cpp=.o)
BIN_DIR=$(PWD)
MAIN = $(BIN_DIR)/matvec

matvec:
	$(CC) $(SRCS) $(INCLUDES) $(CFLAGS) -D _SSE3_ -o $(MAIN) $(LIBS)

SOURCES=matmul.cpp
EXECUTABLES1=loopijk loopikj loopjik loopjki loopkij loopkji
CFLAGS = -Wall

matmul_schedule: $(EXECUTABLES1)

loopijk: $(SOURCES)
	$(CC) $(CFLAGS)  $(BLASFLAG) -DLOOPijk -o $@ $^

loopikj: $(SOURCES)
	$(CC) $(CFLAGS) $(BLASFLAG) -DLOOPikj -o $@ $^
	
loopjik: $(SOURCES)
	$(CC) $(CFLAGS)  $(BLASFLAG) -DLOOPjik -o $@ $^

loopjki: $(SOURCES)
	$(CC) $(CFLAGS)  $(BLASFLAG) -DLOOPjki -o $@ $^

loopkij: $(SOURCES)
	$(CC) $(CFLAGS)  $(BLASFLAG) -DLOOPkij -o $@ $^

loopkji: $(SOURCES)
	$(CC) $(CFLAGS)  $(BLASFLAG) -DLOOPkji -o $@ $^


EXECUTABLES2 = matmul_o0 matmul_o1 matmul_o2 matmul_o3 matmul_o4

matmul_optlevel: $(EXECUTABLES2)

matmul_o0: $(SOURCES)
	$(CC) $(CFLAGS)  $(BLASFLAG) -USTACKALLOCATED -ULOOPINTERCHANGE -o $@ $^

matmul_o1: $(SOURCES)
	$(CC) $(CFLAGS)  $(BLASFLAG) -USTACKALLOCATED -ULOOPINTERCHANGE -O1 -o $@ $^

matmul_o2: $(SOURCES)
	$(CC) $(CFLAGS)  $(BLASFLAG) -USTACKALLOCATED -ULOOPINTERCHANGE -O2 -o $@ $^

matmul_o3: $(SOURCES)
	$(CC) $(CFLAGS)  $(BLASFLAG) -USTACKALLOCATED -ULOOPINTERCHANGE -O3 -o $@ $^

matmul_o4: $(SOURCES)
	$(CC) $(CFLAGS)  $(BLASFLAG) -USTACKALLOCATED -ULOOPINTERCHANGE -O4 -o $@ $^

EXECUTABLES3=sdot_matmul sgemm_matmul 

matmul_blas: $(SOURCES)
	$(CC) $(CFLAGS) $(BLASFLAG) -DSDOT -o $@ $^
	$(CC) $(CFLAGS) $(BLASFLAG) -DSGEMM -o $@ $^



.cpp.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

timeutil.cpp: timeutil.h

clean:
	$(RM) *.o matvec
	$(RM) *.o matmul_schedule
	$(RM) *.o matmul_blas
	$(RM) *.o matmul_optlevel
	$(RM) *.exe

.PHONY: clean all
