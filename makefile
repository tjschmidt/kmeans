CC := g++
CFLAGS := -O0 -Wall -g -c
LFLAGS := -O0 -Wall -g -o
COMPILE = $(CC) $(CFLAGS) $^
LINK = $(CC) $(LFLAGS) $@ $^

all: driver

driver: driver.o KMeans.o
	$(LINK)

driver.o: driver.cpp
	$(COMPILE)
	
KMeans.o: KMeans.cpp
	$(COMPILE)
	
clean:
	rm *.exe *.o *.gch