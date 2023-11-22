PROB ?= 2
N ?= 2
EXECUTABLE=bin/main

all: team main run
main: src/main.cpp src/fem.cpp
	g++ -I /home/resiliente/cs601software/eigen-3.3.9 -o $(EXECUTABLE) src/main.cpp src/fem.cpp

run: main
	./$(EXECUTABLE) $(PROB) $(N)

team:
	@echo "Team Members:"
	@echo "1. Kalidindi Dhrutika-210010021"
	@echo "2. Tejal Ladage-210010026"
	@echo "3. Ashwin Waghmare-210010060"
clean:
	rm -f $(EXECUTABLE)
git:
	git status
	git add .
	git commit -m "auto git"
	git push
	git tag -a -f cs601pa2submission -m "Turnin PA1"
	git push -f --tags

.PHONY: all run team clean