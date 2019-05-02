CXX?=c++
CXXFLAGS?=-O2 -DNDEBUG
CLANGTIDY?=clang-tidy
CLANGFORMAT?=clang-format

TIDYFILES=sparkyrng.hpp sparkyrng.cc
FORMATFILES=sparkyrng.hpp sparkyrng.cc

CXXFLAGS+= -std=c++11

default: all

all:

.PHONY: all default

tidy:
	$(CLANGTIDY) $(TIDYFILES) -- $(CXXFLAGS) -Wall

format:
	$(CLANGFORMAT) -i $(FORMATFILES)

.PHONY: tidy format