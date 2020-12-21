CXX=c++
CXXFLAGS=-O2 -DNDEBUG
CLANGTIDY=clang-tidy
CLANGFORMAT=clang-format

TIDYFILES=random.hpp random.cpp
FORMATFILES=random.hpp random.cpp

CXXFLAGS+= -std=c++17

default: all

all:

.PHONY: all default

tidy:
	$(CLANGTIDY) $(TIDYFILES) -- $(CXXFLAGS) -Wall

format:
	$(CLANGFORMAT) -i $(FORMATFILES)

.PHONY: tidy format

example: example.cc random.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<