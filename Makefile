CXX?=c++
CXXFLAGS?=-O2 -DNDEBUG
CLANGTIDY?=clang-tidy
CLANGFORMAT?=clang-format

TIDYFILES=minion.hpp minion.cc
FORMATFILES=minion.hpp minion.cc

CXXFLAGS+= -std=c++11

default: all

all:

.PHONY: all default

tidy:
	$(CLANGTIDY) $(TIDYFILES) -- $(CXXFLAGS) -Wall

format:
	$(CLANGFORMAT) -i $(FORMATFILES)

.PHONY: tidy format

example: example.cc minion.hpp
	$(CXX) $(CXXFLAGS) -o $@ $<