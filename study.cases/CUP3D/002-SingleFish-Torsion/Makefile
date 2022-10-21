KORALICXX=${CXX}
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

.PHONY: clean all
all: train-swimmer-2D train-swimmer-3D test-swimmer-2D test-swimmer-3D

CUP2DFOLDER=../../CUP2D/_deps/CUP-2D
CUP3DFOLDER=../_deps/CUP-3D
CUBISM2DCFLAGS=$(shell cat ${CUP2DFOLDER}/makefiles/cup.cflags.txt) -I${CUP2DFOLDER}/source -I${CUP2DFOLDER}/Cubism/include -DCUBISM -Wno-shadow -Wno-float-equal
CUBISM3DCFLAGS=$(shell cat ${CUP3DFOLDER}/makefiles/cup.cflags.txt) -I${CUP3DFOLDER}/source -I${CUP3DFOLDER}/Cubism/include -DCUBISM -Wno-shadow -Wno-float-equal
CUBISM2DLIBS= ${CUP2DFOLDER}/makefiles/libcup.a $(shell cat ${CUP2DFOLDER}/makefiles/cup.libs.txt) -lstdc++fs
CUBISM3DLIBS= ${CUP3DFOLDER}/makefiles/libcup.a $(shell cat ${CUP3DFOLDER}/makefiles/cup.libs.txt) -lstdc++fs

train-swimmer-2D.o:
	$(KORALICXX) -c -o train-swimmer-2D.o vracer-swimmer.cpp $(KORALICFLAGS) $(CUBISM2DCFLAGS) -DmodelDIM=2
train-swimmer-3D.o:
	$(KORALICXX) -c -o train-swimmer-3D.o vracer-swimmer.cpp $(KORALICFLAGS) $(CUBISM3DCFLAGS) -DmodelDIM=3
test-swimmer-2D.o:
	$(KORALICXX) -c -o test-swimmer-2D.o vracer-swimmer.cpp $(KORALICFLAGS) $(CUBISM2DCFLAGS) -DmodelDIM=2 -DEVALUATION
test-swimmer-3D.o:
	$(KORALICXX) -c -o test-swimmer-3D.o vracer-swimmer.cpp $(KORALICFLAGS) $(CUBISM3DCFLAGS) -DmodelDIM=3 -DEVALUATION

train-swimmer-2D: train-swimmer-2D.o ${CUP2DFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o train-swimmer-2D $^ $(KORALILIBS) $(CUBISM2DLIBS)
train-swimmer-3D: train-swimmer-3D.o ${CUP3DFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o train-swimmer-3D $^ $(KORALILIBS) $(CUBISM3DLIBS)

test-swimmer-2D: test-swimmer-2D.o ${CUP2DFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o test-swimmer-2D $^ $(KORALILIBS) $(CUBISM2DLIBS)
test-swimmer-3D: test-swimmer-3D.o ${CUP3DFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o test-swimmer-3D $^ $(KORALILIBS) $(CUBISM3DLIBS)

clean:
	rm -rf train-swimmer-3D train-swimmer-2D test-swimmer-2D test-swimmer-3D *.txt *.o *.dat
