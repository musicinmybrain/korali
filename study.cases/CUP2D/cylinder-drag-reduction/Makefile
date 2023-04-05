KORALICXX=${CXX}
KORALICFLAGS=`python3 -m korali.cxx --cflags`
KORALILIBS=`python3 -m korali.cxx --libs`

#2D 
CYLINDER = run-vracer-cylinder eval-vracer-cylinder run-cmaes-cylinder
CUPFOLDER=../_deps/CUP-2D
CUBISMCFLAGS=$(shell cat ${CUPFOLDER}/makefiles/cup.cflags.txt) -I${CUPFOLDER}/source -I${CUPFOLDER}/Cubism/include -DCUBISM -Wno-shadow
CUBISMLIBS= ${CUPFOLDER}/makefiles/libcup.a $(shell cat ${CUPFOLDER}/makefiles/cup.libs.txt) -lstdc++fs

#3D
CYLINDER3D = eval-vracer-cylinder3D
CUPFOLDER3D=../_deps/CUP-3D
CUBISMCFLAGS3D=$(shell cat ${CUPFOLDER3D}/makefiles/cup.cflags.txt) -I${CUPFOLDER3D}/source -I${CUPFOLDER3D}/Cubism/include -DCUBISM -Wno-shadow
CUBISMLIBS3D= ${CUPFOLDER3D}/makefiles/libcup.a $(shell cat ${CUPFOLDER3D}/makefiles/cup.libs.txt) -lstdc++fs
#CUPFOLDER3D=../_deps/CUP-2D
#CUBISMCFLAGS3D=$(shell cat ${CUPFOLDER}/makefiles/cup.cflags.txt) -I${CUPFOLDER}/source -I${CUPFOLDER}/Cubism/include -DCUBISM -Wno-shadow
#CUBISMLIBS3D= ${CUPFOLDER}/makefiles/libcup.a $(shell cat ${CUPFOLDER}/makefiles/cup.libs.txt) -lstdc++fs


.SECONDARY:
.PHONY: all
all: cylinder

cylinder: $(CYLINDER) $(CYLINDER3D)

#2D
$(CYLINDER): %: %.o cylinderEnvironment.o ${CUPFOLDER}/makefiles/libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS) -DmodelDIM=2
%.o: %.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS) -DmodelDIM=2
cylinderEnvironment.o: _model/cylinderEnvironment.cpp
	$(KORALICXX) -c $< $(KORALICFLAGS) $(CUBISMCFLAGS) -DmodelDIM=2

#3D
#run-vracer-cylinder3D.o: run-vracer-cylinder.cpp
#	$(KORALICXX) -c -o run-vracer-cylinder3D.o  run-vracer-cylinder.cpp $(KORALICFLAGS) $(CUBISMCFLAGS3D) -DmodelDIM=3
eval-vracer-cylinder3D.o: eval-vracer-cylinder.cpp
	$(KORALICXX) -c -o eval-vracer-cylinder3D.o  eval-vracer-cylinder.cpp $(KORALICFLAGS) $(CUBISMCFLAGS3D) -DmodelDIM=3
cylinderEnvironment3D.o: _model/cylinderEnvironment.cpp
	$(KORALICXX) -c -o cylinderEnvironment3D.o  _model/cylinderEnvironment.cpp $(KORALICFLAGS) $(CUBISMCFLAGS3D) -DmodelDIM=3
$(CYLINDER3D): %: eval-vracer-cylinder3D.o cylinderEnvironment3D.o ${CUPFOLDER3D}/makefiles/libcup.a
	$(KORALICXX) -o $@ $^ $(KORALILIBS) $(CUBISMLIBS3D) -DmodelDIM=3


.PHONY: clean
clean:
	$(RM) $(CYLINDER) $(CYLINDER3D) *.txt *.o *.ti *.optrpt *.h5 *.dat *.xmf 
