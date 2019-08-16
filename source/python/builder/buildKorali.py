import sys
import os
import json
from buildAux import *

def buildKorali(koraliDir):
 koraliTemplateHeaderFile = koraliDir + '/korali._hpp'
 koraliNewHeaderFile = koraliDir + '/korali.hpp'
 with open(koraliTemplateHeaderFile, 'r') as file: newHeaderString = file.read()
 
 koraliTemplateCodeFile = koraliDir + '/korali._cpp'
 koraliNewCodeFile = koraliDir + '/korali.cpp'
 with open(koraliTemplateCodeFile, 'r') as file: newCodeString = file.read()
  
 # Loading JSON Configuration
 koraliJsonFile = koraliDir + '/korali.json'
 with open(koraliJsonFile, 'r') as file: koraliJsonString = file.read()
 koraliConfig = json.loads(koraliJsonString) 
 
 ##### Processing header file
 
 headerFilesString= ''
 for root, dirs, files in os.walk(koraliDir):
  for file in files:
   if file.endswith(".hpp"):
    relPath = os.path.relpath(root, koraliDir)
    headerFilesString += '#include "' + os.path.join(relPath, file) + '"\n'

 newHeaderString = newHeaderString.replace('// Include Files', headerFilesString)
 
 ##### Processing code file
   
 print('[Korali] Creating: ' + koraliNewHeaderFile + '...')
 with open(koraliNewHeaderFile, 'w') as file: file.write(newHeaderString)
 
 print('[Korali] Creating: ' + koraliNewCodeFile + '...')
 with open(koraliNewCodeFile, 'w') as file: file.write(newCodeString)
 
 
 
 
