import sys
import os
import json

####################################################################
# Helper Functions
####################################################################

def getVariableType(v):
 # Replacing bools with ints for Python compatibility
 return v['Type'].replace('bool', 'int').replace('std::function<void(korali::Sample&)>', 'std::uint64_t')

def getCXXVariableName(v):
 cVarName = ''
 for name in v: cVarName += name
 cVarName = cVarName.replace(" ", "")
 cVarName = cVarName.replace("(", "")
 cVarName = cVarName.replace(")", "")
 cVarName = cVarName.replace("+", "")
 cVarName = cVarName.replace("-", "")
 cVarName = cVarName.replace("[", "")
 cVarName = cVarName.replace("]", "")
 cVarName = '_' + cVarName[0].lower() + cVarName[1:]
 return cVarName

def getVariablePath(v):
 cVarPath = ''
 for name in v["Name"]: cVarPath += '["' + name + '"]'
 return cVarPath

def getVariableOptions(v):
 options = []
 if ( v.get('Options', '') ):
  for item in v["Options"]:
   options.append(item["Value"])
 return options

def getOptionName(path):
 nameList = path.rsplit('/')
 optionName = ''
 for name in nameList[1:-1]:
  optionName += name[0].capitalize() + name[1:] + '/'
 optionName += getModuleName(path)
 return optionName

def getModuleName(path):
 nameList = path.rsplit('/', 1)
 moduleName = nameList[-1]
 moduleName = moduleName[0].capitalize() + moduleName[1:]
 return moduleName

def getClassName(path):
 nameList = path.rsplit('/')
 className = 'korali::'
 for name in nameList[:-1]:
  className += name + '::'
 className += getModuleName(path)
 return className

def getParentClassName(className):
 nameString = className.rsplit('::', 2)
 if (len(nameString) == 2): return 'korali::Module'
 parentClass = nameString[0] + '::' + nameString[-2].capitalize()
 return parentClass

def isLeafModule(path):
 for curDir, relDir, fileNames in os.walk(path):
  if (curDir != path):
   for fileName in fileNames:
    if '.config' in fileName:
     return False
 return True

#####################################################################

def consumeValue(base, moduleName, path, varName, varType, isMandatory, options):
 cString = '\n'

 if ('std::function' in varType):
  cString += ' ' + varName + ' = ' + base + path + '.get<size_t>();\n'
  cString += '   korali::JsonInterface::eraseValue(' + base + ', "' + path.replace('"', "'") + '");\n'
  return cString
  
 if ('korali::Sample' in varType):
  cString = ''
  return cString

 if ('std::vector<korali::Variable' in varType):
  baseType = varType.replace('std::vector<', '').replace('>','')
  cString += ' ' + varName + '.clear();\n'
  cString += ' for(size_t i = 0; i < ' + base + path + '.size(); i++) ' + varName + '.push_back(new korali::Variable);\n'
  return cString

 if ('std::vector<korali::Variable*>' in varType):
  baseType = varType.replace('std::vector<', '').replace('>','')
  cString += ' for(size_t i = 0; i < ' + base + path + '.size(); i++) ' + varName + '.push_back(new korali::Variable());\n'
  cString += ' korali::JsonInterface::eraseValue(' + base + ', "' + path.replace('"', "'") + '");\n\n'
  return cString

 if ('std::vector<korali::' in varType):
  baseType = varType.replace('std::vector<', '').replace('>','')
  cString += ' for(size_t i = 0; i < ' + base + path + '.size(); i++) ' + varName + '.push_back((' + baseType + ')korali::Module::getModule(' + base + path + '[i]));\n'
  cString += ' korali::JsonInterface::eraseValue(' + base + ', "' + path.replace('"', "'") + '");\n\n'
  return cString

 rhs = base + path + '.get<' + varType + '>();\n'
 
 if ('korali::' in varType):
  rhs = 'dynamic_cast<' + varType + '>(korali::Module::getModule(' + base + path + '));\n'

 if ('gsl_rng*' in varType):
  rhs = 'setRange(' + base + path + '.get<std::string>());\n'
 
 cString += ' if (korali::JsonInterface::isDefined(' + base + ', "' + path.replace('"', "'") + '"))  \n  { \n'
 cString += '   ' + varName + ' = ' + rhs
 cString += '   korali::JsonInterface::eraseValue(' + base + ', "' + path.replace('"', "'") + '");\n'
 cString += '  }\n'

 if (isMandatory):
  cString += '  else '
  cString += '  korali::logError("No value provided for mandatory setting: ' + path.replace('"', "'") + ' required by ' + moduleName + '.\\n"); \n'

 cString += '\n'

 if (options):
  cString += '{\n'
  validVarName = 'validOption'
  cString += ' bool ' + validVarName + ' = false; \n'
  for v in options:
   cString += ' if (' + varName + ' == "' + v + '") ' + validVarName + ' = true; \n'
  cString += ' if (' + validVarName + ' == false) korali::logError("Unrecognized value provided for mandatory setting: ' + path.replace('"', "'") + ' required by ' + moduleName + '.\\n"); \n'
  cString += '}\n'

 cString += '\n'
 return cString

#####################################################################

def saveValue(base, path, varName, varType):

 if ('korali::Sample' in varType):
  sString = ''
  return sString
  
 if ('gsl_rng*' in varType):
  sString = '   ' + base + path + ' = getRange(' + varName + ');\n'
  return sString

 if ('korali::Variable' in varType):
  sString = ''
  return sString

 if ('std::vector<korali::' in varType):
  sString = ' for(size_t i = 0; i < ' + varName + '.size(); i++) ' + varName + '[i]->getConfiguration(' + base + path + '[i]);\n'
  return sString

 if ('korali::' in varType):
  sString =  ' ' + varName + '->getConfiguration(' + base + path + ');\n'
  return sString

 sString = '   ' + base + path + ' = ' + varName + ';\n'
 return sString


####################################################################

def createSetConfiguration(module):
 codeString = 'void ' + module["Class"] + '::setConfiguration(nlohmann::json& js) \n{\n'

 # Consume Configuration Settings
 if 'Configuration Settings' in module:
  for v in module["Configuration Settings"]:
   codeString += consumeValue('js', module["Name"], getVariablePath(v), getCXXVariableName(v["Name"]), getVariableType(v), True, getVariableOptions(v))

 if 'Internal Settings' in module:
  for v in module["Internal Settings"]:
   codeString += consumeValue('js', module["Name"], getVariablePath(v),  getCXXVariableName(v["Name"]), getVariableType(v), False, getVariableOptions(v))

 if 'Termination Criteria' in module:
  for v in module["Termination Criteria"]:
   codeString += consumeValue('js', module["Name"], '["Termination Criteria"]' + getVariablePath(v), getCXXVariableName(v["Name"]), getVariableType(v), True, getVariableOptions(v))

 if 'Variables Configuration' in module:
  codeString += ' for (size_t i = 0; i < _k->_js["Variables"].size(); i++) { \n'
  for v in module["Variables Configuration"]:
   codeString += consumeValue('_k->_js["Variables"][i]', module["Name"], getVariablePath(v), '_k->_variables[i]->' + getCXXVariableName(v["Name"]), getVariableType(v), True, getVariableOptions(v))
  codeString += ' } \n'

 if 'Conditional Variables' in module:
  codeString += '  _hasConditionalVariables = false; \n'
  for v in module["Conditional Variables"]:
   codeString += ' if(js' + getVariablePath(v) + '.is_number()) ' + getCXXVariableName(v["Name"]) + ' = js' + getVariablePath(v) + ';\n'
   codeString += ' if(js' + getVariablePath(v) + '.is_string()) { _hasConditionalVariables = true; ' + getCXXVariableName(v["Name"]) + 'Conditional = js' + getVariablePath(v) + '; } \n'
   codeString += ' korali::JsonInterface::eraseValue(js, "' + getVariablePath(v).replace('"', "'") + '");\n\n'

 codeString += ' ' + module["Parent Class"] + '::setConfiguration(js);\n'

 codeString += ' _type = "' + module["Option Name"] + '";\n'
 codeString += ' if(korali::JsonInterface::isDefined(js, "[\'Type\']")) korali::JsonInterface::eraseValue(js, "[\'Type\']");\n'
 codeString += ' if(korali::JsonInterface::isEmpty(js) == false) korali::logError("Unrecognized settings for Korali module: ' + module["Name"] + ': \\n%s\\n", js.dump(2).c_str());\n'
 codeString += '} \n\n'

 return codeString

####################################################################

def createGetConfiguration(module):
 codeString = 'void ' + module["Class"]  + '::getConfiguration(nlohmann::json& js) \n{\n\n'

 codeString += ' js["Type"] = _type;\n'

 if 'Configuration Settings' in module:
  for v in module["Configuration Settings"]:
   codeString += saveValue('js', getVariablePath(v), getCXXVariableName(v["Name"]), getVariableType(v))

 if 'Termination Criteria' in module:
  for v in module["Termination Criteria"]:
   codeString += saveValue('js', '["Termination Criteria"]' + getVariablePath(v), getCXXVariableName(v["Name"]), getVariableType(v))

 if 'Internal Settings' in module:
  for v in module["Internal Settings"]:
   codeString += saveValue('js', getVariablePath(v), getCXXVariableName(v["Name"]), getVariableType(v))

 if 'Variables Configuration' in module:
  codeString += ' for (size_t i = 0; i <  _k->_variables.size(); i++) { \n'
  for v in module["Variables Configuration"]:
   codeString += saveValue('_k->_js["Variables"][i]', getVariablePath(v), '_k->_variables[i]->' + getCXXVariableName(v["Name"]), getVariableType(v))
  codeString += ' } \n'

 if 'Conditional Variables' in module:
  for v in module["Conditional Variables"]:
   codeString += ' if(' + getCXXVariableName(v["Name"]) + 'Conditional == "") js' + getVariablePath(v) + ' = ' + getCXXVariableName(v["Name"]) + ';\n'
   codeString += ' if(' + getCXXVariableName(v["Name"]) + 'Conditional != "") js' + getVariablePath(v) + ' = ' + getCXXVariableName(v["Name"]) + 'Conditional; \n'

 codeString += ' ' + module["Parent Class"] + '::getConfiguration(js);\n'

 codeString += '} \n\n'

 return codeString
 
####################################################################

def createApplyModuleDefaults(module):
 codeString = 'void ' + module["Class"]  + '::applyModuleDefaults(nlohmann::json& js) \n{\n\n'

 if 'Module Defaults' in module:
   codeString += ' std::string defaultString = "' + json.dumps(module["Module Defaults"]).replace('"','\\"') + '";\n'
   codeString += ' nlohmann::json defaultJs = nlohmann::json::parse(defaultString);\n'
   codeString += ' JsonInterface::mergeJson(js, defaultJs); \n'

 codeString += ' '  + module["Parent Class"] + '::applyModuleDefaults(js);\n'
    
 codeString += '} \n\n'

 return codeString
 
####################################################################

def createApplyVariableDefaults(module):
 codeString = 'void ' + module["Class"]  + '::applyVariableDefaults() \n{\n\n'

 if 'Variable Defaults' in module:
  codeString += ' std::string defaultString = "' + json.dumps(module["Variable Defaults"]).replace('"','\\"') + '";\n'
  codeString += ' nlohmann::json defaultJs = nlohmann::json::parse(defaultString);\n'
  codeString += ' for (size_t i = 0; i < _k->_js["Variables"].size(); i++) \n'
  codeString += '  JsonInterface::mergeJson(_k->_js["Variables"][i], defaultJs); \n'

 codeString += ' '  + module["Parent Class"] + '::applyVariableDefaults();\n'
 codeString += '} \n\n'

 return codeString

####################################################################

def createCheckTermination(module):
 codeString = 'bool ' + module["Class"]  + '::checkTermination()\n'
 codeString += '{\n'
 codeString += ' bool hasFinished = false;\n\n'

 if 'Termination Criteria' in module:
  for v in module["Termination Criteria"]:
   codeString += ' if (' + v["Criteria"] + ')\n'
   codeString += ' {\n'
   codeString += '  _terminationCriteria.push_back("' + module["Name"] + getVariablePath(v).replace('"', "'") + ' = " + std::to_string(' + getCXXVariableName(v["Name"]) +') + ".");\n'
   codeString += '  hasFinished = true;\n'
   codeString += ' }\n\n'

 codeString += ' hasFinished = hasFinished || ' + module["Parent Class"] + '::checkTermination();\n'
 codeString += ' return hasFinished;\n'
 codeString += '}\n\n'

 return codeString

####################################################################

def createRunOperation(module):
  codeString = 'bool ' + module["Class"]  + '::runOperation(std::string operation, korali::Sample& sample)\n'
  codeString += '{\n'
  codeString += ' bool operationDetected = false;\n\n'

  for v in module["Available Operations"]:
   codeString += ' if (operation == "' + v["Name"] + '")\n'
   codeString += ' {\n'
   codeString += '  ' + v["Function"] + '(sample);\n'
   codeString += '  return true;\n'
   codeString += ' }\n\n'

  codeString += ' operationDetected = operationDetected || ' + module["Parent Class"] + '::runOperation(operation, sample);\n'
  codeString += ' if (operationDetected == false) korali::logError("Operation %s not recognized for problem ' + module["Class"] + '.\\n", operation.c_str());\n'
  codeString += ' return operationDetected;\n'
  codeString += '}\n\n'

  return codeString

####################################################################

def createGetPropertyPointer(module):
  codeString = 'double* ' + module["Class"]  + '::getPropertyPointer(std::string property)\n'
  codeString += '{\n'

  for v in module["Conditional Variables"]:
   codeString += ' if (property == "' + v["Name"][0] + '") return &' + getCXXVariableName(v["Name"]) + ';\n'

  codeString += ' korali::logError("Property %s not recognized for distribution ' + module["Class"] + '.\\n", property.c_str());\n'
  codeString += ' return NULL;\n'
  codeString += '}\n\n'

  return codeString

####################################################################

def createHeaderDeclarations(module):
 headerString = ''

 if 'Configuration Settings' in module:
  for v in module["Configuration Settings"]:
   headerString += ' ' + getVariableType(v) + ' ' + getCXXVariableName(v["Name"]) + ';\n'

 if 'Internal Settings' in module:
  for v in module["Internal Settings"]:
   headerString += ' ' + getVariableType(v) + ' ' + getCXXVariableName(v["Name"]) + ';\n'

 if 'Termination Criteria' in module:
  for v in module["Termination Criteria"]:
   headerString += ' ' + getVariableType(v) + ' ' + getCXXVariableName(v["Name"]) + ';\n'

 if 'Conditional Variables' in module:
  for v in module["Conditional Variables"]:
   headerString += ' double ' + getCXXVariableName(v["Name"]) + ';\n'
   headerString += ' std::string ' + getCXXVariableName(v["Name"]) + 'Conditional;\n'

 return headerString


####################################################################

def createVariableDeclarations(module):
 variableDeclarationString = ''

 if 'Variables Configuration' in module:
  for v in module["Variables Configuration"]:
   variableDeclarationString += '  ' + getVariableType(v) + ' ' + getCXXVariableName(v["Name"]) + ';\n'

 return variableDeclarationString

####################################################################

def save_if_different(filename, content):
    try:
        with open(filename) as f:
            existing = f.read()
        if content == existing:
            return
    except FileNotFoundError:
        pass

    with open(filename, 'w') as f:
        print('[Korali] Creating: ' + filename + '...')
        f.write(content)

####################################################################
# Main Parser Routine
####################################################################

print("\n[Korali] Start Parser")

koraliDir = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# modules List
moduleDetectionList = ''
moduleIncludeList = ''

# Variable Declaration List
varDeclarationSet = set()

# Detecting modules' json file
for moduleDir, relDir, fileNames in os.walk(koraliDir):
 for fileName in fileNames:
  if '.config' in fileName:
   filePath = moduleDir + '/' + fileName;
   print('[Korali] Opening: ' + filePath + '...')
   moduleFilename = fileName.replace('.config', '')
   
   # Loading Json configuration file
   with open(filePath, 'r') as file: moduleConfig = json.load(file)
   
   # Processing Module information
   modulePath = os.path.relpath(moduleDir, koraliDir)
   moduleConfig["Name"] =  getModuleName(modulePath)
   moduleConfig["Class"] =  getClassName(modulePath)
   moduleConfig["Parent Class"] =  getParentClassName(moduleConfig["Class"])
   moduleConfig["Option Name"] = getOptionName(modulePath)
   moduleConfig["Is Leaf"] = isLeafModule(modulePath)

   ####### Adding module to list
   if (moduleConfig["Is Leaf"]):
    relpath = os.path.relpath(moduleDir, koraliDir)
    filepath = os.path.join(relpath, moduleFilename + '.hpp')
    moduleIncludeList += '#include "' + filepath + '" \n'
    moduleDetectionList += '  if(moduleType == "' + moduleConfig["Option Name"] + '") module = new ' + moduleConfig["Class"] + '();\n'

   ###### Producing module code

   moduleCodeString = createSetConfiguration(moduleConfig)
   moduleCodeString += createGetConfiguration(moduleConfig)
   moduleCodeString += createApplyModuleDefaults(moduleConfig)
   moduleCodeString += createApplyVariableDefaults(moduleConfig)
   moduleCodeString += createCheckTermination(moduleConfig)

   if 'Available Operations' in moduleConfig:
     moduleCodeString += createRunOperation(moduleConfig)

   if 'Conditional Variables' in moduleConfig:
     moduleCodeString += createGetPropertyPointer(moduleConfig)

   ####### Producing header file

   # Loading template header .hpp file
   moduleTemplateHeaderFile = moduleDir + '/' + moduleFilename + '._hpp'
   with open(moduleTemplateHeaderFile, 'r') as file: moduleTemplateHeaderString = file.read()

   # Adding overridden function declarations
   functionOverrideString = ''
   functionOverrideString += ' bool checkTermination() override;\n'
   functionOverrideString += ' void getConfiguration(nlohmann::json& js) override;\n'
   functionOverrideString += ' void setConfiguration(nlohmann::json& js) override;\n'
   functionOverrideString += ' void applyModuleDefaults(nlohmann::json& js) override;\n'
   functionOverrideString += ' void applyVariableDefaults() override;\n'

   if 'Available Operations' in moduleConfig:
     functionOverrideString += ' bool runOperation(std::string, korali::Sample& sample) override;\n'

   if 'Conditional Variables' in moduleConfig:
     functionOverrideString += ' double* getPropertyPointer(std::string property) override;\n'

   newHeaderString = moduleTemplateHeaderString.replace('public:', 'public: \n' + functionOverrideString + '\n')

   # Adding declarations
   declarationsString = createHeaderDeclarations(moduleConfig)
   newHeaderString = newHeaderString.replace('public:', 'public: \n' + declarationsString + '\n')

   # Retrieving variable declarations
   for varDecl in createVariableDeclarations(moduleConfig).splitlines():
    varDeclarationSet.add(varDecl)

   # Saving new header .hpp file
   moduleNewHeaderFile = moduleDir + '/' + moduleFilename + '.hpp'
   save_if_different(moduleNewHeaderFile, newHeaderString)

   ###### Creating code file

   moduleBaseCodeFileName = moduleDir + '/' + moduleFilename + '._cpp'
   moduleNewCodeFile = moduleDir + '/' + moduleFilename + '.cpp'
   baseFileTime = os.path.getmtime(moduleBaseCodeFileName)
   newFileTime = baseFileTime
   if (os.path.exists(moduleNewCodeFile)): newFileTime = os.path.getmtime(moduleNewCodeFile)

   if (baseFileTime >= newFileTime):
    with open(moduleBaseCodeFileName, 'r') as file: moduleBaseCodeString = file.read()
    moduleBaseCodeString += '\n\n' + moduleCodeString
    save_if_different(moduleNewCodeFile, moduleBaseCodeString)

###### Updating module source file

moduleBaseCodeFileName = koraliDir + '/module._cpp'
moduleNewCodeFile = koraliDir + '/module.cpp'
baseFileTime = os.path.getmtime(moduleBaseCodeFileName)
newFileTime = baseFileTime
if (os.path.exists(moduleNewCodeFile)): newFileTime = os.path.getmtime(moduleNewCodeFile)

if (baseFileTime >= newFileTime):
  with open(moduleBaseCodeFileName, 'r') as file: moduleBaseCodeString = file.read()
  newBaseString = moduleBaseCodeString.replace('// Module Include List',  moduleIncludeList)
  newBaseString = newBaseString.replace(' // Module Selection List', moduleDetectionList)
  save_if_different(moduleNewCodeFile, newBaseString)

###### Updating module header file

moduleBaseHeaderFileName = koraliDir + '/module._hpp'
moduleNewHeaderFile = koraliDir + '/module.hpp'
with open(moduleBaseHeaderFileName, 'r') as file: moduleBaseHeaderString = file.read()
newBaseString = moduleBaseHeaderString
save_if_different(moduleNewHeaderFile, newBaseString)

###### Updating variable header file

variableDeclarationList = '\n'.join(sorted(varDeclarationSet))

variableBaseHeaderFileName = koraliDir + '/experiment/variable/variable._hpp'
variableNewHeaderFile = koraliDir + '/experiment/variable/variable.hpp'
with open(variableBaseHeaderFileName, 'r') as file: variableBaseHeaderString = file.read()
newBaseString = variableBaseHeaderString
newBaseString = newBaseString.replace(' // Variable Declaration List', variableDeclarationList)
save_if_different(variableNewHeaderFile, newBaseString)

print("[Korali] End Parser\n")
