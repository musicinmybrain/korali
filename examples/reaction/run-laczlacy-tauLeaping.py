#!/usr/bin/env python3

## In this example, we demonstrate how Korali simulates a reaction.

# Starting Korali's Engine
import korali
k = korali.Engine()

# Creating new experiment
e = korali.Experiment()
e["Random Seed"] = 0xC001

# Configuring Problem
e["Problem"]["Type"] = "Reaction"
e["Problem"]["Reactions"][0]["Equation"] = "PLac+RNAP->PLacRNAP"
e["Problem"]["Reactions"][0]["Rate"] = 0.17

e["Problem"]["Reactions"][1]["Equation"] = "PLacRNAP->PLac+RNAP"
e["Problem"]["Reactions"][1]["Rate"] = 10.

e["Problem"]["Reactions"][2]["Equation"] = "PLacRNAP->TrLacZ1"
e["Problem"]["Reactions"][2]["Rate"] = 1.

e["Problem"]["Reactions"][3]["Equation"] = "TrLacZ1->RbsLacZ + PLac + TrLacZ2"
e["Problem"]["Reactions"][3]["Rate"] = 1.

e["Problem"]["Reactions"][4]["Equation"] = "TrLacZ2->TrLacY1"
e["Problem"]["Reactions"][4]["Rate"] = 0.015

e["Problem"]["Reactions"][5]["Equation"] = "TrLacY1->RbsLacY+TrLacY2"
e["Problem"]["Reactions"][5]["Rate"] = 1.

e["Problem"]["Reactions"][6]["Equation"] = "TrLacY2->RNAP"
e["Problem"]["Reactions"][6]["Rate"] = 0.36

e["Problem"]["Reactions"][7]["Equation"] = "Ribosome+RbsLacZ->RbsRibosomeLacZ"
e["Problem"]["Reactions"][7]["Rate"] = 0.17

e["Problem"]["Reactions"][8]["Equation"] = "Ribosome+RbsLacY->RbsRibosomeLacY"
e["Problem"]["Reactions"][8]["Rate"] = 0.17

e["Problem"]["Reactions"][9]["Equation"] = "RbsRibosomeLacZ->Ribosome+RbsLacZ"
e["Problem"]["Reactions"][9]["Rate"] = 0.45

e["Problem"]["Reactions"][10]["Equation"] = "RbsRibosomeLacY->Ribosome+RbsLacY"
e["Problem"]["Reactions"][10]["Rate"] = 0.45

e["Problem"]["Reactions"][11]["Equation"] = "RbsRibosomeLacZ->TrRbsLacZ+RbsLacZ"
e["Problem"]["Reactions"][11]["Rate"] = 0.4

e["Problem"]["Reactions"][12]["Equation"] = "RbsRibosomeLacY->TrRbsLacY+RbsLacY"
e["Problem"]["Reactions"][12]["Rate"] = 0.4

e["Problem"]["Reactions"][13]["Equation"] = "TrRbsLacZ->LacZ"
e["Problem"]["Reactions"][13]["Rate"] = 0.015

e["Problem"]["Reactions"][14]["Equation"] = "TrRbsLacY->LacY"
e["Problem"]["Reactions"][14]["Rate"] = 0.036

e["Problem"]["Reactions"][15]["Equation"] = "LacZ->dgrLacZ"
e["Problem"]["Reactions"][15]["Rate"] = 6.42e-5

e["Problem"]["Reactions"][16]["Equation"] = "LacY->dgrLacY"
e["Problem"]["Reactions"][16]["Rate"] = 6.42e-5

e["Problem"]["Reactions"][17]["Equation"] = "RbsLacZ->dgrRbsLacZ"
e["Problem"]["Reactions"][17]["Rate"] = 0.3

e["Problem"]["Reactions"][18]["Equation"] = "RbsLacY->dgrRbsLacY"
e["Problem"]["Reactions"][18]["Rate"] = 0.3

e["Problem"]["Reactions"][19]["Equation"] = "LacZ+lactose->LacZlactose"
e["Problem"]["Reactions"][19]["Rate"] = 9.52e-5

e["Problem"]["Reactions"][20]["Equation"] = "LacZlactose->product+LacZ"
e["Problem"]["Reactions"][20]["Rate"] = 431.

e["Problem"]["Reactions"][21]["Equation"] = "LacY->lactose+LacY"
e["Problem"]["Reactions"][21]["Rate"] = 14.

# Configuring Reactants
e["Variables"][0]["Name"] = "PLac"
e["Variables"][0]["Initial Reactant Number"] = 1

e["Variables"][1]["Name"] = "RNAP"
e["Variables"][1]["Initial Reactant Number"] = 35

e["Variables"][2]["Name"] = "Ribosome"
e["Variables"][2]["Initial Reactant Number"] = 350

e["Variables"][3]["Name"] = "PLacRNAP"
e["Variables"][3]["Initial Reactant Number"] = 0

e["Variables"][4]["Name"] = "TrLacZ1"
e["Variables"][4]["Initial Reactant Number"] = 0

e["Variables"][5]["Name"] = "TrLacZ2"
e["Variables"][5]["Initial Reactant Number"] = 0

e["Variables"][6]["Name"] = "RbsLacY"
e["Variables"][6]["Initial Reactant Number"] = 0

e["Variables"][7]["Name"] = "RbsLacZ"
e["Variables"][7]["Initial Reactant Number"] = 0

e["Variables"][8]["Name"] = "TrLacY1"
e["Variables"][8]["Initial Reactant Number"] = 0

e["Variables"][9]["Name"] = "TrLacY2"
e["Variables"][9]["Initial Reactant Number"] = 0

e["Variables"][10]["Name"] = "RbsRibosomeLacY"
e["Variables"][10]["Initial Reactant Number"] = 0

e["Variables"][11]["Name"] = "RbsRibosomeLacZ"
e["Variables"][11]["Initial Reactant Number"] = 0

e["Variables"][12]["Name"] = "TrRbsLacY"
e["Variables"][12]["Initial Reactant Number"] = 0

e["Variables"][13]["Name"] = "TrRbsLacZ"
e["Variables"][13]["Initial Reactant Number"] = 0

e["Variables"][14]["Name"] = "LacY"
e["Variables"][14]["Initial Reactant Number"] = 0

e["Variables"][15]["Name"] = "LacZ"
e["Variables"][15]["Initial Reactant Number"] = 0

e["Variables"][16]["Name"] = "dgrLacY"
e["Variables"][16]["Initial Reactant Number"] = 0

e["Variables"][17]["Name"] = "dgrLacZ"
e["Variables"][17]["Initial Reactant Number"] = 0

e["Variables"][18]["Name"] = "dgrRbsLacY"
e["Variables"][18]["Initial Reactant Number"] = 0

e["Variables"][19]["Name"] = "dgrRbsLacZ"
e["Variables"][19]["Initial Reactant Number"] = 0

e["Variables"][20]["Name"] = "lactose"
e["Variables"][20]["Initial Reactant Number"] = 0

e["Variables"][21]["Name"] = "LacZlactose"
e["Variables"][21]["Initial Reactant Number"] = 0

e["Variables"][22]["Name"] = "product"
e["Variables"][22]["Initial Reactant Number"] = 0

# Configuring TauLeaping parameters
e["Solver"]["Type"] = "SSM/TauLeaping"
e["Solver"]["Simulation Length"] = 2000.
e["Solver"]["Simulations Per Generation"] = 1
e["Solver"]["Nc"] = 10
e["Solver"]["Epsilon"] = 0.03
e["Solver"]["Num SSA Steps"] = 100
e["Solver"]["Acceptance Factor"] = 10
e["Solver"]["Termination Criteria"]["Max Num Simulations"] = 1.
e["Solver"]["Diagnostics"]["Num Bins"] = 5000

# Configuring results path
e["File Output"]["Enabled"] = True
e["File Output"]["Path"] = '_korali_result_laczlacy_tau_leaping'
e["File Output"]["Frequency"] = 1

# Running Korali
k.run(e)
