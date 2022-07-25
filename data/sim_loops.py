from __future__ import division

#new lists to hold data
day_list = [] #added
numPositive_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
compliance_list_yes = []
compliance_list_no = []
agent_base_compliance = []

import pickle
import numpy
import time
import networkx
import matplotlib
import random


def run_tti_sim(model, T, max_dt=None,
                intervention_start_pct_infected=0, average_introductions_per_day=0,
                testing_cadence='everyday', pct_tested_per_day=1.0, test_falseneg_rate='temporal',
                testing_compliance_symptomatic=[None], max_pct_tests_for_symptomatics=1.0,
                testing_compliance_traced=[None], max_pct_tests_for_traces=1.0,
                testing_compliance_random=[None], random_testing_degree_bias=0,
                tracing_compliance=[None], num_contacts_to_trace=None, pct_contacts_to_trace=1.0, tracing_lag=1,
                isolation_compliance_symptomatic_individual=[None], isolation_compliance_symptomatic_groupmate=[None],
                isolation_compliance_positive_individual=[None], isolation_compliance_positive_groupmate=[None],
                isolation_compliance_positive_contact=[None], isolation_compliance_positive_contactgroupmate=[None],
                isolation_lag_symptomatic=1, isolation_lag_positive=1, isolation_lag_contact=0, isolation_groups=None,
                cadence_testing_days=None, cadence_cycle_length=28, temporal_falseneg_rates=None,
                backlog_skipped_intervals=False, base_testing_compliance_rate_symptomatic = 0, base_testing_compliance_rate_traced = 0, base_testing_compliance_rate_random = 0,
                base_tracing_compliance_rate = 0, base_isolation_compliance_rate_symptomatic_individual = 0, base_isolation_compliance_rate_symptomatic_groupmate = 0, base_isolation_compliance_rate_positive_individual = 0,
                base_isolation_compliance_rate_positive_groupmate = 0, base_isolation_compliance_rate_positive_contact = 0, base_isolation_compliance_rate_positive_contactgroupmate = 0, produce_image = False, save_folder = None
                ):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    N = len(model.G)
    agent_base_compliance = numpy.random.rand(N)
    testing_compliance_symptomatic = (agent_base_compliance < base_testing_compliance_rate_symptomatic)
    testing_compliance_traced = (agent_base_compliance < base_testing_compliance_rate_traced)
    testing_compliance_random = (agent_base_compliance < base_testing_compliance_rate_random)
    tracing_compliance = (agent_base_compliance < base_tracing_compliance_rate)
    isolation_compliance_symptomatic_individual = (agent_base_compliance < base_isolation_compliance_rate_symptomatic_individual)
    isolation_compliance_symptomatic_groupmate = (agent_base_compliance < base_isolation_compliance_rate_symptomatic_groupmate)
    isolation_compliance_positive_individual = (agent_base_compliance < base_isolation_compliance_rate_positive_individual)
    isolation_compliance_positive_groupmate = (agent_base_compliance < base_isolation_compliance_rate_positive_groupmate)
    isolation_compliance_positive_contact = (agent_base_compliance < base_isolation_compliance_rate_positive_contact)
    isolation_compliance_positive_contactgroupmate = (agent_base_compliance < base_isolation_compliance_rate_positive_contactgroupmate)

    # Testing cadences involve a repeating 28 day cycle starting on a Monday
    # (0:Mon, 1:Tue, 2:Wed, 3:Thu, 4:Fri, 5:Sat, 6:Sun, 7:Mon, 8:Tues, ...)
    # For each cadence, testing is done on the day numbers included in the associated list.

    if (cadence_testing_days is None):
        cadence_testing_days = {
            'TESTING_COMPLIANCE_RATE_RANDOM': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                                               21, 22, 23, 24, 25, 26, 27],
            'workday': [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 14, 15, 16, 17, 18, 21, 22, 23, 24, 25],
            'semiweekly': [0, 3, 7, 10, 14, 17, 21, 24],
            'weekly': [0, 7, 14, 21],
            'biweekly': [0, 14],
            'monthly': [0],
            'cycle_start': [0]
        }

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (temporal_falseneg_rates is None):
        temporal_falseneg_rates = {
            model.E: {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00},
            model.I_pre: {0: 0.25, 1: 0.25, 2: 0.22},
            model.I_sym: {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34, 9: 0.38,
                          10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76, 18: 0.79,
                          19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97,
                          28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
            model.I_asym: {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34, 9: 0.38,
                           10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76, 18: 0.79,
                           19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97,
                           28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
            model.Q_E: {0: 1.00, 1: 1.00, 2: 1.00, 3: 1.00},
            model.Q_pre: {0: 0.25, 1: 0.25, 2: 0.22},
            model.Q_sym: {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34, 9: 0.38,
                          10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76, 18: 0.79,
                          19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97,
                          28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
            model.Q_asym: {0: 0.19, 1: 0.16, 2: 0.16, 3: 0.17, 4: 0.19, 5: 0.22, 6: 0.26, 7: 0.29, 8: 0.34, 9: 0.38,
                           10: 0.43, 11: 0.48, 12: 0.52, 13: 0.57, 14: 0.62, 15: 0.66, 16: 0.70, 17: 0.76, 18: 0.79,
                           19: 0.82, 20: 0.85, 21: 0.88, 22: 0.90, 23: 0.92, 24: 0.93, 25: 0.95, 26: 0.96, 27: 0.97,
                           28: 0.97, 29: 0.98, 30: 0.98, 31: 0.99},
        }

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Custom simulation loop:
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    interventionOn = False
    interventionStartTime = None

    timeOfLastIntervention = -1
    timeOfLastIntroduction = -1

    testingDays = cadence_testing_days[testing_cadence]
    cadenceDayNumber = 0

    tests_per_day = int(model.numNodes * pct_tested_per_day)
    max_tracing_tests_per_day = int(tests_per_day * max_pct_tests_for_traces)
    max_symptomatic_tests_per_day = int(tests_per_day * max_pct_tests_for_symptomatics)

    tracingPoolQueue = [[] for i in range(tracing_lag)]
    isolationQueue_symptomatic = [[] for i in range(isolation_lag_symptomatic)]
    isolationQueue_positive = [[] for i in range(isolation_lag_positive)]
    isolationQueue_contact = [[] for i in range(isolation_lag_contact)]

    model.tmax = T
    running = True
    while running:

        def addition():
            # our addition starts here
            print(str(sum(numPositive_list[-14:]) / N * 100) + "% cummulative 2 week positive test")
            print(numPositive_list)
            new_testing_compliance_symptomatic = (numpy.random.rand(N) < min(1, 0.5 + ((1 - 0.5) * (sum(numPositive_list[-14:]) / N * 2))))
            #print(str(min(1, 0.3 + (sum(numPositive_list[-14:]) / N * 3))) + "aaaaa")
            new_testing_compliance_rate_traced = base_testing_compliance_rate_traced
            new_testing_compliance_random = (numpy.random.rand(N) < min(1, 0.5 + ((1 - 0.5) * (sum(numPositive_list[-14:]) / N * 2))))
            new_tracing_compliance_rate = base_tracing_compliance_rate
            new_isolation_compliance_rate_symptomatic_individual = base_isolation_compliance_rate_symptomatic_individual
            new_isolation_compliance_rate_symptomatic_groupmate = base_isolation_compliance_rate_symptomatic_groupmate
            new_isolation_compliance_rate_positive_individual = (numpy.random.rand(N) < min(1, 0.8 + ((1 - 0.8) * (sum(numPositive_list[-14:]) / N * 2))))
            new_isolation_compliance_rate_positive_groupmate = base_isolation_compliance_rate_positive_groupmate
            new_isolation_compliance_rate_positive_contact = base_isolation_compliance_rate_positive_contact
            new_isolation_compliance_rate_positive_contactgroupmate = base_isolation_compliance_rate_positive_contactgroupmate

            for i in range(len(agent_base_compliance)):
                testing_compliance_symptomatic = (agent_base_compliance <= new_testing_compliance_symptomatic)
                testing_compliance_traced = (agent_base_compliance <= new_testing_compliance_rate_traced)
                testing_compliance_random = (agent_base_compliance <= new_testing_compliance_random)
                tracing_compliance = (agent_base_compliance <= new_tracing_compliance_rate)
                isolation_compliance_symptomatic_individual = (agent_base_compliance <= new_isolation_compliance_rate_symptomatic_individual)
                isolation_compliance_symptomatic_groupmate = (agent_base_compliance <= new_isolation_compliance_rate_symptomatic_groupmate)
                isolation_compliance_positive_individual = (agent_base_compliance <= new_isolation_compliance_rate_positive_individual)
                isolation_compliance_positive_groupmate = (agent_base_compliance <= new_isolation_compliance_rate_positive_groupmate)
                isolation_compliance_positive_contact = (agent_base_compliance <= new_isolation_compliance_rate_positive_contact)
                isolation_compliance_positive_contactgroupmate = (agent_base_compliance <= new_isolation_compliance_rate_positive_contactgroupmate)


            #for agent in range(N):
            #    for edge in model.G.edges(agent):
            #        if model.X[edge[1]] >= 11:  # in isolation
            #            testing_compliance_random[agent] = 1
                    # for edge_of_edge in model.G.edges(edge[1]):
                    #    if model.X[edge_of_edge[1]] >= 11:  # in isolation
                    #       testing_compliance_random[agent] = 1

            #for i in range(N):
            #    if numpy.random.rand() < testing_compliance_random[i]:
            #        testing_compliance_random[i] = 1
            #    else:
            #        testing_compliance_random[i] = 0

            # for agent in range(N):
            #     testing_compliance_random[agent] = 0
            #     for edge in model.G.edges(agent):
            #         if model.X[edge[1]] >= 11:  # in isolation
            #             testing_compliance_random[agent] = 1
            # our addition ends here
            if produce_image:
                record_model(model, model.t, save_folder)
            compliance_list_yes.append(numpy.count_nonzero(testing_compliance_random == 1))
            compliance_list_no.append(numpy.count_nonzero(testing_compliance_random == 0))
            day_list.append(int(model.t))


        running = model.run_iteration(max_dt=max_dt)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Introduce exogenous exposures randomly:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if (int(model.t) != int(timeOfLastIntroduction)):

            timeOfLastIntroduction = model.t

            numNewExposures = numpy.random.poisson(lam=average_introductions_per_day)

            model.introduce_exposures(num_new_exposures=numNewExposures)

            if (numNewExposures > 0):
                print("[NEW EXPOSURE @ t = %.2f (%d exposed)]" % (model.t, numNewExposures))

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Execute testing policy at designated intervals:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        if (int(model.t) != int(timeOfLastIntervention)):

            addition() #run additions here

            cadenceDayNumbers = [int(model.t % cadence_cycle_length)]

            if (backlog_skipped_intervals):
                cadenceDayNumbers = [int(i % cadence_cycle_length) for i in
                                     numpy.arange(start=timeOfLastIntervention, stop=int(model.t), step=1.0)[
                                     1:]] + cadenceDayNumbers

            timeOfLastIntervention = model.t

            for cadenceDayNumber in cadenceDayNumbers:

                currentNumInfected = model.total_num_infected()[model.tidx]
                currentPctInfected = model.total_num_infected()[model.tidx] / model.numNodes

                if (currentPctInfected >= intervention_start_pct_infected and not interventionOn):
                    interventionOn = True
                    interventionStartTime = model.t

                if (interventionOn):

                    print("[INTERVENTIONS @ t = %.2f (%d (%.2f%%) infected)]" % (
                    model.t, currentNumInfected, currentPctInfected * 100))

                    nodeStates = model.X.flatten()
                    nodeTestedStatuses = model.tested.flatten()
                    nodeTestedInCurrentStateStatuses = model.testedInCurrentState.flatten()
                    nodePositiveStatuses = model.positive.flatten()

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # tracingPoolQueue[0] = tracingPoolQueue[0]Queue.pop(0)

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    newIsolationGroup_symptomatic = []
                    newIsolationGroup_contact = []

                    # ----------------------------------------
                    # Isolate SYMPTOMATIC cases without a test:
                    # ----------------------------------------
                    numSelfIsolated_symptoms = 0
                    numSelfIsolated_symptomaticGroupmate = 0

                    if (any(isolation_compliance_symptomatic_individual)):
                        symptomaticNodes = numpy.argwhere((nodeStates == model.I_sym)).flatten()
                        for symptomaticNode in symptomaticNodes:
                            if (isolation_compliance_symptomatic_individual[symptomaticNode]):
                                if (model.X[symptomaticNode] == model.I_sym):
                                    numSelfIsolated_symptoms += 1
                                    newIsolationGroup_symptomatic.append(symptomaticNode)

                                # ----------------------------------------
                                # Isolate the GROUPMATES of this SYMPTOMATIC node without a test:
                                # ----------------------------------------
                                if (isolation_groups is not None and any(isolation_compliance_symptomatic_groupmate)):
                                    isolationGroupmates = next(
                                        (group for group in isolation_groups if symptomaticNode in group), None)
                                    for isolationGroupmate in isolationGroupmates:
                                        if (isolationGroupmate != symptomaticNode):
                                            if (isolation_compliance_symptomatic_groupmate[isolationGroupmate]):
                                                numSelfIsolated_symptomaticGroupmate += 1
                                                newIsolationGroup_symptomatic.append(isolationGroupmate)

                    # ----------------------------------------
                    # Isolate the CONTACTS of detected POSITIVE cases without a test:
                    # ----------------------------------------
                    numSelfIsolated_positiveContact = 0
                    numSelfIsolated_positiveContactGroupmate = 0

                    if (any(isolation_compliance_positive_contact) or any(
                            isolation_compliance_positive_contactgroupmate)):
                        for contactNode in tracingPoolQueue[0]:
                            if (isolation_compliance_positive_contact[contactNode]):
                                newIsolationGroup_contact.append(contactNode)
                                numSelfIsolated_positiveContact += 1

                                # ----------------------------------------
                            # Isolate the GROUPMATES of this self-isolating CONTACT without a test:
                            # ----------------------------------------
                            if (isolation_groups is not None and any(isolation_compliance_positive_contactgroupmate)):
                                isolationGroupmates = next(
                                    (group for group in isolation_groups if contactNode in group), None)
                                for isolationGroupmate in isolationGroupmates:
                                    # if(isolationGroupmate != contactNode):
                                    if (isolation_compliance_positive_contactgroupmate[isolationGroupmate]):
                                        newIsolationGroup_contact.append(isolationGroupmate)
                                        numSelfIsolated_positiveContactGroupmate += 1

                    # ----------------------------------------
                    # Update the nodeStates list after self-isolation updates to model.X:
                    # ----------------------------------------
                    nodeStates = model.X.flatten()

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # ----------------------------------------
                    # Allow SYMPTOMATIC individuals to self-seek tests
                    # regardless of cadence testing days
                    # ----------------------------------------
                    symptomaticSelection = []

                    if (any(testing_compliance_symptomatic)):

                        symptomaticPool = numpy.argwhere((testing_compliance_symptomatic == True)
                                                         & (nodeTestedInCurrentStateStatuses == False)
                                                         & (nodePositiveStatuses == False)
                                                         & ((nodeStates == model.I_sym) | (nodeStates == model.Q_sym))
                                                         ).flatten()

                        numSymptomaticTests = min(len(symptomaticPool), max_symptomatic_tests_per_day)

                        if (len(symptomaticPool) > 0):
                            symptomaticSelection = symptomaticPool[numpy.random.choice(len(symptomaticPool),
                                                                                       min(numSymptomaticTests,
                                                                                           len(symptomaticPool)),
                                                                                       replace=False)]

                    # ----------------------------------------
                    # Test individuals randomly and via contact tracing
                    # on cadence testing days:
                    # ----------------------------------------

                    tracingSelection = []
                    randomSelection = []

                    if (cadenceDayNumber in testingDays):

                        # ----------------------------------------
                        # Apply a designated portion of this day's tests
                        # to individuals identified by CONTACT TRACING:
                        # ----------------------------------------

                        tracingPool = tracingPoolQueue.pop(0)

                        if (any(testing_compliance_traced)):

                            numTracingTests = min(len(tracingPool), min(tests_per_day - len(symptomaticSelection),
                                                                        max_tracing_tests_per_day))

                            for trace in range(numTracingTests):
                                traceNode = tracingPool.pop()
                                if ((nodePositiveStatuses[traceNode] == False)
                                        and (testing_compliance_traced[traceNode] == True)
                                        and (model.X[traceNode] != model.R)
                                        and (model.X[traceNode] != model.Q_R)
                                        and (model.X[traceNode] != model.H)
                                        and (model.X[traceNode] != model.F)):
                                    tracingSelection.append(traceNode)

                        # ----------------------------------------
                        # Apply the remainder of this day's tests to random testing:
                        # ----------------------------------------

                        if (any(testing_compliance_random)):

                            testingPool = numpy.argwhere((testing_compliance_random == True)
                                                         & (nodePositiveStatuses == False)
                                                         & (nodeStates != model.R)
                                                         & (nodeStates != model.Q_R)
                                                         & (nodeStates != model.H)
                                                         & (nodeStates != model.F)
                                                         ).flatten()

                            numRandomTests = max(min(tests_per_day - len(tracingSelection) - len(symptomaticSelection),
                                                     len(testingPool)), 0)

                            testingPool_degrees = model.degree.flatten()[testingPool]
                            testingPool_degreeWeights = numpy.power(testingPool_degrees,
                                                                    random_testing_degree_bias) / numpy.sum(
                                numpy.power(testingPool_degrees, random_testing_degree_bias))

                            if (len(testingPool) > 0):
                                randomSelection = testingPool[
                                    numpy.random.choice(len(testingPool), numRandomTests, p=testingPool_degreeWeights,
                                                        replace=False)]

                    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                    # ----------------------------------------
                    # Perform the tests on the selected individuals:
                    # ----------------------------------------

                    selectedToTest = numpy.concatenate((symptomaticSelection, tracingSelection, randomSelection)).astype(int)

                    numTested = 0
                    numTested_random = 0
                    numTested_tracing = 0
                    numTested_symptomatic = 0
                    numPositive = 0
                    numPositive_random = 0
                    numPositive_tracing = 0
                    numPositive_symptomatic = 0
                    numIsolated_positiveGroupmate = 0

                    newTracingPool = []

                    newIsolationGroup_positive = []

                    for i, testNode in enumerate(selectedToTest):

                        model.set_tested(testNode, True)

                        numTested += 1
                        if (i < len(symptomaticSelection)):
                            numTested_symptomatic += 1
                        elif (i < len(symptomaticSelection) + len(tracingSelection)):
                            numTested_tracing += 1
                        else:
                            numTested_random += 1

                            # If the node to be tested is not infected, then the test is guaranteed negative,
                        # so don't bother going through with doing the test:
                        if (model.X[testNode] == model.S or model.X[testNode] == model.Q_S):
                            pass
                        # Also assume that latent infections are not picked up by tests:
                        elif (model.X[testNode] == model.E or model.X[testNode] == model.Q_E):
                            pass
                        elif (model.X[testNode] == model.I_pre or model.X[testNode] == model.Q_pre
                              or model.X[testNode] == model.I_sym or model.X[testNode] == model.Q_sym
                              or model.X[testNode] == model.I_asym or model.X[testNode] == model.Q_asym):

                            if (test_falseneg_rate == 'temporal'):
                                testNodeState = model.X[testNode][0]
                                testNodeTimeInState = model.timer_state[testNode][0]
                                if (testNodeState in list(temporal_falseneg_rates.keys())):
                                    falseneg_prob = temporal_falseneg_rates[testNodeState][int(min(testNodeTimeInState,
                                                                                                   max(list(
                                                                                                       temporal_falseneg_rates[
                                                                                                           testNodeState].keys()))))]
                                else:
                                    falseneg_prob = 1.00
                            else:
                                falseneg_prob = test_falseneg_rate

                            if (numpy.random.rand() < (1 - falseneg_prob)):
                                # +++++++++++++++++++++++++++++++++++++++++++++
                                # The tested node has returned a positive test
                                # +++++++++++++++++++++++++++++++++++++++++++++
                                numPositive += 1
                                numPositive_list.append(numPositive) #addition here

                                if (i < len(symptomaticSelection)):
                                    numPositive_symptomatic += 1
                                elif (i < len(symptomaticSelection) + len(tracingSelection)):
                                    numPositive_tracing += 1
                                else:
                                    numPositive_random += 1

                                    # Update the node's state to the appropriate detected case state:
                                model.set_positive(testNode, True)

                                # ----------------------------------------
                                # Add this positive node to the isolation group:
                                # ----------------------------------------
                                if (isolation_compliance_positive_individual[testNode]):
                                    newIsolationGroup_positive.append(testNode)

                                # ----------------------------------------
                                # Add the groupmates of this positive node to the isolation group:
                                # ----------------------------------------
                                if (isolation_groups is not None and any(isolation_compliance_positive_groupmate)):
                                    isolationGroupmates = next(
                                        (group for group in isolation_groups if testNode in group), None)
                                    for isolationGroupmate in isolationGroupmates:
                                        if (isolationGroupmate != testNode):
                                            if (isolation_compliance_positive_groupmate[isolationGroupmate]):
                                                numIsolated_positiveGroupmate += 1
                                                newIsolationGroup_positive.append(isolationGroupmate)

                                # ----------------------------------------
                                # Add this node's neighbors to the contact tracing pool:
                                # ----------------------------------------
                                if (any(tracing_compliance) or any(isolation_compliance_positive_contact) or any(
                                        isolation_compliance_positive_contactgroupmate)):
                                    if (tracing_compliance[testNode]):
                                        testNodeContacts = list(model.G[testNode].keys())
                                        numpy.random.shuffle(testNodeContacts)
                                        if (num_contacts_to_trace is None):
                                            numContactsToTrace = int(pct_contacts_to_trace * len(testNodeContacts))
                                        else:
                                            numContactsToTrace = num_contacts_to_trace
                                        newTracingPool.extend(testNodeContacts[0:numContactsToTrace])

                    # Add the nodes to be isolated to the isolation queue:
                    isolationQueue_positive.append(newIsolationGroup_positive)
                    isolationQueue_symptomatic.append(newIsolationGroup_symptomatic)
                    isolationQueue_contact.append(newIsolationGroup_contact)

                    # Add the nodes to be traced to the tracing queue:
                    tracingPoolQueue.append(newTracingPool)

                    print("\t" + str(numTested_symptomatic) + "\ttested due to symptoms  [+ " + str(
                        numPositive_symptomatic) + " positive (%.2f %%) +]" % (
                              numPositive_symptomatic / numTested_symptomatic * 100 if numTested_symptomatic > 0 else 0))
                    print("\t" + str(numTested_tracing) + "\ttested as traces        [+ " + str(
                        numPositive_tracing) + " positive (%.2f %%) +]" % (
                              numPositive_tracing / numTested_tracing * 100 if numTested_tracing > 0 else 0))
                    print("\t" + str(numTested_random) + "\ttested randomly         [+ " + str(
                        numPositive_random) + " positive (%.2f %%) +]" % (
                              numPositive_random / numTested_random * 100 if numTested_random > 0 else 0))
                    print("\t" + str(numTested) + "\ttested TOTAL            [+ " + str(
                        numPositive) + " positive (%.2f %%) +]" % (
                              numPositive / numTested * 100 if numTested > 0 else 0))

                    print("\t" + str(numSelfIsolated_symptoms) + " will isolate due to symptoms         (" + str(
                        numSelfIsolated_symptomaticGroupmate) + " as groupmates of symptomatic)")
                    print("\t" + str(numPositive) + " will isolate due to positive test    (" + str(
                        numIsolated_positiveGroupmate) + " as groupmates of positive)")
                    print("\t" + str(numSelfIsolated_positiveContact) + " will isolate due to positive contact (" + str(
                        numSelfIsolated_positiveContactGroupmate) + " as groupmates of contact)")

                    # ----------------------------------------
                    # Update the status of nodes who are to be isolated:
                    # ----------------------------------------

                    numIsolated = 0

                    isolationGroup_symptomatic = isolationQueue_symptomatic.pop(0)
                    for isolationNode in isolationGroup_symptomatic:
                        model.set_isolation(isolationNode, True)
                        numIsolated += 1

                    isolationGroup_contact = isolationQueue_contact.pop(0)
                    for isolationNode in isolationGroup_contact:
                        model.set_isolation(isolationNode, True)
                        numIsolated += 1

                    isolationGroup_positive = isolationQueue_positive.pop(0)
                    for isolationNode in isolationGroup_positive:
                        model.set_isolation(isolationNode, True)
                        numIsolated += 1

                    print("\t" + str(numIsolated) + " entered isolation")

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        old_t = model.t
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #new addition
    print("days till finish :" + str( int(old_t)+1))
    Susceptible = 0
    for i in range(len(model.G.nodes)):
        if model.X[i] == 1 or model.X[i] == 11:
            Susceptible +=1
    print("percentage of agents that were infected: " + str(int(len(model.G.nodes)-Susceptible*100/len(model.G.nodes))) + "%")

    interventionInterval = (interventionStartTime, model.t)

    # our additions start here again
    yes = numpy.array(compliance_list_yes)
    no = numpy.array(compliance_list_no)
    total = yes + no
    fig = matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(day_list, yes)
    matplotlib.pyplot.plot(day_list, total)
    matplotlib.pyplot.fill_between(day_list, yes, 0, color='green')
    matplotlib.pyplot.fill_between(day_list, yes, total, color='red')
    matplotlib.pyplot.ylabel("agents")
    matplotlib.pyplot.xlabel("day")

    matplotlib.pyplot.savefig(save_folder + "/compliance.png")

    matplotlib.pyplot.show()

    return interventionInterval


def record_model(model, time, save_folder):
    colour_map = []
    label_list = []

    for i in range(len(model.G.nodes)):
        number = model.X[i]

        if number == 1:
            colour_map.append('tab:green')
            label_list.append("Susceptible")
        elif number == 2:
            colour_map.append('orange')
            label_list.append("Exposed")
        elif number == 3 or number == 4 or number == 5:
            colour_map.append('crimson')
            label_list.append("Infectious")
        elif number == 6:
            colour_map.append('tab:pink')
            label_list.append("Hospitalised")
        elif number == 7:
            colour_map.append('tab:blue')
            label_list.append("Recovered")
        elif number == 8:
            colour_map.append('black')
            label_list.append("Fatality")
        elif number == 11 or number == 12 or number == 13 or number == 14 or number == 15 or number == 17:
            colour_map.append('mediumorchid')
            label_list.append("Isolated")
        else:
            colour_map.append('tab:bron')

    networkx.draw(model.G, pos=networkx.kamada_kawai_layout(model.G), node_color=colour_map, node_size=50)

    G = networkx.Graph()
    G.add_node(1)
    networkx.draw(G, node_color='tab:green', label='Susceptible', alpha=None, node_size=50)
    networkx.draw(G, node_color='orange', label='Exposed', alpha=None, node_size=50)
    networkx.draw(G, node_color='crimson', label='Infectious', alpha=None, node_size=50)
    networkx.draw(G, node_color='tab:pink', label='Hospitalised', alpha=None, node_size=50)
    networkx.draw(G, node_color='mediumorchid', label='Isolated', alpha=None, node_size=50)
    networkx.draw(G, node_color='tab:blue', label='Recovered', alpha=None, node_size=50)
    networkx.draw(G, node_color='black', label='Fatality', alpha=None, node_size=50)
    networkx.draw(G, label="Day: " + str(int(time)), alpha=None, node_size=0)
    G = 1

    matplotlib.pyplot.legend(scatterpoints=1)
    matplotlib.pyplot.savefig(save_folder + "/"+str(time) + ".png")
    matplotlib.pyplot.cla()
    # matplotlib.pyplot.show()
    # print(model.G.nodes)
    # print(model.X[0])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
