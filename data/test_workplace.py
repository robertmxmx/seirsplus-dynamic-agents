import numpy.random

from models import *
from networks import *
from sim_loops import *
from utilities import *
import networkx
import os
import datetime
import matplotlib.pyplot as pyplot

for i in range(1):
    NUM_COHORTS              = 5 #faculty
    NUM_NODES_PER_COHORT     = 20 #people
    NUM_TEAMS_PER_COHORT     = 1 #classes

    MEAN_INTRACOHORT_DEGREE  = 6
    PCT_CONTACTS_INTERCOHORT = 0.15
    N = NUM_NODES_PER_COHORT*NUM_COHORTS
    INIT_EXPOSED = 2


    G_baseline, cohorts, teams = generate_workplace_contact_network(
                                     num_cohorts=NUM_COHORTS, num_nodes_per_cohort=NUM_NODES_PER_COHORT,
                                     num_teams_per_cohort=NUM_TEAMS_PER_COHORT,
                                     mean_intracohort_degree=MEAN_INTRACOHORT_DEGREE,
                                     pct_contacts_intercohort=PCT_CONTACTS_INTERCOHORT,
                                     farz_params={'alpha':5.0, 'gamma':5.0, 'beta':0.5, 'r':1, 'q':0.0, 'phi':10,
                                                  'b':0, 'epsilon':1e-6, 'directed': False, 'weighted': False})

    networkx.draw(G_baseline, pos=networkx.spring_layout(G_baseline))
    G_quarantine = networkx.classes.function.create_empty_copy(G_baseline)

    latentPeriod_mean, latentPeriod_coeffvar = 3.0, 0.6
    SIGMA   = 1 / gamma_dist(latentPeriod_mean, latentPeriod_coeffvar, N)

    presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar = 2.2, 0.5
    LAMDA   = 1 / gamma_dist(presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar, N)

    symptomaticPeriod_mean, symptomaticPeriod_coeffvar = 4.0, 0.4
    GAMMA   = 1 / gamma_dist(symptomaticPeriod_mean, symptomaticPeriod_coeffvar, N)

    infectiousPeriod = 1/LAMDA + 1/GAMMA

    onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar = 11.0, 0.45
    ETA     = 1 / gamma_dist(onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar, N)

    hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar = 11.0, 0.45
    GAMMA_H = 1 / gamma_dist(hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar, N)

    hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar = 7.0, 0.45
    MU_H    = 1 / gamma_dist(hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar, N)

    PCT_ASYMPTOMATIC = 0.25
    PCT_HOSPITALIZED = 0.035
    PCT_FATALITY = 0.001

    R0_mean     = 3 # 9.5 is omicron  , 5.4 is delta,
    R0_coeffvar = 0.2
    R0 = gamma_dist(R0_mean, R0_coeffvar, N)

    BETA = 1/infectiousPeriod * R0
    P_GLOBALINTXN = 0.4
    INTERVENTION_START_PCT_INFECTED = 0/100
    AVERAGE_INTRODUCTIONS_PER_DAY   = 0         # expected number of new exogenous exposures per day

    TESTING_CADENCE                 = 'semiweekly'      # how often to do testing (other than self-reporting symptomatics who can get tested any day)
    PCT_TESTED_PER_DAY              = 1.0          # max daily test allotment defined as a percent of population size
    #TEST_FALSENEG_RATE              = 'temporal'    # test false negative rate, will use FN rate that varies with disease time
    TEST_FALSENEG_RATE              = 0.36  #https://www.abc.net.au/news/2022-02-05/staff-children-preschool-childcare-protected-covid-national-plan/100805610
    MAX_PCT_TESTS_FOR_SYMPTOMATICS  = 1.0           # max percent of daily test allotment to use on self-reporting symptomatics
    MAX_PCT_TESTS_FOR_TRACES        = 0.0           # max percent of daily test allotment to use on contact traces
    RANDOM_TESTING_DEGREE_BIAS      = 0             # magnitude of degree bias in random selections for testing, none here

    PCT_CONTACTS_TO_TRACE           = 0.0           # percentage of primary cases' contacts that are traced
    TRACING_LAG                     = 1            # number of cadence testing days between primary tests and tracing tests

    ISOLATION_LAG_SYMPTOMATIC       = 1             # number of days between onset of symptoms and self-isolation of symptomatics
    ISOLATION_LAG_POSITIVE          = 0             # test turn-around time (TAT): number of days between administration of test and isolation of positive cases
    ISOLATION_LAG_CONTACT           = 0             # number of days between a contact being traced and that contact self-isolating

    TESTING_COMPLIANCE_RATE_SYMPTOMATIC                  = 0.5
    TESTING_COMPLIANCE_RATE_TRACED                       = -10
    TESTING_COMPLIANCE_RATE_RANDOM                       = 0.5
    TRACING_COMPLIANCE_RATE                              = -10
    ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL     = -10
    ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE      = -10
    ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL        = 10
    ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE         = -10
    ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT           = -10
    ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE  = -10

    Use_Behavioural_Model_Bool = True
    Use_Structural_Strategic_Bool = False
    Use_Global_Rate_Strategic_Bool = False
    Use_Local_State_Strategic_Bool = False
    Use_Global_State_Strategic_Bool = False


    random_factor_range = 0.1

    model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                                  beta=BETA, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                                  gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H,
                                  a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,
                                  G_Q=G_quarantine, isolation_time=14,
                                  initE=INIT_EXPOSED)

    T = 500
    isolation_groups = list(teams.values())

    bool_produce_image = True

    date_time = str(datetime.datetime.today())
    date_time = date_time.replace(" ","-")
    date_time = date_time.replace(":","-")
    date_time = date_time.split(".")[0]

    save_folder = os.path.dirname(__file__)
    save_folder = save_folder.split("data")[0]
    save_folder += "\output_results\."
    save_folder = save_folder[:-1]
    save_folder += date_time
    if (not os.path.exists(save_folder)):
        os.makedirs(save_folder)
        os.makedirs(save_folder + "\plot_reduced")
        os.makedirs(save_folder + "\plot_normal")

    file = open(save_folder + "\parameters_and_results.txt", "w+")
    file.writelines(["Network Parameters", "\n"])
    file.writelines(["\tNumber of Agents: ", str(NUM_COHORTS*NUM_NODES_PER_COHORT) , "\n"])
    file.writelines(["\tNumber of Cohorts: ", str(NUM_COHORTS) , "\n"])
    file.writelines(["\tNumber of Nodes per Cohort: ", str(NUM_NODES_PER_COHORT) , "\n"])
    file.writelines(["\tNumber of Teams per Cohort: ", str(NUM_TEAMS_PER_COHORT) , "\n"])
    file.writelines(["\tMean Intracohort Degree: ", str(MEAN_INTRACOHORT_DEGREE) , "\n"])
    file.writelines(["\tPercent of Intercohort Edges: ", str(PCT_CONTACTS_INTERCOHORT) , "\n"])
    file.writelines(["\tInitial Number of Agents Exposed: ", str(INIT_EXPOSED) , "\n"])
    file.writelines(["Disease Parameters", "\n"])
    file.writelines(["\tPercentage of Asymtomatic Cases: ", str(PCT_ASYMPTOMATIC) , "\n"])
    file.writelines(["\tPercentage of Hospitalised Cases: ", str(PCT_HOSPITALIZED) , "\n"])
    file.writelines(["\tPercentage of Fatal Cases: ", str(PCT_FATALITY) , "\n"])
    file.writelines(["\tR0 Base Disease Spread Rate: ", str(R0_mean) , "\n"])
    file.writelines(["\tAverage New Introductions Per Day: ", str(AVERAGE_INTRODUCTIONS_PER_DAY) , "\n"])
    file.writelines(["Testing Parameters", "\n"])
    file.writelines(["\tTesting Cadence: ", str(TESTING_CADENCE) , "\n"])
    file.writelines(["\tPercantage of Tests Alloted per Day: ", str(PCT_TESTED_PER_DAY) , "\n"])
    file.writelines(["\tTest False Negative Rate: ", str(TEST_FALSENEG_RATE) , "\n"])
    file.writelines(["\tPercentage of tests offered to Symtomatic: ", str() , "\n"])
    file.writelines(["\tIsolation Lag Symptomatic: ", str(ISOLATION_LAG_SYMPTOMATIC) , "\n"])
    file.writelines(["\tIsolation Lag Days Positive: ", str(ISOLATION_LAG_POSITIVE) , "\n"])
    file.writelines(["Compliance Parameters", "\n"])
    file.writelines(["\tTESTING_COMPLIANCE_RATE_SYMPTOMATIC: ", str(TESTING_COMPLIANCE_RATE_SYMPTOMATIC) , "\n"])
    file.writelines(["\tTESTING_COMPLIANCE_RATE_RANDOM: ", str(TESTING_COMPLIANCE_RATE_RANDOM) , "\n"])
    file.writelines(["\tISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL: ", str(ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL) , "\n"])
    file.writelines(["\tISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE: ", str(ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE) , "\n"])
    file.writelines(["\tISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL: ", str(ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL) , "\n"])
    file.writelines(["\tISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE: ", str(ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE) , "\n"])
    file.close()


    run_tti_sim(model, T,
                intervention_start_pct_infected=INTERVENTION_START_PCT_INFECTED, average_introductions_per_day=AVERAGE_INTRODUCTIONS_PER_DAY,
                testing_cadence=TESTING_CADENCE, pct_tested_per_day=PCT_TESTED_PER_DAY, test_falseneg_rate=TEST_FALSENEG_RATE, max_pct_tests_for_symptomatics=MAX_PCT_TESTS_FOR_SYMPTOMATICS, max_pct_tests_for_traces=MAX_PCT_TESTS_FOR_TRACES,
                random_testing_degree_bias=RANDOM_TESTING_DEGREE_BIAS, pct_contacts_to_trace=PCT_CONTACTS_TO_TRACE, tracing_lag=TRACING_LAG,
                isolation_lag_symptomatic=ISOLATION_LAG_SYMPTOMATIC, isolation_lag_positive=ISOLATION_LAG_POSITIVE,
                isolation_groups=list(teams.values()), base_testing_compliance_rate_symptomatic = TESTING_COMPLIANCE_RATE_SYMPTOMATIC, base_testing_compliance_rate_traced = TESTING_COMPLIANCE_RATE_TRACED, base_testing_compliance_rate_random = TESTING_COMPLIANCE_RATE_RANDOM,
                    base_tracing_compliance_rate = TRACING_COMPLIANCE_RATE, base_isolation_compliance_rate_symptomatic_individual = ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL, base_isolation_compliance_rate_symptomatic_groupmate = ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE, base_isolation_compliance_rate_positive_individual = ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL,
                    base_isolation_compliance_rate_positive_groupmate = ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE, base_isolation_compliance_rate_positive_contact = ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT, base_isolation_compliance_rate_positive_contactgroupmate = ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE, produce_image = bool_produce_image, save_folder = save_folder,
                random_factor_range_behavioural = random_factor_range, Use_Behavioural_Model_bool = Use_Behavioural_Model_Bool,Use_Structural_Strategic_Bool = Use_Structural_Strategic_Bool, Use_Global_Rate_Strategic_Bool = Use_Global_Rate_Strategic_Bool, Use_Global_State_Strategic_Bool = Use_Global_State_Strategic_Bool, Use_Local_State_Strategic_Bool = Use_Local_State_Strategic_Bool)


    #ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE                       = 0  # Assume employee testing is mandatory, so 100% compliance
    #ISOLATION_COMPLIANCE_TEAMMATE0                        = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE)

    #ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE                       = 1  # Assume employee testing is mandatory, so 100% compliance
    #ISOLATION_COMPLIANCE_TEAMMATE1                        = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE)

    #checkpoints = {'t':       [40,80],
    #               'isolation_compliance_positive_groupmate':       [ISOLATION_COMPLIANCE_TEAMMATE0,ISOLATION_COMPLIANCE_TEAMMATE1]}



    results_summary(model)

    #fig, ax = model.figure_infections(vlines=checkpoints['t'],combine_Q_infected=False, plot_Q_R='stacked', plot_Q_S='stacked')
    fig, ax = model.figure_infections(combine_Q_infected=False, plot_Q_R='stacked', plot_Q_S='stacked', save_folder = save_folder, show=bool_produce_image)
