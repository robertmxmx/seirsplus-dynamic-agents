from models import *
from networks import *
from sim_loops import *
from utilities import *
import networkx
import matplotlib.pyplot as pyplot

NUM_COHORTS              = 10
NUM_NODES_PER_COHORT     = 30
NUM_TEAMS_PER_COHORT     = 2

MEAN_INTRACOHORT_DEGREE  = 6
PCT_CONTACTS_INTERCOHORT = 0.1
N = NUM_NODES_PER_COHORT*NUM_COHORTS
INIT_EXPOSED = 3

G_baseline, cohorts, teams = generate_workplace_contact_network(
                                 num_cohorts=NUM_COHORTS, num_nodes_per_cohort=NUM_NODES_PER_COHORT,
                                 num_teams_per_cohort=NUM_TEAMS_PER_COHORT,
                                 mean_intracohort_degree=MEAN_INTRACOHORT_DEGREE,
                                 pct_contacts_intercohort=PCT_CONTACTS_INTERCOHORT,
                                 farz_params={'alpha':5.0, 'gamma':5.0, 'beta':0.5, 'r':1, 'q':0.0, 'phi':10,
                                              'b':0, 'epsilon':1e-6, 'directed': False, 'weighted': False})

#network_info(G_baseline, "Baseline", plot=True)
networkx.draw(G_baseline, pos=networkx.kamada_kawai_layout(G_baseline))
G_quarantine = networkx.classes.function.create_empty_copy(G_baseline)


latentPeriod_mean, latentPeriod_coeffvar = 3.0, 0.6
SIGMA   = 1 / gamma_dist(latentPeriod_mean, latentPeriod_coeffvar, N)

presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar = 2.2, 0.5
LAMDA   = 1 / gamma_dist(presymptomaticPeriod_mean, presymptomaticPeriod_coeffvar, N)

#dist_info([1/LAMDA, 1/SIGMA, 1/LAMDA+1/SIGMA], ["latent period", "pre-symptomatic period", "total incubation period"], plot=True, colors=['gold', 'darkorange', 'black'], reverse_plot=True)

symptomaticPeriod_mean, symptomaticPeriod_coeffvar = 4.0, 0.4
GAMMA   = 1 / gamma_dist(symptomaticPeriod_mean, symptomaticPeriod_coeffvar, N)

infectiousPeriod = 1/LAMDA + 1/GAMMA

#dist_info([1/LAMDA, 1/GAMMA, 1/LAMDA+1/GAMMA], ["pre-symptomatic period", "(a)symptomatic period", "total infectious period"], plot=True, colors=['darkorange', 'crimson', 'black'], reverse_plot=True)

onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar = 11.0, 0.45
ETA     = 1 / gamma_dist(onsetToHospitalizationPeriod_mean, onsetToHospitalizationPeriod_coeffvar, N)

hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar = 11.0, 0.45
GAMMA_H = 1 / gamma_dist(hospitalizationToDischargePeriod_mean, hospitalizationToDischargePeriod_coeffvar, N)

#dist_info([1/ETA, 1/GAMMA_H, 1/ETA+1/GAMMA_H], ["onset-to-hospitalization period", "hospitalization-to-discharge period", "onset-to-discharge period"], plot=True, colors=['crimson', 'violet', 'black'], reverse_plot=True)

hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar = 7.0, 0.45
MU_H    = 1 / gamma_dist(hospitalizationToDeathPeriod_mean, hospitalizationToDeathPeriod_coeffvar, N)

#dist_info([1/ETA, 1/MU_H, 1/ETA+1/MU_H], ["onset-to-hospitalization period", "hospitalization-to-death period", "onset-to-death period"], plot=True, colors=['crimson', 'darkgray', 'black'], reverse_plot=True)

PCT_ASYMPTOMATIC = 0.25
PCT_HOSPITALIZED = 0.035
PCT_FATALITY = 0.08

R0_mean     = 3.0
R0_coeffvar = 0.2

R0 = gamma_dist(R0_mean, R0_coeffvar, N)

#dist_info(R0, "Individual R0", bin_size=0.1, plot=True, colors='crimson')

BETA = 1/infectiousPeriod * R0

P_GLOBALINTXN = 0.4

INTERVENTION_START_PCT_INFECTED = 0/100
AVERAGE_INTRODUCTIONS_PER_DAY   = 0         # expected number of new exogenous exposures per day

TESTING_CADENCE                 = 'workday'      # how often to do testing (other than self-reporting symptomatics who can get tested any day)
PCT_TESTED_PER_DAY              = 1.0           # max daily test allotment defined as a percent of population size
TEST_FALSENEG_RATE              = 'temporal'    # test false negative rate, will use FN rate that varies with disease time
MAX_PCT_TESTS_FOR_SYMPTOMATICS  = 1.0           # max percent of daily test allotment to use on self-reporting symptomatics
MAX_PCT_TESTS_FOR_TRACES        = 0.0           # max percent of daily test allotment to use on contact traces
RANDOM_TESTING_DEGREE_BIAS      = 0             # magnitude of degree bias in random selections for testing, none here

PCT_CONTACTS_TO_TRACE           = 0.0           # percentage of primary cases' contacts that are traced
TRACING_LAG                     = 2             # number of cadence testing days between primary tests and tracing tests

ISOLATION_LAG_SYMPTOMATIC       = 1             # number of days between onset of symptoms and self-isolation of symptomatics
ISOLATION_LAG_POSITIVE          = 2             # test turn-around time (TAT): number of days between administration of test and isolation of positive cases
ISOLATION_LAG_CONTACT           = 0             # number of days between a contact being traced and that contact self-isolating

TESTING_COMPLIANCE_RATE_SYMPTOMATIC                  = 0.3
TESTING_COMPLIANCE_RATE_TRACED                       = 0.0
TESTING_COMPLIANCE_RATE_RANDOM                       = 0.8  # Assume employee testing is mandatory, so 100% compliance
TRACING_COMPLIANCE_RATE                              = 0.0
ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL     = 0.0
ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE      = 0.0
ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL        = 1
ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE         = 0.0  # Isolate teams with a positive member, but suppose 20% of employees are essential workforce
ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT           = 0.0
ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE  = 0.0

#TESTING_COMPLIANCE_SYMPTOMATIC                   = (numpy.random.rand(N) < TESTING_COMPLIANCE_RATE_SYMPTOMATIC)


model = ExtSEIRSNetworkModel(G=G_baseline, p=P_GLOBALINTXN,
                              beta=BETA, sigma=SIGMA, lamda=LAMDA, gamma=GAMMA,
                              gamma_asym=GAMMA, eta=ETA, gamma_H=GAMMA_H, mu_H=MU_H,
                              a=PCT_ASYMPTOMATIC, h=PCT_HOSPITALIZED, f=PCT_FATALITY,
                              G_Q=G_quarantine, isolation_time=14,
                              initE=INIT_EXPOSED)

T = 500
isolation_groups = list(teams.values())

run_tti_sim(model, T,
            intervention_start_pct_infected=INTERVENTION_START_PCT_INFECTED, average_introductions_per_day=AVERAGE_INTRODUCTIONS_PER_DAY,
            testing_cadence=TESTING_CADENCE, pct_tested_per_day=PCT_TESTED_PER_DAY, test_falseneg_rate=TEST_FALSENEG_RATE, max_pct_tests_for_symptomatics=MAX_PCT_TESTS_FOR_SYMPTOMATICS, max_pct_tests_for_traces=MAX_PCT_TESTS_FOR_TRACES,
            random_testing_degree_bias=RANDOM_TESTING_DEGREE_BIAS, pct_contacts_to_trace=PCT_CONTACTS_TO_TRACE, tracing_lag=TRACING_LAG,
            isolation_lag_symptomatic=ISOLATION_LAG_SYMPTOMATIC, isolation_lag_positive=ISOLATION_LAG_POSITIVE,
            isolation_groups=list(teams.values()), base_testing_compliance_rate_symptomatic = TESTING_COMPLIANCE_RATE_SYMPTOMATIC, base_testing_compliance_rate_traced = TESTING_COMPLIANCE_RATE_TRACED, base_testing_compliance_rate_random = TESTING_COMPLIANCE_RATE_RANDOM,
                base_tracing_compliance_rate = TRACING_COMPLIANCE_RATE, base_isolation_compliance_rate_symptomatic_individual = ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_INDIVIDUAL, base_isolation_compliance_rate_symptomatic_groupmate = ISOLATION_COMPLIANCE_RATE_SYMPTOMATIC_GROUPMATE, base_isolation_compliance_rate_positive_individual = ISOLATION_COMPLIANCE_RATE_POSITIVE_INDIVIDUAL,
                base_isolation_compliance_rate_positive_groupmate = ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE, base_isolation_compliance_rate_positive_contact = ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACT, base_isolation_compliance_rate_positive_contactgroupmate = ISOLATION_COMPLIANCE_RATE_POSITIVE_CONTACTGROUPMATE)


#ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE                       = 0  # Assume employee testing is mandatory, so 100% compliance
#ISOLATION_COMPLIANCE_TEAMMATE0                        = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE)

#ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE                       = 1  # Assume employee testing is mandatory, so 100% compliance
#ISOLATION_COMPLIANCE_TEAMMATE1                        = (numpy.random.rand(N) < ISOLATION_COMPLIANCE_RATE_POSITIVE_GROUPMATE)

#checkpoints = {'t':       [40,80],
#               'isolation_compliance_positive_groupmate':       [ISOLATION_COMPLIANCE_TEAMMATE0,ISOLATION_COMPLIANCE_TEAMMATE1]}



results_summary(model)

#fig, ax = model.figure_infections(vlines=checkpoints['t'],combine_Q_infected=False, plot_Q_R='stacked', plot_Q_S='stacked')
fig, ax = model.figure_infections(combine_Q_infected=False, plot_Q_R='stacked', plot_Q_S='stacked')
