import random
import numpy as np
import pandas as pd
import math


#######################
def encode_cmd_s( cmd, s ) :
    return 3 * cmd + s


########################
def values_long_format(actions, values):
    res = np.zeros(3)
    for i in range( 0, len(actions)):
        res[ actions[i].strategy ] = values[i]
    return res

#########################
def zipfian(s, N):
    res = np.zeros(N)
    denum = 0
    for n in range(N):
        denum += 1/ ( (n+1)**s )

    for k in range(N):
        num = 1/( (k+1)**s)
        res [k] = num / denum 
    return res


########################
def soft_max(beta, vec):
    vec = vec * beta
    return np.exp( vec ) / np.sum( np.exp( vec ), axis=0)



########################################################################
# Estimate the log likelihood  given a vector of probabilities         #
# Input:                                                               #
#    - p: an array of probabilities (to choose an action at trial t )  #
# Ouptput:                                                             #
#    -  a float: \sum_t ( log( P_t ) )                                 #
########################################################################
def log_likelihood( prob_vec ):
    return np.sum(np.log(prob_vec))
    


########################################################################
# Estimate the BIC Score                                               #
# Input:                                                               #
#    - n_observations (int): number of observations to predict         #
#    - k_parameters   (int): number of parameters of the model         #
#    - likelihood   (float): likelihood                                #
# Ouptput:                                                             #
#    -  log(n) * k - 2 likelihood                                      #
########################################################################
def bic_score( n_observations, k_parameters, likelihood ):
    a = np.log(n_observations)
    return np.log(n_observations) * k_parameters - 2 * likelihood




###################################
def strategies_from_technique( technique_name ):
    if technique_name == "disable":
        return np.array( [Strategy.HOTKEY, Strategy.LEARNING], dtype=int )
    else:
        return np.array( [Strategy.MENU, Strategy.HOTKEY, Strategy.LEARNING], dtype=int )



#######################
def compound_soft_max(beta_1, vec_1, beta_2, vec_2):
    vec_1 = vec_1 * beta_1
    vec_2 = vec_2 * beta_2     
    return np.exp( vec_1 + vec_2 ) / np.sum( np.exp( vec_1 + vec_2 ), axis=0)


def extended_soft_max(beta_vec, value_vec):
    vec = np.zeros( len(value_vec[0]) )
    for i in range(0, len(value_vec)):
        vec += beta_vec[i] * value_vec[i]
    return np.exp( vec ) / np.sum( np.exp( vec ), axis = 0)




###########################
#   Command
###########################
class Command(object):
    
    def __init__(self, id=0, name="None", hotkey ="hk"):
        self.id     = id
        self.name   = name
        self.hotkey = hotkey


###########################################################################
#       STEP RESULT                                                       #
# INPUT:                                                                  #
# - cmd (int) : command                                                   #
# - action (Action) : action (see util.py)                                #
# - time (float)    : time to execute the command with the given strategy #
# - error (boolean) : Does the use executed the correct command           #
###########################################################################
class StepResult(object):

    def __init__(self, cmd = -1, action = None, time = -1, success = True):
        self.cmd     = cmd
        self.action  = action
        self.time    = time
        self.success = success



###########################
#                         #
#    SIMULATION RESULT    #
#                         #
###########################
class Simulation_Result( object ):
    def __init__( self, name = "" ):
        self.name           = name     # str
        self.user_data      = None     # User_Data
        # self.user_id        = -1       # int
        # self.technique_name = ""       # str
        # self.input          = None     # list< int > cmd ids
        self.output         = None     # User_Output        
        self.prob           = None     # Model_Output
        self.whole_time     = 0        # float


###########################
#                         #
#       MODEL RESULT      #
#                         #
###########################
class Model_Result( object ):

    ############################
    def __init__( self, name = "" ):
        self.name           = name
        self.user_id        = np.array( [], dtype=int)
        self.technique      = np.array( [] )
        self.log_likelihood = np.array( [] )
        self.prob           = []    # proability to peform the user action
        self.output         = []    # output: {menu, hotkey, learning } vector<float>, probability to perform the conrresponding strategy
        self.time           = []    #time of the simulation
        self.parameters     = []
        self.whole_time     = 0
        self.n_observations = np.array([], dtype=int)
        self.n_parameters   = 0 

    ###################################
    def create( model_name, user_id_vec, debug = False) :
        model_result                = Model_Result( model_name )
        model_result.user_id        = user_id_vec
        model_result.technique      = np.array( [ None ] * len( user_id_vec) ) if debug else []
        model_result.time           = np.zeros( len(user_id_vec ) ) 
        model_result.output         = np.array( [ None ] * len( user_id_vec) ) if debug else []
        model_result.prob           = np.array( [ None ] * len( user_id_vec) ) if debug else []
        model_result.parameters     = np.array( [ None ] * len( user_id_vec) ) 
        model_result.log_likelihood = np.zeros( len(user_id_vec ) )
        model_result.n_observations = np.zeros( len(user_id_vec ), dtype=int )
        model_result.n_parameters   = 0


        return model_result


###########################
#                         #
#       MODEL OUTPUT      #
#                         #
###########################
class Model_Output( object ):

    ##################################
    def __init__( self, n = 0 ):
        self.menu      = np.zeros( n )
        self.hotkey    = np.zeros( n )
        self.learning  = np.zeros( n )

    ##################################
    def n( self ) :
        return len( self.action )


###########################
#                         #
#  MODEL OUTPUT DEBUG     #
#                         #
###########################
class Model_Output_Debug( Model_Output ):
    def __init__( self, n = 0 ):
        super().__init__( n )
        self.meta_info_1 = np.zeros( n )
        self.meta_info_2 = np.zeros( n )

##################################################################################
#                                                                                #
#        USER OUTPUT                                                             #
#  Data structure containing what the user id at each trial (step)               #
#  time (float), success (bool), action (Action)                                 #
# INPUT                                                                          #
# - n (int):    the number of trials (steps), i.e. number of executed commands   #
##################################################################################
class User_Output( object ):

    ##################################
    def __init__( self, n = 0 ):
        self.time    = np.zeros( n )
        self.success = np.zeros( n, dtype=bool )
        self.action  = np.array( [ None ] * n ) 

    def n( self ) :
        return len( self.action )


###########################
#                         #
#      COMMAND INFOS      #
#                         #
###########################
class Command_Info( object ):

    ##################################
    def __init__( self ):
        self.id               = np.array([], dtype=int)
        self.name             = np.array([], dtype=str)
        self.frequency        = np.array([], dtype=int)
        self.start_transition = np.array([], dtype=int)
        self.stop_transition  = np.array([], dtype=int)



###########################
#                         #
#   EXPERIMENT OTHER      #
#                         #
###########################
class Experiment_Other( object ):

    ##################################
    def __init__( self ):
        self.block       = np.array([], dtype=int)
        self.block_trial = np.array([], dtype=int)
        self.encounter   = np.array([], dtype=int)
        self.method_id   = np.array([], dtype=int)
        self.method_name = np.array([], dtype=str)



###########################
#                         #
#        USER DATA        #
#                         #
###########################
class User_Data( object ):

    ##################################
    def __init__( self ):
        self.id = -1
        self.technique_id   = -1
        self.technique_name = "empty"
        self.command_info = Command_Info()
        self.output = User_Output()
        self.cmd = np.array([],dtype=int)

        self.other = Experiment_Other()

    ##################################
    def set( self, id, technique_id , technique_name ):
        self.id = id
        self.technique_name = technique_name
        self.technique_id = technique_id

    ##################################
    def trial_info( self, trial_id ):
        info = dict()
        if trial_id <0 or trial_id > len( self.cmd ) -1 :
            return info
        
        #cmd_index = self.command_info.id.index( self.cmd[ trial_id ] )
        cmd_index = np.where( self.command_info.id == self.cmd[ trial_id ] )[ 0 ][ 0 ]
        print("cmd index ..............", cmd_index )
        info["User id"]          = str( self.id )
        info["Technique"]        = self.technique_name
        info["Command"]          = str( self.cmd[ trial_id ] )
        info["Frequence" ]       = str( self.command_info.frequency[ cmd_index ] )
        info["Name"]             = self.command_info.name[ cmd_index ]
        info["Transition start"] = str( self.command_info.start_transition[ cmd_index ] )
        info["Transition stop"]  = str( self.command_info.stop_transition[ cmd_index ] )
        info["Time"]             = str( self.output.time[ trial_id ] )
        info["Success"]          = str( self.output.success[ trial_id ] )
        info["Strategy"]         = str( self.output.action[ trial_id ].strategy )
        info["Executed command"] = str( self.output.action[ trial_id ].cmd )
        info["block"]            = str( self.other.block[ trial_id ] )
        info["block trial"]      = str( self.other.block_trial[ trial_id ] )
        info["encounter"]        = str( self.other.encounter[ trial_id ] )
        info["method id"]        = str( self.other.method_id[ trial_id ] )
        info["method name"]      = str( self.other.method_name[ trial_id ] )
        return info
        




###########################
#                         #
#        FREEDOM          #
#                         #
###########################
class Freedom(object):
    FIXED = 0
    USER_FREE = 1
    TECHNIQUE_FREE = 2
    EXPERIMENT_FREE = 3

###########################
#                         #
#        Strategy         #
#                         #
###########################
class Strategy(object):
    MENU = 0
    HOTKEY = 1
    LEARNING = 2

class Strategy_Filter(Strategy):
    NONE = -1
    ALL = 10

###########################
#                         #
#          ACTION         #
#                         #
###########################
class Action():
#    def __init__(self, bin_number):
#        self.bin_number = bin_number

    def __init__(self, cmd, strategy):
        self.cmd = cmd
        self.strategy = strategy






