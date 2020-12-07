#import numpy as np
import pandas as pd
import time as TIME

from util import *
from dataframe_util import *

##########################################
#                                        #
#             MODEL SIMULATION           #
#                                        #
##########################################
class Model_Simulation( object ):

    ######################################
    def __init__( self, debug = False ):
        self.command_ids   = []                     # Type int
        self.model_vec     = []                     # Type Model_Interface
        self.user_data_vec = []                     # Type User_Data
        self.parameters    = pd.DataFrame()         # DataFrame
        self.debug  = debug
        self.debug_var = 0
    
    ##########################################################################
    # Given a sequence of commands, the model produces a sequence of actions #
    # of the same length
    # INPUT                                                                  #
    #   - model ( Model ) : model (see model_interface.py)                   #
    #   - cmd_sequence ( list< int > ) : sequence of command ids             #
    # OUTPUT                                                                 #
    #   - agent_output ( User_Ouput ) : what the agent produces. See util.py #
    #   - model_output ( Model_Output ): Ignore                              #
    ##########################################################################
    def simulate( self, model, cmd_sequence ):
        agent_output = User_Output(  len( cmd_sequence ) )         # Multi sequence : Time, Success, Action
        model_output = Model_Output( len( cmd_sequence ) )
        start = TIME.time()

        # TOOD 5.a (7 lines of code)
        # use the methods select_strategy, generate_step and update_model from Model (model_interface.py)
        # cautious! select_strategy returns a list. Only considers the first element
        # for ...
        #   ...
        #   ...
        #   ...
        #   agent_output.time[i]    = ...
        #   agent_output.success[i] = ...
        #   agent_output.action[i]  = ...

        for i, cmd in enumerate(cmd_sequence):
            strategy = model.select_strategy(cmd)[0]
            result = model.generate_step(cmd, Action(cmd, strategy))
            
            model.update_model(result)
           
            agent_output.time[i] = result.time
            agent_output.success[i] = result.success
            agent_output.action[i] = result.action

        t = TIME.time() - start
        return agent_output, model_output, t



    ######################################
    #
    # Output: vector< Simulation_Result >
    # Simulation_Result {name (string), user_id (int), input (list<int>  of command ids), output ( list<Output> ), prob ( list<Model_Output> ), time (float) }
    #
    ######################################
    def run( self, parameters = None ):
        result = []                               # Type Model_Result
        for model in self.model_vec:
            start = TIME.time()
            for i , user_data in enumerate( self.user_data_vec ):
                simulation_result                = Simulation_Result( model.name )
                simulation_result.user_data      = user_data
                if not self.parameters.empty :
                    params = parameters_from_df( self.parameters, model.name, user_data.id )
                    #print( "simulation run", model, user_data.id,  params.values_str() )
                    model.params = params
                model.reset( self.command_ids, strategies_from_technique( user_data.technique_name ) )
                agent_output, model_output, time = self.simulate( model, user_data.cmd )
                simulation_result.output       = agent_output
                simulation_result.prob         = model_output
                simulation_result.time         = time

                result.append( simulation_result )
                    
        return result 


