import numpy as np
import time as TIME
import pandas as pd
from scipy.optimize import *

from parameters_export import *
from util import *



##########################################
#                                        #
#           FIT OUTPUT DEBUG             #
#                                        #
##########################################
class Fit_Output_Debug( object ):
    
    ######################################
    def __init__( self, sequence_length) :
        self.prob = np.zeros( sequence_length )             # probability to choose the user action for each coomand to execute in the seqeuence (array of float)
        self.time = 0                                       #ignore
        self.output = Model_Output_Debug( sequence_length ) #ignore

        
##########################################
#                                        #
#      INDIVIDUAL MODEL FITTING          #
#                                        #
##########################################
class Individual_Model_Fitting( object ):

    ######################################
    def __init__( self  ):
        self.command_sequence = []      # sequence of commands
        self.user_output = None         # User_Output> see. util.py.  Time, Success, Action
        self.model = None


    ########################################################################
    # Given the data of a participant,                                     #
    # build an object Output_Debug containing                              #
    # the probablity of the model to choose each action of the participant #
    # INPUT:                                                               #              
    #   - command_sequence ( list< int > ) : sequence of command ids       #
    #   - user_ouput ( User_Output ) : What the user did (action & reward) #
    #                                  See util.py for more details        #
    #   - model ( Model ): The model. See model_interface.py               #
    # OUTPUT:                                                              #
    #   - Fit_Output_Debug : an object containing the probabilities        #
    #                        of each action (see above).                   #
    #                        i.e. res.prob[i] : probability to choose the  #
    #                        user action at the step i                     #
    ########################################################################
    def run_debug( self ):
        start = TIME.time()
        res = Fit_Output_Debug( len( self.command_sequence ) )

        # TODO 3.a ( 4 lines of code )
        # For each command command of the sequence, estimate the probabily
        # of the model to choose the user action. Update 
        # 
        # Methods: action_prob() and update_model() of Model (model_interface.py)
        # Data structures: StepResult and Action from util.py
        #  
        user_output = User_Output()
        for i,(cmd, time, success, action) in enumerate(zip(self.command_sequence, self.user_output.time, self.user_output.success, self.user_output.action)):
            res.prob[i] = self.model.action_prob(cmd, action)
            result = StepResult(cmd, action, time, success)
            self.model.update_model(result)
        
        res.time = TIME.time() - start
        return res



##########################################
#                                        #
#             MODEL FITTING              #
#                                        #
##########################################
class Model_Fitting( object ):

    ######################################
    def __init__( self, debug = False ):
        self.command_ids   = []                     # Type list < int > : 
        self.model_vec     = []                     # Type list < Model_Interface >
        self.user_data_vec = []                     # Type list < User_Data >
        self.parameters    = pd.DataFrame()                   # Type Dataframe
        self.method = Individual_Model_Fitting()    
        self.debug = debug
        self.debug_var = 0
        
        
    ######################################
    def optimize( self ):
        self.is_valid()
        timestamp    = TIME.strftime("%Y-%m-%d-%H-%M-%S", TIME.gmtime() )
        result_vec   = []
        user_id_vec  = [ user_data.id for user_data in self.user_data_vec ]
        for model in self.model_vec:
            model.params = Parameters( model.name, model.default_parameters_path() )
            model_result = Model_Result.create( model.name, np.array( user_id_vec ), self.debug )
            model_result.n_parameters = model.params.n( Freedom.USER_FREE )   
            for i , user_data in enumerate( self.user_data_vec ):
                #PROBABLY A BUG HERE TO UPDATE
                model_result.n_observations[ i ] = len( user_data.cmd )
                model_result.technique[ i ]      = user_data.technique_name
                start = TIME.time()
                available_strategies = strategies_from_technique( user_data.technique_name )
                
                self.method.model       = model
                self.method.command_sequence  = user_data.cmd
                self.method.user_output = user_data.output

                params = model.get_params()
                free_param_name_vec = []                    # name of the parameters to optimize
                free_param_bnds_vec = []                    # Bounds for parameter values list( [min, max] )
                for param in params.values(): 
                    if param.freedom == Freedom.USER_FREE :
                        free_param_name_vec.append( param.name )
                        free_param_bnds_vec.append( [ param.min, param.max ] )
                self.debug_var = 1000000000


                #TODO 4.a (1 line of code)
                # use the method differential_evolution from scipy.optimize
                # differential_evolution has several paramaters. We will use: 
                # - func   = self.to_minimize (the function to minimize)
                # - bounds =  free_param_bnds_vec (the bounds ofthe model parameters)
                # - args   = (free_param_name_vec, self.method, available_strategies )
                #               i.e. the parameters of the method to minimize (self.to_minimize() )
                res = differential_evolution(self.to_minimize, free_param_bnds_vec, (free_param_name_vec, self.method, available_strategies))



                end = TIME.time()
                print("optmize the model: ", model.name, "on user: ", user_data.id, "in ",  end - start,"s")
                print( res )

                parameters = Parameters( model.name, model.default_parameters_path() )
                for name, value in zip( free_param_name_vec, res.x ):
                    parameters[ name ].value = value
                

                model_result.log_likelihood[ i ] = - res.fun
                model_result.time[ i ]           = end - start
                model_result.parameters[ i ]     = parameters
                self.backup_parameters( model_result, timestamp ) 
            result_vec.append( model_result )
        
        return result_vec


    ######################################
    # Differential evaluation requires a function to minimize
    # as log_likelihood( goodness_of_fit.prob ) is negative, it returns  the inverse
    # INPUT:
    #   - param_value: the values of the parameters (chosen by the optimization method)
    #   - param_name : the name of the parameters
    #   - method     : the fitting method
    #   - available_strategies: only useful for one specific technique (ignore)   
    # OUTPUT:
    #   - the inverse of the likelihood
    ######################################
    def to_minimize( self, param_value, param_name, method, available_strategies ):
        # assign the novel values of the freeparameters
        for name, value in zip( param_name, param_value ):
             method.model.params[ name ].value = value
        method.model.reset( self.command_ids, available_strategies )
        goodness_of_fit = method.run_debug()
        if self.debug_var > - log_likelihood( goodness_of_fit.prob ) :
            self.debug_var = - log_likelihood( goodness_of_fit.prob )
            print( self.debug_var )
        return - log_likelihood( goodness_of_fit.prob )

    
    
    ######################################
    def run( self ):
        self.is_valid()
        result = []                                 # Type Model_Result
        for model in self.model_vec:

            self.method.model = model
            user_id_vec  = [ user_data.id for user_data in self.user_data_vec ]
            model_result = Model_Result.create( model.name, np.array( user_id_vec ), self.debug )
            model_result.n_parameters = model.params.n( Freedom.USER_FREE )
            start = TIME.time()
            
            for i , user_data in enumerate( self.user_data_vec ):
                model_result.n_observations[ i ] = len( user_data.cmd )
                model_result.technique[ i ] = user_data.technique_name
                if not self.parameters.empty :
                    params = parameters_from_df( self.parameters, model.name, user_data.id )
                    model.params = params
                model.reset( self.command_ids, strategies_from_technique( user_data.technique_name ) )
                self.method.model = model
                self.method.command_sequence  = user_data.cmd
                self.method.user_output = user_data.output
                
                goodness_of_fit = None
                if self.debug :
                    goodness_of_fit = self.method.run_debug()
                    model_result.output[ i ] =  goodness_of_fit.output
                    model_result.prob[ i ]   =  goodness_of_fit.prob
                else :
                    goodness_of_fit = self.method.run()

                model_result.log_likelihood[ i ] = log_likelihood( goodness_of_fit.prob )
                model_result.time[ i ] = goodness_of_fit.time
            
            model_result.whole_time = TIME.time() - start
            result.append( model_result )
        
        return result 

    ######################################
    def backup_parameters( self, parameters, timestamp ):
        path = "./backup/" + timestamp + "/"
        #filename = parameters.name + "_model_"+ str( user_id ) + ".csv"
        Parameters_Export.write( [parameters], path )

    ######################################
    def is_valid( self ):
        res = True
        if len( self.command_ids ) == 0 :
            res = False
            raise ValueError(" Model_Fit: command_ids is empty ")

        if len( self.model_vec ) == 0 :
            res = False
            raise ValueError(" Model_Fit: model_vec is empty ")

        if len( self.user_data_vec ) == 0 : 
            res = False
            raise ValueError(" Model_Fit: user_data_vec is empty ")
        if not res:
            exit(0)
        return True  










