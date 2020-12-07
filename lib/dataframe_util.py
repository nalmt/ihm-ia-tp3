
import pandas as pd
import numpy as np
from util import *
from parameters import Parameter, Parameters

###########################
def model_res_vec_to_data_frame( model_result_vec ):
    columns = np.array( ['model', 'user_id', 'technique', 'parameter', 'value', 'freedom', 'log_likelihood', 'n_observations', 'n_parameters', 'BIC'] )
    
    df = pd.DataFrame( columns = columns )    

    for model_result in model_result_vec :

        for i, user_id in enumerate( model_result.user_id ) :
            row = dict()
            row[ 'model' ]          = model_result.name
            row[ 'n_parameters']    = model_result.n_parameters
            row[ 'user_id' ]        = user_id
            row[ 'technique']       = model_result.technique[i]
            row[ 'log_likelihood' ] = round( model_result.log_likelihood[ i ], 2)
            row[ 'n_observations' ] = model_result.n_observations[ i ]
            row [ 'BIC' ]           = bic_score( row[ 'n_observations' ], row[ 'n_parameters'], row[ 'log_likelihood' ] )
            if model_result.parameters[i]  :
                for parameter in model_result.parameters[ i ].values() :
                    row[ 'parameter' ] = parameter.name
                    row[ 'value'     ] = round( parameter.value, 5)
                    row[ 'freedom'   ] = parameter.freedom
                    df = df.append( row, ignore_index = True )
            else:
                row[ 'parameter' ] = 'none'
                row[ 'value'     ] = 0
                row[ 'freedom'   ] = -1
                df = df.append( row, ignore_index = True )

    return df



############################
def parameters_from_df( df, model, user_id ):
    df = df[ ( df.model == model ) & ( df.user_id == user_id ) ]
    parameters = Parameters( name = model )
    n_row = df.shape[0]
    for i in range( 0, n_row ):
        parameter = Parameter()
        parameter.name    = df.at[i, 'parameter']
        parameter.value   = df.at[i, 'value'] #float( df.at[i, 'value'] ) if '.' in df.at[i, 'value'] else int( df.at[i, 'value'] )
        # parameter.min     = float( df[i, 'value'] ) if '.' in row[2] else int( row[2] )
        # parameter.max     = float( row[3] ) if '.' in row[3] else int( row[3] )
        # parameter.step    = float( row[4] ) if '.' in row[4] else int( row[4] )
        parameter.freedom = int(  df.at[i, 'freedom'] )
        # parameter.comment = row[ 6 ]
        parameters[ parameter.name ] = parameter
                
    return parameters


############################
def user_data_vec_to_data_frame( user_data_vec ):
    df = pd.DataFrame()
    for user_data in user_data_vec:
        df_user = user_data_to_data_frame( user_data )
        if  df.empty:
            df = df_user
        else:
            df = pd.concat( [df, df_user] )
    return df




############################
def user_data_to_data_frame( user_data ):
    df = pd.DataFrame({    'model'          : 'Observations',
                           'user_id'        : user_data.id,
                           'technique_name' : user_data.technique_name,
                           'block_id'       : user_data.other.block,
                           'trial_id'       : np.arange(0, len( user_data.output.action ) ),
                           'cmd_input'      : user_data.cmd,
                           'time'           : user_data.output.time,
                           'success'        : user_data.output.success,
                           'strategy'       : np.array( [a.strategy for a in user_data.output.action] ),
                           'cmd_output'     : np.array( [a.cmd for a in user_data.output.action] ),
                           'start_transition': 0,
                           'stop_transition'  : 0 })

    cmd_vec = df.cmd_input.unique()
    for cmd in cmd_vec :
        start_transition = user_data.command_info.start_transition[ np.where( user_data.command_info.id == cmd )[ 0 ][ 0 ] ]
        stop_transition   = user_data.command_info.stop_transition  [ np.where( user_data.command_info.id == cmd )[ 0 ][ 0 ] ]
        df.loc[ df.cmd_input == cmd , 'start_transition' ] = start_transition
        df.loc[ df.cmd_input == cmd , 'stop_transition'  ] = stop_transition
        
    return df


############################
def simulation_vec_to_data_frame( model_simulation_vec ):
    df = pd.DataFrame()
    for model_simulation in model_simulation_vec:
        df_model = simulation_to_data_frame( model_simulation )
        if  df.empty:
            df = df_model
        else:
            df = pd.concat( [df, df_model] )
    return df

############################
def simulation_to_data_frame( model_simulation ):
    df = pd.DataFrame({    'model'          : model_simulation.name,
                           'user_id'        : model_simulation.user_data.id,
                           'technique_name' : model_simulation.user_data.technique_name,
                           'block_id'       : model_simulation.user_data.other.block,
                           'trial_id'       : np.arange(0, len( model_simulation.user_data.output.action ) ),
                           'cmd_input'      : model_simulation.user_data.cmd,
                           'menu_prob'      : model_simulation.prob.menu,
                           'hotkey_prob'    : model_simulation.prob.hotkey,
                           'learning_prob'  : model_simulation.prob.learning,                           
                           'time'           : model_simulation.output.time,
                           'success'        : model_simulation.output.success,
                           'strategy'       : np.array( [a.strategy for a in model_simulation.output.action] ),
                           'cmd_output'     : np.array( [a.cmd for a in model_simulation.output.action] ) })
    return df











