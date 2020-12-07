import sys
import time as TIME
import numpy as np
import timeit
import os.path
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QCoreApplication
sys.path.append("./GUI/")
sys.path.append("./lib/")
sys.path.append("./plugins/")
sys.path.append("./plugins/loader/")
sys.path.append("./plugins/export/")
sys.path.append("./plugins/model/")


from user_data_loader import *
from user_overview import User_Overview
from filter import *
from data_explorer import  Empirical_Panel
from random_model import Random_Model
from CK_model import CK_Model
from RW_model import RW_Model
from RWCK_model import RWCK_Model

from model_fit import Model_Fitting
from model_simulation import Model_Simulation
from model_fitting_visualisation import *
from model_simulation_visualisation import *
from parameters_export import Parameters_Export
from parameters_loader import Parameters_Loader

##########################################################################
# function used to see more details how well the model fits              #
# the data of one users                                                  #
# INPUT:                                                                 #
#   - fitting_explorer (Empirical_Panel) : Qt Widget to display data     #
#   - fitting_res (list< Model_Result > ): output of model_fitting.run() #
#   - users_df ( Datframe) : information about users                     #
#   - user_id : id of the user we want to investigate                    #
#                                                                        #
# NOTE                                                                   #
#  to efficiently use this method, you can augment the amount of         #
#  information contained in Model_Result. To do that, in the method      #
#  run_debug() in model_fit.py (step 3.a), add the following             #
#   lines:                                                               #
#       prob_vec      = self.model.action_probs( cmd )                   #                                                              #
#       res.output.menu[ i ]     = probs[ Strategy.MENU ]                #
#       res.output.hotkey[ i ]   = probs[ Strategy.HOTKEY ]              #
#       res.output.learning[ i ] = probs[ Strategy.LEARNING ]            #
#                                                                        #
#  This displays the probability for each action to be chosen            # 
##########################################################################
def show_fitting_details( fitting_explorer, fitting_res, users_df, user_id ):
    df = users_df[ users_df.user_id == user_id ]
    df = df.copy()
    #exit(0)
    #fitting_explorer = Empirical_Panel()
    fitting_explorer.set_model_fitting_df( fitting_res, df )
    fitting_explorer.show()

#######################################################
#                       MAIN                          #
#######################################################
if __name__=="__main__":


    app = QApplication(sys.argv)
    fitting_explorer = Empirical_Panel()
    fitting_explorer.hide()
    simulation_explorer = Empirical_Panel()
    simulation_explorer.hide()


    #######  Load Empirical data ##########
    path = './data/user_data.csv'
    loader = User_Data_Loader()
    users_data = loader.load( path )                        #users_data: array< User_Data > (see util.py )
    # keep only a subset of the data 
    #( 5 participants with traditional and 5 participants with audio )
    my_filter = Filter( user_max = 10, techniques=["traditional", "audio"] )         
    user_data_vec = my_filter.filter( users_data )
    users_df = user_data_vec_to_data_frame( user_data_vec ) # users_df : DataFrame (seaborn)
    
    ###### Load models ##########
    model_vec = [  Random_Model(), CK_Model(), RW_Model(), RWCK_Model() ] 
    
    print( "----------------------------------------------------------" )
    print( "\nlist of users id: ", users_df['user_id'].unique() )
    print( "list of models: ", [model.name for model in model_vec ] )
    print( "\n--------------------------------------------------------" )
    
    

    #######  Show an overview of the data (1.a) ##########
    #overview = User_Overview()
    #overview.set_users_df( users_df )
    #overview.show()



    ####### Show the sequence of commands (TODO 1.b) ##########
    # # you will have some warnings, but it is not a problem  #
    ###########################################################
    #explorer = Empirical_Panel()
    #explorer.subwin_height = 750   
    #explorer.set_users_df( users_df )
    #explorer.show()





    ##############################################################
    ######           Model fitting  (TODO 3.c)          ##########
    ##############################################################
    # model_fitting  = Model_Fitting()
    # model_fitting.debug = True
    # model_fitting.command_ids   = range(0,14)    # 14 commands
    # model_fitting.user_data_vec = user_data_vec
    # model_fitting.model_vec     = model_vec
    # model_fitting.parameters = Parameters_Loader.load('./optimal_parameters_copy/')
    # fitting_res = model_fitting.run()            # res: list < Model_Result > ( see util.py )
    # # # display the results
    # fitting_visu = Model_Fitting_Visualisation()
    # fitting_visu.update_canvas( fitting_res )
    
    
    

    ###############################################################
    #######  Optimize parameters (TODO 4.b) ##########
    ###############################################################
    # model_fitting       = Model_Fitting()
    # model_fitting.debug = True
    # model_fitting.command_ids   = range(0,14)
    # model_fitting.user_data_vec = user_data_vec
    # model_fitting.model_vec     = model_vec
    # fitting_res = model_fitting.optimize()
    # # save parameters
    # Parameters_Export.write(fitting_res, './optimal_parameters/')
    # print("the optimisation is done")
    # exit(0)


    ###############################################################
    #######  Random Model: Model Simulation (TODO 5.b)   ##########
    ###############################################################
    parameters = Parameters_Loader.load('./optimal_parameters_copy/')
    model_simulation = Model_Simulation()
    model_simulation.command_ids   = range(0,14)
    model_simulation.user_data_vec = user_data_vec
    model_simulation.model_vec     = model_vec
    model_simulation.parameters    = Parameters_Loader.load('./optimal_parameters_copy/')
    simulation_res = model_simulation.run()

    # # display the results
    simulation_visu = Model_Simulation_Visualisation()
    simulation_visu.update_canvas( simulation_res, users_df )



    sys.exit(app.exec_())




