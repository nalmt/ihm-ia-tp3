import numpy as np
import itertools
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import QCoreApplication, QPoint

from gui_util import *
from matplotlib_view import *

######################################
#                                    #
#          TRIAL INFO                #
#                                    #
######################################
class Trial_Info(QScrollArea) :

    ##################################
    def __init__(self ):
        super( QScrollArea, self ).__init__()
        self.setWidgetResizable( True )
        self.resize( 250, 350 )
        self.setMinimumWidth(250)
        

    ##################################
    def set_info(self, infos) :
        print( infos )
        self.container = QWidget()
        self.l = QGridLayout()
        self.container.setLayout( self.l )
        self.setWidget( self.container )
        self.container.show()
        for i, key in enumerate( infos.keys() ) :
            print(i, key, infos[key] )
            name_label = QLabel( key )
            value_text = Trial_Info.strategy( infos[key] ) if key == "Strategy" else infos[ key ]
            value_label = LineEdit( value_text )
            value_label.setReadOnly( True )
            self.l.addWidget( name_label, i, 0 )
            self.l.addWidget( value_label, i, 1 )
        self.hide()
        self.show()

    #################################
    def strategy( strategy_id ):
        if strategy_id == "0" :
            return "Menu"
        elif strategy_id == "1" :
            return "Hotkey"
        elif strategy_id == "2" :
            return "Learning"
        else:
            return "Unknow (" + strategy_id + ")"
    




######################################
#                                    #
#      Empirical Panel               #
#                                    #
######################################
class Empirical_Panel ( QTabWidget ) :

    view_selected = pyqtSignal( str ) #trial infos

    ##################################
    def __init__(self):
        super( QTabWidget, self ).__init__()
        self.move(20,20)
        self.resize(750, 750)
        self.subwin_height = 250
        self.subwin_width  = 750
        config_element   = ["Technique", "User", "Command"]
        self.configuration_vec = list(itertools.permutations( config_element ) )
        self.configuration_id = 1
        print( "Configuration:", self.configuration() )

        
        self.gallery = dict()
        #print( self.categories() )
        #exit(0)
        for category in self.categories() :
            #self.gallery[ category ] = Serie2DGallery()
            area = Area()
            self.gallery[ category ] = area
            self.addTab( area, category )
            area.show()
            area.view_moved.connect( self.align_views )

        self.view_vec = dict()
        self.trial_info = Trial_Info( )
      
        #self.show()


    ##################################
    def maximize_sub_window( self ):
        for category in self.categories():
            mdi_area = self.gallery[ category ].container
            position = QPoint( 0, 0 )
            for win in mdi_area.subWindowList():
                win.move( position )
                position.setY(position.y() + win.height() )


        #for category in self.categories():
            #self.gallery[ category ].container.tileSubWindows()

    ##################################
    def show_relevant_tab( self, category_vec ):
        for i in self.count():
            name = self.tabText( i )
            self.setTabVisible( i , name in category_vec )


    ##################################
    def ensure_view_visible( self, technique, user_id, cmd ) :
        index = 0
        gallery = None
        print("ensure view visible")
        tab_title = list( self.gallery.keys() )
        if self.configuration()[0] == "Technique" :
            self.setCurrentIndex( tab_title.index( technique ) )
            gallery = self.gallery[ technique ]          

        elif self.configuration()[0] == "Command" :
            self.setCurrentIndex( tab_title.index( str( cmd ) ) )
            gallery = self.gallery[ str( cmd ) ]            
        
        elif self.configuration()[0] == "User" :
            self.setCurrentIndex( tab_title.index( str( user_id ) ) )
            gallery = self.gallery[ str( user_id ) ]            
        
        QCoreApplication.processEvents()
        key = self.key(technique, user_id, cmd )

        gallery.ensureWidgetVisible( self.view_vec[ key ], 750, 500 )


    ##################################
    def configuration( self ) :
        return self.configuration_vec[ self.configuration_id ]


    ##################################
    def categories( self ) :
        if self.configuration()[0] == "Technique" :
            return ["traditional", "audio", "disable"]
        
        elif self.configuration()[0] == "Command" :
            res = []
            for i in range( 0, 14 ) :
                res.append( str( i ) )
            return res
        
        elif self.configuration()[0] == "User" :
            res = []
            for i in range( 0, 42 ) :
                res.append( str( i ) )
            return res
        else:
            print("EFFOR IN Categories - Configuration", self.configuration()[0], "--- ", self.configuration() )
            exit(0)


    ##################################
    def category( self, d, cmd ) :
        if self.configuration()[0] == "Technique" :
            return d.technique_name
        
        elif self.configuration()[0] == "Command" :
            return str( cmd )
        
        elif self.configuration()[0] == "User" :
            return str( d.id )
        
        else:
            print("EFFOR IN Category - Configuration", self.configuration()[0], "--- ", self.configuration() )
            exit(0)


    ##################################
    def category_bis( self, user_id, cmd, technique_name ) :
        if self.configuration()[0] == "Technique" :
            return technique_name
        
        elif self.configuration()[0] == "Command" :
            return str( cmd )
        
        elif self.configuration()[0] == "User" :
            return str( user_id )
        
        else:
            print("EFFOR IN Category - Configuration", self.configuration()[0], "--- ", self.configuration() )
            exit(0)
        

    ##################################
    def row_bis( self, user_id, cmd, technique_name ) :

        if self.configuration()[1] == "Technique" :
            return self.technique_pos_bis( user_id, cmd, technique_name )
        
        elif self.configuration()[1] == "Command" :
            return self.command_pos_bis( user_id, cmd, technique_name )
        
        elif self.configuration()[1] == "User" :
            return self.user_pos_bis( user_id, cmd, technique_name )
        
        else:
            print("EFFOR in Row - Configuration", self.configuration()[1], "--- ", self.configuration() )
            exit(0)


    ##################################
    def column_bis( self, user_id, cmd, technique_name ):
        if self.configuration()[2] == "Technique" :
            return self.technique_pos_bis( user_id, cmd, technique_name )
        
        elif self.configuration()[2] == "Command" :
            return self.command_pos_bis( use_id, cmd, technique_name )
        
        elif self.configuration()[2] == "User" :
            return self.user_pos_bis( user_id, cmd, technique_name )
        
        else:
            print("EFFOR in Row - Configuration", self.configuration()[1], "--- ", self.configuration() )
            exit(0)     

    ##################################
    def command_pos_bis(self, user_id, cmd, technique_name ) :
        return cmd

    ##################################
    def user_pos_bis( self, user_id, cmd, technique_name ) :
        return int( user_id / 3 )

    ##################################
    def technique_pos_bis( self, user_id, cmd, technique_name ) :
        techniques = ["traditional", "audio", "disable"]
        return np.where( np.array( techniques ) == technique_name )[0][0]


    ##################################
    def row( self, d, cmd ) :

        if self.configuration()[1] == "Technique" :
            return self.technique_pos( d, cmd )
        
        elif self.configuration()[1] == "Command" :
            return self.command_pos( d, cmd )
        
        elif self.configuration()[1] == "User" :
            return self.user_pos( d, cmd )
        
        else:
            print("EFFOR in Row - Configuration", self.configuration()[1], "--- ", self.configuration() )
            exit(0)



    ##################################
    def column( self, d, cmd ):
        if self.configuration()[2] == "Technique" :
            return self.technique_pos( d, cmd )
        
        elif self.configuration()[2] == "Command" :
            return self.command_pos( d, cmd )
        
        elif self.configuration()[2] == "User" :
            return self.user_pos( d, cmd )
        
        else:
            print("EFFOR in Row - Configuration", self.configuration()[1], "--- ", self.configuration() )
            exit(0)

   

    ##################################
    def command_pos(self, d, cmd ) :
        ordered_freq_index = np.argsort( d.command_info.frequency )
        res = len(ordered_freq_index) - 1 - np.where( np.array( ordered_freq_index ) == cmd )[0][0] 
        return res

    ##################################
    def user_pos( self, d, cmd ) :
        return int( d.id / 3 )

    ##################################
    def technique_pos( self, d, cmd ) :
        techniques = ["traditional", "audio", "disable"]
        return np.where( np.array( techniques ) == d.technique_name )[0][0]



    #####################################
    def key( self, technique, user_id, cmd, model ="User" ) :
        return technique + "," + str(user_id) + "," + str(cmd) + "," + model 

    #####################################
    def set_users_df( self, users_df ) :
        user_vec = list( users_df.user_id.unique() )
        cmd_vec  = list( users_df.cmd_input.unique() )
        users_df[ 'bounded_time' ] = np.where( users_df['time'] > 10, 10, users_df['time'] )
        for user_id in user_vec :
            for cmd in cmd_vec :
                df = users_df[ ( users_df.user_id == user_id ) & ( users_df.cmd_input == cmd ) ]
                technique_name = list( df.technique_name.unique() )[ 0 ]
                
                c = self.category_bis( user_id, cmd, technique_name )
                row_id = self.row_bis( user_id, cmd, technique_name )
                col_id = self.column_bis( user_id, cmd, technique_name )
                key = self.key( technique_name, user_id, cmd, "User" )

                view = EpisodeView( )
                view.set_user_cmd_df( df, user_id, cmd, technique_name )
                view.view_selected.connect( self.set_info )
                view.cursor_moved.connect( self.update_cursor )
                self.gallery[ c ].add_view( view.canvas, "User", row_id, col_id, self.subwin_width, self.subwin_height )

                #self.view_vec[ key ] = view.canvas 
                QCoreApplication.processEvents()


    ##################################
    def data_from_id( self, data_vec, user_id ):
        for data in data_vec:
            if data.id == user_id:
                return data


    ##################################
    def set_model_fitting_df( self, goodness_of_fit_vec, users_df ):
        first_visu = True
        user_ids_to_show = users_df.user_id.unique()

        #print( "trace 1", users_df )
        for goodness_of_fit in goodness_of_fit_vec:
            model_name = goodness_of_fit.name + " fitting"
            cmd_vec = users_df.cmd_input.unique()
            users_df[ 'bounded_time' ] = np.where( users_df['time'] > 10, 10, users_df['time'] )
            
            # user_id_fit = goodness_of_fit.user_id
            # user_id_vec = np.intersect1d( user_id_df, user_id_fit )
            # print("set_model_fitting_df", user_id_fit, user_id_df, user_id_vec)

            for i, user_id in enumerate( goodness_of_fit.user_id ) :
                if user_id in user_ids_to_show : 
                    user_df = users_df[ ( users_df.user_id == user_id ) ]
                    user_df = user_df.copy()
                    user_df[ 'menu_prob' ]     = goodness_of_fit.output[ i ].menu * 10
                    user_df[ 'hotkey_prob' ]   = goodness_of_fit.output[ i ].hotkey * 10
                    user_df[ 'learning_prob' ] = goodness_of_fit.output[ i ].learning * 10
                    user_df[ 'model_prob' ]    = goodness_of_fit.prob[ i ] * 10
                    user_df[ 'meta_info']      = goodness_of_fit.output[ i ].meta_info_1 

                    for cmd in cmd_vec :
                        

                        df = user_df[ user_df.cmd_input == cmd ]
                        technique_name = list( df.technique_name.unique() )[ 0 ]
                        c = self.category_bis( user_id, cmd, technique_name )
                        row_id = self.row_bis( user_id, cmd, technique_name )
                        col_id = self.column_bis( user_id, cmd, technique_name )
                        key = self.key( technique_name, user_id, cmd, model_name )
                        if first_visu : #bug to solve
                            view = EpisodeView( )
                            view.set_model_data( df, user_id, cmd, technique_name, model_name )
                            view.canvas.show()
                            view.canvas.hide()
                            first_visu = False
                        view = EpisodeView( )
                        view.set_model_data( df, user_id, cmd, technique_name, model_name )
                        view.view_selected.connect( self.set_info )
                        view.cursor_moved.connect( self.update_cursor )
                        self.gallery[ c ].add_view( view.canvas, model_name, row_id, col_id, self.subwin_width, self.subwin_height )
                        self.view_vec[ key ] = view.canvas
                        
                        QCoreApplication.processEvents()
                        self.maximize_sub_window()

    ##################################
    def set_model_simulation_df( self, _simulation_df, selected_user ):
        simulation_df = _simulation_df.copy()
        model_vec = simulation_df.model.unique()
        user_vec  = simulation_df.user_id.unique()
        cmd_vec   = simulation_df.cmd_input.unique()
        simulation_df[ 'bounded_time' ]  = simulation_df[ 'time' ]
        simulation_df[ 'menu_prob' ]     = simulation_df[ 'menu_prob' ] * 10
        simulation_df[ 'hotkey_prob' ]   = simulation_df[ 'hotkey_prob' ] * 10
        simulation_df[ 'learning_prob' ] = simulation_df[ 'learning_prob' ] * 10
        
        
        for model_name in model_vec:
            for i, user_id in enumerate( user_vec ) :
                if user_id in selected_user:
                    for cmd in cmd_vec :
                        df = simulation_df[ ( simulation_df.model == model_name ) & ( simulation_df.user_id == user_id ) & ( simulation_df.cmd_input == cmd ) ]
                        technique_name = list( df.technique_name.unique() )[ 0 ]
                        c = self.category_bis( user_id, cmd, technique_name )
                        row_id = self.row_bis( user_id, cmd, technique_name )
                        col_id = self.column_bis( user_id, cmd, technique_name )
                        key = self.key( technique_name, user_id, cmd, model_name )

                        view = EpisodeView( )
                        name = model_name + ": Simulation"
                        view.set_model_data( df, user_id, cmd, technique_name, name )
                        #view.view_selected.connect( self.set_info )
                        #view.cursor_moved.connect( self.update_cursor )
                        self.gallery[ c ].add_view( view.canvas, name, row_id, col_id, self.subwin_width, self.subwin_height )
                        self.view_vec[ key ] = view.canvas
                        
                        QCoreApplication.processEvents()
                        self.maximize_sub_window()

    ##################################
    def set_model_simulation_sequences( self, simulation_result_vec, data_vec ):

        for simulation_result in simulation_result_vec:
            model_name = simulation_result.name + ": Simulation"
            user_id    = simulation_result.user_id
            d = self.data_from_id( data_vec, user_id )
            for cmd in d.command_info.id :
                c      = self.category( d, cmd )
                row_id = self.row( d, cmd )
                col_id = self.column( d, cmd )
                key    = self.key(d.technique_name, d.id, cmd, model_name )
                view   = None
                if key in self.view_vec:
                    view = self.view_vec[ key ]
                else:
                    view = EpisodeView( )
                    view.set_agent_data( simulation_result.input, simulation_result.output, user_id, simulation_result.technique_name, cmd )
                    self.view_vec[ key ] = view
                    view.view_selected.connect( self.set_info )
                    view.cursor_moved.connect( self.update_cursor )
                self.gallery[ c ].add_view( view, model_name, row_id, col_id, self.subwin_width, self.subwin_height )
                input = simulation_result.input 
                model_output = simulation_result.prob
                #prob   = simulation_result.prob
                view.set_model_data( model_name, input, model_output, [] )
                #view.set_meta_info(  model_name, d.cmd, model_output.meta_info_1 )
                QCoreApplication.processEvents()
                self.maximize_sub_window()

    

    ##################################
    def update_configuration( self, _id ) :
        self.configuration_id = _id
        print("Update configuration: ", self.configuration(), self.configuration_vec[ _id ] )
        self.gallery.clear()
        self.clear() #TODO THIS DOES NOT REALLY REMOVE ELEMENTS.....

        for category in self.categories() :
            area = Area()
            self.gallery[ category ] = area
            area.view_moved.connect( self.align_views )
            self.addTab( area, category )
            area.show()


        for view in self.view_vec.values():
            cmd            = view.cmd
            user_id        = view.user_id
            d = self.data_from_id( self.data_vec, user_id )

            c = self.category( d, cmd )
            row_id = self.row( d, cmd )
            col_id = self.column( d, cmd )
            self.gallery[ c ].add_view( view, view.model_name, row_id, col_id )

        self.maximize_sub_window()
        self.show()

    ##################################
    def align_views( self, h_value, v_value, _type ):
        for category in self.categories():
            mdi_area = self.gallery[ category ].container
            for win in mdi_area.subWindowList():
                if _type == 1:
                    win.container.verticalScrollBar().setValue( v_value )
                else:
                    win.container.horizontalScrollBar().setValue( h_value )
                    

    ##################################
    def get_exact_trial(self, user_data, cmd, trial_id):
        shift_vec = [ 0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8, 8, -9, 9, -10, 10 ]
        for shift in shift_vec:
            if user_data.cmd[ trial_id + shift ] == cmd:
                return trial_id+shift
        print("get exact trial did not work", trial_id, cmd)
        return trial_id
        
    def update_cursor( self, x ):
        for view in self.view_vec.values() :
            view.update_cursor( x )
        
    ##################################
    def set_info(self, user_id, cmd, _trial_id ) :
        #self.text_edit.setPlainText( "user: " + str(user_id) + "\t cmd:" + str(cmd) + "\t trial:" + str(trial_id) + "->" + "\n" +extra_infos + "\n" + self.text_edit.toPlainText()   )
        info = None
        for user in self.data_vec :
            if user.id == user_id:
                trial_id = self.get_exact_trial( user, cmd, _trial_id )
                info = user.trial_info( trial_id )
        self.trial_info.set_info( info )
        #self.view_selected.emit( extra_infos )
        print( "user: " + str(user_id) + "\t cmd:" + str(cmd) + "\t trial:" + str(trial_id) + "\n"   )


    # def print_all( self, graph_path ) :
    #     technique_vec = ["traditional", "audio", "disable"]
    #     res = 750
    #     if not graph_path[-1]  == '/': 
    #         graph_path = graph_path + '/'
    #     if not os.path.isdir( graph_path ) :
    #         os.mkdir( graph_path )


    #     now = datetime.now()
    #     now_str = now.strftime("%Y_%m_%d__%H_%M_%S")
    #     graph_path += now_str + '/'
    #     if not os.path.isdir( graph_path ) :
    #         os.mkdir( graph_path )


    #     for t in technique_vec :
    #         printer = QPrinter()
    #         printer.setOutputFormat( QPrinter.PdfFormat )
    #         printer.setPageOrientation(QPageLayout.Landscape)
    #         printer.setOutputFileName(graph_path + t + '_gallery.pdf')
    #         #logicalDPIX = printer.logicalDpiX()
    #         #PointsPerInch = 200.0
    #         painter = QPainter()
    #         if not painter.begin(printer):
    #             print("failed to open file, is it writable?");
        
    #         pix = QPixmap( self.gallery[t].container.grab() )
    #         print(t + "pix to print: ", pix.width(), pix.height() )
    #         painter.drawPixmap(0,0, 750,pix.height() * 750. / pix.width(), pix)        
    #         painter.end()


    # def print_unit(self, graph_path) :
    #     res = 750

    #     if not graph_path[-1]  == '/': 
    #         graph_path = graph_path + '/'
    #     if not os.path.isdir( graph_path ) :
    #         os.mkdir( graph_path )

    #     now = datetime.now()
    #     now_str = now.strftime("%Y_%m_%d__%H_%M_%S") 
    #     graph_path += now_str + '/'
    #     if not os.path.isdir( graph_path ) :
    #         os.mkdir( graph_path )


    #     technique_vec = ["traditional", "audio", "disable"]
    #     for technique in technique_vec :
    #         if not os.path.isdir( graph_path + technique + '/' ) :
    #             os.mkdir( graph_path + technique + '/' )


    #     for key in self.view_vec :
    #         technique = key.split(',')[0]
    #         user_id = key.split(',')[1]
    #         cmd = key.split(',')[2]
    #         view = self.view_vec[ key ]
    #         #print(technique, str(view.d.user_id), str(view.cmd))
    #         path = graph_path + technique + '/technique_' + technique + '_user_'+ user_id + '_cmd_' + cmd + '.pdf'
    #         #print(path)
    #         printer = QPrinter()
    #         printer.setOutputFormat( QPrinter.PdfFormat )
    #         printer.setPageOrientation(QPageLayout.Landscape)
    #         printer.setOutputFileName(path)
    #         painter = QPainter()
    #         if not painter.begin(printer):
    #             print("failed to open file, is it writable?");
    #         pix = QPixmap( view.grab() )
    #         painter.drawPixmap(0,0, res,pix.height() * res / pix.width(), pix)        
    #         painter.end()

