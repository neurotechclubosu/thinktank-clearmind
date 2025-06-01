from EEG_DistanceFunc import DistanceFunc

def EEG_Plotting(Data_val, Timestep_Select, Trial_Select = 1, NodeNum = 100, ElectrodeList = ['Fp1', 'Fp2', 'F3', 'F4', 'T5', 'T6', 'O1', 'O2', 'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4'], ax=None):
    """

    Parameters
    ----------
    Data_val : Dictionary
        Dictionary of the chemical value to be plotted for the hemisphere
        This dictionary should be generated through the EEG_Implement and EEG_Transmute functions
    Timestep_Select : Int
        Selection of the exact timestep in a trial where the values should be plotted: 0-7
    Trial_Select : Int, optional
        The trial to be selected from the data dictionary, if only 1 trial completed set as 1.
    NodeNum : Int, optional
        The number of additional nodes to be generated for the system within the bounds of the 10-20 electrodes
    ElectrodeList : List of Str
        All electrodes in use during trials

    Returns
    -------
    PlotDataSet : List of Lists
        PlotDataSet : List
            Values organized based on EEG_Nodes_List for the chemical value output plotted in the system
        
        EEG_Nodes_List : List
            Coordinate points for all Electrode positions and generated nodes in order relative to PlotDataSet

    """
    
    import numpy as np
    import math
    from scipy.stats import qmc
    import matplotlib.pyplot as plt
    import matplotlib.colors
    
    #All 10-20 EEG system nodes defined and coordinates given to pull from
    EEG_Nodes = {}
    EEG_Nodes['Fp1'] = [0.95, 0.309, -0.0349]
    EEG_Nodes['Fp2'] = [0.95, -0.309, -0.0349]
    EEG_Nodes['F9'] = [0.52, 0.742, -0.423]
    EEG_Nodes['F7'] = [0.587, 0.809, -0.0349]
    EEG_Nodes['F3'] = [0.673, 0.545, 0.5]
    EEG_Nodes['Fz'] = [0.719, 0, 0.695]
    EEG_Nodes['F4'] = [0.673, -0.545, 0.5]
    EEG_Nodes['F8'] = [0.587, -0.809,-0.0349]
    EEG_Nodes['F10'] = [0.52, -0.742, -0.423]
    EEG_Nodes['T9'] = [0, 0.906, -0.423]
    EEG_Nodes['T3'] = [0, 0.999, -0.0349]
    EEG_Nodes['C3'] = [0,0.719,0.695]
    EEG_Nodes['Cz'] = [0,0,1]
    EEG_Nodes['C4'] = [0,-0.719,0.695]
    EEG_Nodes['T4'] = [0,-0.999,-0.0349]
    EEG_Nodes['T10'] = [0,-0.906,-0.423]
    EEG_Nodes['P9'] = [-0.522, 0.733, -0.423]
    EEG_Nodes['T5'] = [-0.587,0.809,-0.0349]
    EEG_Nodes['P3'] = [-0.672, 0.545, 0.5]
    EEG_Nodes['Pz'] = [-0.719, 0, 0.695]
    EEG_Nodes['P4'] = [-0.673, -0.545, 0.5]
    EEG_Nodes['T6'] = [-0.587, -0.809, -0.0349]
    EEG_Nodes['P10'] = [-0.533, -0.733, -0.423]
    EEG_Nodes['O1'] = [-0.95, 0.309, -0.0349] 
    EEG_Nodes['O2'] = [-0.95, -0.309, -0.0349] 
    
    
    #This dictionary will contain all the EEG_Nodes that are actually utilized in the system based on user function input
    Electrode_Nodes_Use = {}
    
    #Selects objects from EEG_Nodes dictionary that are in use within the system
    for value in EEG_Nodes.keys():
            if value in ElectrodeList:
                Electrode_Nodes_Use[value] = EEG_Nodes[value]
    
    #Selects data from the imported Data dictionary. Values are based on electrodes in use, trial number selected, and timestep selected
    Electrode_Data_Use = {}
    
    for dataset in ElectrodeList:
        Electrode_Data_Use[dataset] = Data_val[dataset+ ' trial ' + str(Trial_Select)][Timestep_Select] 
    
    #Creates a list object of the electrode nodes in use, this will be appended based on the space filling function
    EEG_Nodes_List = []    

    for DataListCreate in Electrode_Nodes_Use.keys():
        EEG_Nodes_List.append(Electrode_Nodes_Use[DataListCreate])

    #Defines the space to create a Latin Hypercube (space filling design)
    n_nodes = NodeNum
    radius = 1 #Radius will always be 1 for the node geometry
    engine = qmc.LatinHypercube(d=2)
    
    #Changes the coordinates of the latin hypercube to spherical so only points within our semicircle will generate
    u = engine.random(n_nodes)
    r = np.sqrt(u[:, 0]) * radius
    theta = 2 * np.pi * u[:, 1]
    
    #Defining X and Y coordinates based on the generated spherical hypercube coordinates
    x_cr = r * np.cos(theta)
    y_cr = r * np.sin(theta)
    
    sample_xy = np.column_stack((x_cr, y_cr))
    
    #Initializes and adds Z axis points based on 
    z_cr = [0]*n_nodes
    
    #Finds the maximum height available at each x,y coordinate using pythagorean theorum, then
    #chooses a random height based on this input
    for PyThmCalc in range(n_nodes):
        originDistance = math.sqrt(sample_xy[PyThmCalc][0]**2 + sample_xy[PyThmCalc][1]**2)
        z_cr[PyThmCalc] = math.sqrt(1-(originDistance**2))*np.random.random()
    
    #Final *generated* coordinate array
    sample_xyz = np.column_stack((x_cr, y_cr, z_cr))
    
    #The 3 below for loops each add the newly generated points and values to the electrode
    #values defined through previous functions. 
    initialPoints = []
    for initialPointDisp in Electrode_Nodes_Use.keys():
        initialPoints.append(Electrode_Nodes_Use[initialPointDisp])
    
    #The data values for the entire plotset combined into one list
    PlotDataSet = []
    
    for initialPlotset in Electrode_Data_Use.keys():
        PlotDataSet.append(Electrode_Data_Use[initialPlotset])

    #Uses distance function to determine each individual generated point based on an iterative group of determined values
    for DataCompileA in range(len(sample_xyz)):
        placeholder = DistanceFunc(sample_xyz[DataCompileA],initialPoints,PlotDataSet, 4)
        PlotDataSet.append(placeholder)
        initialPoints.append(sample_xyz[DataCompileA])

    EEG_Nodes_List.extend(sample_xyz)
    
    #Plotting in 3d the full point set. This uses matplotlib scatter function so plugging other plotting methods should be relatively simple using the data.
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.figure 
        ax.clear()

    from matplotlib import colors
    norm = colors.Normalize(vmin=min(PlotDataSet), vmax=max(PlotDataSet))
    #ax.scatter(xs=[sublist[0] for sublist in PlotList],ys=[sublist[1] for sublist in PlotList],zs=[sublist[2] for sublist in PlotList], s=5, c='b', marker="o", label='Space Filling')
    plot = ax.scatter(
    xs=[sublist[0] for sublist in EEG_Nodes_List],
    ys=[sublist[1] for sublist in EEG_Nodes_List],
    zs=[sublist[2] for sublist in EEG_Nodes_List],
    s=10, c=PlotDataSet, marker="o", cmap='Reds', norm=norm
)
    


# Labels
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([0, 1.1])

    plt.title("3D Neurovascular Concentration Plot") 
    

    #print(EEG_Nodes_List)
    
    #Returns a list of the the full Data and coordinates of all points in the system.
    return plot
