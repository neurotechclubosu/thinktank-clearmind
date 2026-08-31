def calculate_neurovascular_variables(EEG_CMRO2):
    
    import math

    # Constants from the paper
    n = 2.28  # Neurovascular coupling constant
    oef0 = 0.42  # Baseline oxygen extraction fraction
    pH_A = 7.40  # Baseline arterial pH
    pCO2_A = 40  # Baseline arterial CO2 pressure in torr
    pO2_A = 100  # Baseline arterial O2 pressure in torr
    pO2_T0 = 27  # Baseline tissue O2 partial pressure
    P50 = 27  # Hemoglobin P50 value in torr
    h = 2.8  # Hill coefficient for oxygen binding
    BC = 26.0  # Buffering capacity constant
    HC = 0.33  # Haldane constant
    CF = 0.14  # Correction factor for hemoglobin binding
    Rglyc = 3.1  # Ratio of nonoxidative glycolysis increase
    cmro2_baseline = 2.1  # Baseline CMRO2 in μmol/(g·min)
    KCO2 = 20  # Normalized capillary CO2 mass transfer coefficient (μmol/(g·min·torr))
    O2_A_conc = 6.9  # Arteriole blood oxygen concentration (mmol/L)
    CO2_A_conc = 25.6  # Arteriole blood CO2 concentration (mmol/L)
    OEF0 = 0.42  # Baseline oxygen extraction fraction
    GEF0 = 0.10  # Baseline glucose extraction fraction
    OGI0 = 5.50  # Oxygen-glucose index
    Tmax = 2.20  # Maximum glucose transport rate
    Kt = 1.16  # Michaelis-Menten constant for glucose transport
    fCO2_plasma = 0.66  # Fraction of total CO2 in plasma
    fVplasma = 0.55  # Plasma volume fraction in whole blood
    pka = 6.10 #pka of carbonic acid in blood

    CBF_Data = {}
    OEF_Data = {}
    ph_V_Data = {}
    pCO2_V_Data = {}
    pO2_cap_Data = {}
    CMRO2_Data = {}
    DeltaHCO2_Data = {}
    DeltaLAC_Data = {}
    
    
    # Calculate OEF
    for Routing in EEG_CMRO2.keys():
        CBF_Data[str(Routing)] = []
        OEF_Data[str(Routing)] = []
        ph_V_Data[str(Routing)] = []
        pCO2_V_Data[str(Routing)] = []
        pO2_cap_Data[str(Routing)] = []
        CMRO2_Data[str(Routing)] = []
        DeltaHCO2_Data[str(Routing)] = []
        DeltaLAC_Data[str(Routing)] = []
        

        for Timestep in range(len(EEG_CMRO2[Routing])):

            delta_cmro2_prime = EEG_CMRO2[Routing][Timestep]
            
            
            
            oef = oef0 * (1+delta_cmro2_prime)/(1+ n * delta_cmro2_prime)
        
            # Define CMRO2 from delta CMRO2
            cmro2 = cmro2_baseline * (1 +  delta_cmro2_prime)
        
            #Define CBF
            CBF = cmro2/(O2_A_conc*oef)
        
            #Define delta_cmro2
            delta_cmro2 = cmro2_baseline * delta_cmro2_prime
        
            #calculate deltaH_solution
            deltaH_lac = 2 * Rglyc * (1/6) * delta_cmro2/CBF
            deltaH_CO2 = ((1-HC) * O2_A_conc * oef)
            deltaH_hemo = (1-HC) * O2_A_conc * oef
            deltaH_solution =  deltaH_CO2 + deltaH_lac - CF * deltaH_hemo
            delta_lac_corrected = CBF * (deltaH_lac)
            deltaH_CO2_corrected = CBF * (deltaH_CO2 - CF * deltaH_hemo)
        
        
            # Calculate pH_V (venous pH) using corrected formula
            delta_pH = (-1 / BC) * deltaH_solution
            pH_V = pH_A + delta_pH
        
            # Calculate pCO2_V (venous CO2 pressure) using corrected formula
            CO2_gas = (CO2_A_conc + fCO2_plasma/fVplasma * O2_A_conc * oef )/ (1 + 10**(pH_V-pka))
            pCO2_V = (1/0.0307)* CO2_gas
        
            #Calculate pO2_cap
            pO2_cap = P50 * (2/oef - 1)**(1/h)
            
            
            CBF_Data[Routing].append(CBF)
            OEF_Data[Routing].append(oef)
            ph_V_Data[Routing].append(pH_V)
            pCO2_V_Data[Routing].append(pCO2_V)
            pO2_cap_Data[Routing].append(pO2_cap)
            CMRO2_Data[Routing].append(cmro2)
            DeltaHCO2_Data[Routing].append(deltaH_CO2_corrected)
            DeltaLAC_Data[Routing].append(delta_lac_corrected)


    return {
        'CBF' : CBF_Data,
        'OEF' : OEF_Data,
        'ph_V' : ph_V_Data,
        'p_CO2_V' : pCO2_V_Data,
        'pO2_cap' : pO2_cap_Data,
        'CMRO2' : CMRO2_Data,
        'DeltaHCO2' : DeltaHCO2_Data,
        'DeltaLAC' : DeltaLAC_Data
            }
