def EEG_Implement_Welch(EEG_raw, RecordedColumns=32, Channels=16, Timestep=256,
                        TestDuration=2, TimeColumn=1, ElectrodeList=['Fp1', 'Fp2', 'F3', 'F4', 'T5', 'T6', 'O1', 'O2', 'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4']):
    """
    

    Parameters
    ----------
    EEG_raw : string
        path of EEG file in users directory
    RecordedColumns : Int
        number of total columns in text document uploaded --
        16 channels + ~15 dead lines + 1 time line
        Default is 32
    Channels : Int
        number of electrode channels being used for EEG recording. This program
        assumes that the EEG channels are adjacent to one another in the text
        document when data is recorded, IE: eeg columns 1-16 = channels, 17-32 != channels
    
    Timestep : Int
        timestep used between electrode measurements, 1/(Timestep) seconds
        is measurement time used
        Default is 256
    
    TestDuration: Int
        time in seconds that each EEG test lasts for in the compiled dataset
        
    TimeColumn : Int
        Column number in which the time values are recorded. Used
    
    ElectrodeList : List of String, optional
        All electrodes taking measurements in the EEG input file. The default is ['Fp1', 'Fp2', 'F3', 'F4', 'T5', 'T6', 'O1', 'O2', 'F7', 'F8', 'C3', 'C4', 'T3', 'T4', 'P3', 'P4'].

    Returns
    -------
    EEG_Welch_Spectra : Dict of Lists of Floats
        Dictionary containing all welch spectra data from the EEG input file. 
        
    Trials : Int
        The number of trials run per EEG input file

    """
  
    import numpy as np
    import math
    from scipy.signal import welch
    
    EEG_access = open(EEG_raw, 'r').read().split()
    EEG_use = [[] for _ in range(RecordedColumns)]

    # Read and organize EEG data into channels
    lenCheck = 0
    while (lenCheck * RecordedColumns) < len(EEG_access):
        for EEG_throughput in range(RecordedColumns):
            EEG_use[EEG_throughput].append(float(EEG_access[EEG_throughput + lenCheck * RecordedColumns]))
        lenCheck += 1

    del EEG_use[TimeColumn - 1]      # Remove time column first
    EEG_use = EEG_use[:Channels]    # Keep only EEG channels

    EEG_Welch_Spectra = {}
    
    for Trials in range(1,int(len(EEG_use[0])/(Timestep*TestDuration))+1):
    
        segment_size = 32  
        num_segments = 32  
    
        # Define EEG frequency bands
        bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 100)
        }
        
        #Creates the individual welch spectras based on "segments" or individual electrodes in the system
        for H, electrode in enumerate(ElectrodeList):
            trial_key = f'{electrode} trial ' + str(Trials)
            signal = np.array(EEG_use[H])
    
            #This will be our final output and combines the data as a dict of lists for our data to pull from on other functions
            EEG_Welch_Spectra[trial_key] = {'segments': []}
    
            for seg in range(num_segments):
                start_idx = seg * segment_size
                end_idx = start_idx + segment_size
                if end_idx > len(signal):
                    break
                
                #Welch function via scipy, conditions can be changed via this segment
                window_signal = signal[start_idx:end_idx]
                freqs, psd = welch(window_signal, fs=Timestep, nperseg=segment_size, scaling='density')
    
                # Compute power in each EEG band
                band_power = {}
                for band, (low, high) in bands.items():
                    indices = np.where((freqs >= low) & (freqs <= high))
                    band_power[band] = float(np.sum(psd[indices]))
    
                EEG_Welch_Spectra[trial_key]['segments'].append({
                    'frequencies': freqs.tolist(),
                    'power': psd.tolist(),
                    'band_power': band_power  # Add band power to each segment
                })

    return EEG_Welch_Spectra, Trials
