# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 12:10:33 2021

@author: jan-g
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 16:55:41 2021

@author: jan-g
"""
# import filter_env
from osim.env import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from scipy.stats import pearsonr, spearmanr
from scipy.signal import butter, lfilter, freqz, filtfilt, sosfiltfilt
from stable_baselines import DDPG
import math

class load_Bovi_data:
    def load_kinematic_Bovi_data(self, walking_speed = "Natural"):
      file_name = "Bovi_data.xls"
      sheet_name = "Joint Rotations"
      age = "Adult"  
      
      df = pd.read_excel(file_name, sheet_name = sheet_name, header = [0,1], index_col=0)
      df_gait_cycle = df["Unnamed: 1_level_0"].filter(regex="%")*100
      df_gait_parameter = df[age].filter(regex = walking_speed)
      
      df_time = df_gait_cycle.loc["Hip Flex/Extension"].reset_index().drop("index",axis=1)
      df_hip = df_gait_parameter.loc["Hip Flex/Extension"].reset_index().drop("index",axis=1)
      df_knee = df_gait_parameter.loc["Knee Flex/Extension"].reset_index().drop("index",axis=1)
      df_ankle = df_gait_parameter.loc["Ankle Dorsi/Plantarflexion"].reset_index().drop("index",axis=1)
      
      self.df_Bovi_joints = pd.DataFrame()
      self.df_Bovi_joints["time"] = df_time["% Gait Cycle"]
      self.df_Bovi_joints["ankle_avg"] = df_ankle[walking_speed + ".1"]
      self.df_Bovi_joints["knee_avg"] = df_knee[walking_speed + ".1"]
      self.df_Bovi_joints["hip_avg"] = df_hip[walking_speed + ".1"]
      self.df_Bovi_joints["ankle_std"] = df_ankle[walking_speed + ".1"] - df_ankle[walking_speed]
      self.df_Bovi_joints["knee_std"] = df_knee[walking_speed + ".1"] - df_knee[walking_speed]
      self.df_Bovi_joints["hip_std"] = df_hip[walking_speed + ".1"] - df_hip[walking_speed]
      
      return self.df_Bovi_joints

    # Load grf Bovi data
    def load_kinetic_Bovi_data(self, walking_speed = "Natural"):
      file_name = "Bovi_data.xls"
      sheet_name = "Ground Reaction Forces"
      age = "Adult"  
      
      df = pd.read_excel(file_name, sheet_name = sheet_name, header = [0,1], index_col=0)
      df_gait_cycle = df["Unnamed: 1_level_0"].filter(regex="%")*100
      df_gait_parameter = df[age].filter(regex = walking_speed)
      
      df_time = df_gait_cycle.loc["Anterior/Posterior"].reset_index().drop("index",axis=1)
      df_ap = df_gait_parameter.loc["Anterior/Posterior"].reset_index().drop("index",axis=1)
      df_ver = df_gait_parameter.loc["Medio/Lateral"].reset_index().drop("index",axis=1)
      
      self.df_Bovi_grf = pd.DataFrame()
      self.df_Bovi_grf["time"] = df_time["% Gait Cycle"]
      self.df_Bovi_grf["pa_avg"] = df_ap[walking_speed + ".1"]
      self.df_Bovi_grf["ver_avg"] = df_ver[walking_speed + ".1"]
      self.df_Bovi_grf["pa_std"] = df_ap[walking_speed + ".1"] - df_ap[walking_speed]
      self.df_Bovi_grf["ver_std"] = df_ver[walking_speed + ".1"] - df_ver[walking_speed]
      
      return self.df_Bovi_grf

class gait_analysis:
    
    def __init__(self, filename, trim_begin, trim_end, visualize, 
                 filter_grf_cut_off = 7, filter_grf_order = 2,
                 algorithm_info = None, data_info = None, walking_speed="Natural"):       
        self.filename = filename
        self.trim_begin = trim_begin
        self.trim_end = trim_end
        self.visualize = visualize
        self.filter_grf_cut_off = filter_grf_cut_off
        self.filter_grf_order = filter_grf_order
        self.algorithm_info = algorithm_info
        self.data_info = data_info
        
        self.handle_all_data()
        load_Bovi = load_Bovi_data()
        self.df_Bovi_joints = load_Bovi.load_kinematic_Bovi_data(walking_speed=walking_speed)
        self.df_Bovi_grf = load_Bovi.load_kinetic_Bovi_data(walking_speed=walking_speed)
        
    # do a simulation and save all data    
    def load_simulation(self):
        env = L2RunEnv(self.visualize) #show the simulation
        # Load the trained agent
        model = DDPG.load(self.filename) 
        obs = env.reset()
        states = []
    
        for i in range(env.time_limit):
            states.append(env.get_state_desc()) #save the model states
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            if done:
                env.reset()
                break
        
        self.all_data = states 
    
    # This function gets the joint data frame    
    def get_joint_df(self, par = "pos"):
        joint_list = ["ankle_l", "ankle_r", "hip_l", "hip_r", "knee_l", "knee_r"]
        df = pd.DataFrame(columns=joint_list)
        
        for data in self.all_data:
            dict_you_want = {i: data.get("joint_" + par).get(i)[0] for i in joint_list}
            df = df.append(dict_you_want, ignore_index=True)     
        return df

    # This function gets a dataframe with muscle parameter(par) , --> fiber_length, fiber_velocity
    def get_muscle_df(self, par="activation"):
        muscle_list = ["hamstrings_r","glut_max_r",
                        "iliopsoas_r","vasti_r",
                        "gastroc_r","soleus_r",
                        "tib_ant_r", "hamstrings_l",
                        "glut_max_l", "iliopsoas_l",
                        "vasti_l", "gastroc_l",
                       "soleus_l", "tib_ant_l"]
        df = pd.DataFrame(columns=muscle_list)
        for data in self.all_data:
            dict_you_want = {i: data.get("muscles").get(i).get(par) for i in muscle_list}
            df = df.append(dict_you_want, ignore_index=True)
        return df

    # This function gives the ground reaction forces 
    def get_grf_df(self):
        force_list = ["foot_l","foot_r"]
        df = pd.DataFrame(columns=force_list)
        for data in self.all_data:
            dict_you_want = {i: data.get("forces").get(i)[1] for i in force_list}
            df = df.append(dict_you_want, ignore_index=True)  
            
        
        impact_left = self.get_heel_impacts(df,"foot_l") # This function finds heel impacts
        impact_right = self.get_heel_impacts(df,"foot_r")
        
        foot_off_left = self.get_foot_off(df, "foot_l")
        foot_off_right = self.get_foot_off(df, "foot_r")
        
        df = self.add_impacts_2_df(df,impact_left,"heel_impacts_left") # Add foot impacts to df  
        df = self.add_impacts_2_df(df,impact_right,"heel_impacts_right")
        
        df = self.add_impacts_2_df(df,foot_off_left,"toes_off_left") # Add foot off to df  
        df = self.add_impacts_2_df(df,foot_off_right,"toes_off_right")
        
        df_pa_l = pd.DataFrame([self.all_data[i]["forces"]["foot_l"][0] for i in range(len(self.all_data))], columns=["foot_l_pa"])
        df_pa_r = pd.DataFrame([self.all_data[i]["forces"]["foot_r"][0] for i in range(len(self.all_data))], columns=["foot_r_pa"])
        
        df = pd.concat([df,df_pa_l], axis = 1)
        df = pd.concat([df,df_pa_r], axis = 1)
        return df
    
    # filter the drf df
    def filter_grf_df(self, cuttoff, fs, order=2):
        df_filtered = pd.DataFrame()
        df_filtered["foot_l"] = self.butter_lowpass_filter(self.df_grf["foot_l"], cuttoff, fs, order=order)
        df_filtered["foot_r"] = self.butter_lowpass_filter(self.df_grf["foot_r"], cuttoff, fs, order=order)
        df_filtered["foot_l_pa"] = self.butter_lowpass_filter(self.df_grf["foot_l_pa"], cuttoff, fs, order=order)
        df_filtered["foot_r_pa"] = self.butter_lowpass_filter(self.df_grf["foot_r_pa"], cuttoff, fs, order=order)
        return df_filtered
    
    # lowpass filter
    def butter_lowpass_filter(self, data, cutoff, fs, order=2):
        nyq = 0.5 * fs # Nyquist frequency  
        normal_cutoff = cutoff / nyq
        sos = butter(order, normal_cutoff, "low", output="sos")
        y = sosfiltfilt(sos, data)
        return y

    # This function finds heelimpacts and returs them
    def get_heel_impacts(self, df, foot):
        force = df[foot]
        heel_impact = []
        n = 0
        for i in range(len(force)-1):
            if force[i] > -0.01:
                if force[i+1] < -0.01 and np.mean(force[i:i+10]) < -300: #Ignore small ground contact in swing phase
                    if force[i-5] > -0.01: # -5 to trim irregularities
                        if len(heel_impact) == 0:
                            heel_impact.append(i)
                        elif len(heel_impact) != 0 and abs(i - heel_impact[n]) > 50:
                            heel_impact.append(i)
                            n += 1
        return np.array(heel_impact)
    
    # This function finds foot offs NEED ATTENTION
    def get_foot_off(self, df_grf, foot):
        force = df_grf[foot]
        foot_off = []
        n = 0
        for i in range(10,len(force)-1):
            if force[i] > -2.01:
                if force[i+1] > -2.01:# and np.mean(force[i:i+2]) > -0.01:
                    if force[i-1] < -0.01 and np.mean(force[i-10]) < -0.01:                
                        if len(foot_off) == 0:
                            foot_off.append(i)
                        elif len(foot_off) != 0 and abs(i - foot_off[n]) > 50:
                            foot_off.append(i)
                            n += 1
        return np.array(foot_off)

    # floors any toe bumps
    def floor_segmented_grf_data(self, df_grf_segmented, ref_side):
        if ref_side == "left":
            force_name_list = ["foot_l","foot_l_pa"]
        elif ref_side == "right":
            force_name_list = ["foot_r","foot_r_pa"]
        else:
            force_name_list = ["foot_l","foot_r","foot_l_pa","foot_r_pa"]
    
        for force_name in force_name_list:
            for i in range(len(df_grf_segmented)):
                if df_grf_segmented["swing_phase"].loc[i] != 0:
                    df_grf_segmented[force_name].loc[i] = 0.0
        return df_grf_segmented
                
    # This function adds the impacts to the force dataframe
    def add_impacts_2_df(self, df,heel_impacts, side):
        df[side] = np.zeros(len(df))
        for i in heel_impacts:
            df[side][i] = 1
        return df

    # Impacts can be trimmed to exclude start and end impacts
    def trim_impacts(self,points, trim_begin,trim_end):
        if trim_end > len(points):
            trim_end = len(points)
        return points[trim_begin:trim_end+1]

    # get muscle power of one time step 
    def get_muscle_power_df(self,df_m_force, df_m_velo):
        self.df_muscle_power = pd.DataFrame()
        for muscle in df_m_velo.columns.values:
            self.df_muscle_power[muscle] = df_m_force[muscle] * df_m_velo[muscle]
        return self.df_muscle_power

    # get all metabolic energy    
    def get_metabolic_energy_df(self,df_muscle_force, df_muscle_velo):
        self.df_CoT = pd.DataFrame()
        df_muscle_power = self.get_muscle_power_df(df_muscle_force,df_muscle_velo)
        for muscle in df_muscle_power.columns.values:
            power_list = [] 
            for power in df_muscle_power[muscle]:
                if power < 0:
                    power_list.append(-1/0.25*power)
                else:
                    power_list.append(1/1.2*power)                
            self.df_CoT[muscle] = np.array(power_list)
        return self.df_CoT

    # add segment info to data
    def add_segment_info(self, df_in, ref_leg):
        points = self.df_grf[self.df_grf["heel_impacts_" + ref_leg] == 1].index.values # first get impact points
        points = self.trim_impacts(points,self.trim_begin,self.trim_end) # Next trim impact
        df_new_segements = pd.DataFrame()
        df_new_segements = pd.concat([df_in, df_new_segements], axis = 0)
        
        pnt_cntr = 1
        
        df_new_segements["segments"] = np.zeros(len(df_new_segements))
        for p in range(len(points)-1):
            p_begin = points[p]
            p_end = points[p+1]
            df_new_segements["segments"][p_begin:p_end] = np.ones(len(df_new_segements["segments"][p_begin:p_end])) * pnt_cntr
            pnt_cntr += 1
        
        # add swing phase     
        points_swing = self.df_grf[self.df_grf["toes_off_" + ref_leg] == 1].index.values
        
        min_segment = df_new_segements["segments"].index[df_new_segements["segments"] == 1].min()
        max_segment = df_new_segements["segments"].index[df_new_segements["segments"] == len(points)-1].max()
        
        pnt_cntr = 1
        points_swing_cut = []
        for p in points_swing:
            if p > min_segment and p < max_segment:
                points_swing_cut.append(p)
                
        points_swing = np.array((points_swing_cut))   
        df_new_segements["swing_phase"] = np.zeros(len(df_new_segements))
        for p in range(len(points_swing)):
            p_begin = points_swing[p]
            p_end = points[p+1]
            df_new_segements["swing_phase"][p_begin:p_end] = np.ones(len(df_new_segements["swing_phase"][p_begin:p_end])) * pnt_cntr
            pnt_cntr += 1
            
        df_new_segements = df_new_segements[df_new_segements["segments"] != 0]
        df_new_segements = df_new_segements.reset_index()
        
        return df_new_segements

    # get swing phase size
    def swing_phase_size(self, df_segmented):
        toe_off_list = []
        max_i = int(df_segmented["swing_phase"].max())
        
        for i in range(1,max_i+1):
            toe_off_list.append(df_segmented.index[df_segmented["swing_phase"] == i].size / df_segmented.index[df_segmented["segments"] == i].size)
        return np.mean(toe_off_list), np.std(toe_off_list)
    
    def get_toe_off_gait(self, side="right"):
        if side == "right": 
            mean_toe_off, std_toe_off = self.swing_phase_size(self.df_seg_kin_right)
        else: 
            mean_toe_off, std_toe_off = self.swing_phase_size(self.df_seg_kin_left)
            
        return 100-mean_toe_off*100

    # gtet mean per segment df
    def get_mean_per_segment_df(self,df_segmented):
        df_mean = pd.DataFrame()
        for i in range(1, int(df_segmented["segments"].max())):
            mean_segment = df_segmented.where(df_segmented["segments"] == i).mean()
            df_mean = df_mean.append(mean_segment, ignore_index=True)
            df_mean = df_mean.drop(columns=["segments"])
        return df_mean

    # Concat two df with mean and std as index
    def concat_mean_per_seg(self,df_in, df_out,i):
        for par in df_out.columns.values:
            df_in[par+ " k "+ str(i)] = df_out[par]
        return df_in
        
    # get the distance traveled on the pelvis
    def get_pelvis_pos(self):
        pelvis_pos = [] 
        for i in range(len(self.all_data)):
            pelvis_pos.append(self.all_data[i]["body_pos"]["pelvis"][0])
            
        df_pelvis_pos = pd.DataFrame()
        df_pelvis_pos["pelvis_x"] = np.array(pelvis_pos)    
        return df_pelvis_pos

    # Get the Cost of Transport in a nice dataframe
    def get_CoT_info(self):
        df_CoT = pd.DataFrame()
        df_muscle_force = self.get_muscle_df(self.all_data, par = "fiber_force")
        df_muscle_velo = self.get_muscle_df(self.all_data, par = "fiber_velocity")
        df_met_energy = self.get_metabolic_energy_df(df_muscle_force, df_muscle_velo)
        df_met_energy = df_met_energy.sum(axis=1)
        df_pelvis_pos = self.get_pelvis_pos(self.all_data)
        
        dx_list = [0]
        CoT_list = [0]
        time = [0]
            
        for i in range(1,len(df_pelvis_pos)):
            dx_list.append(df_pelvis_pos["pelvis_x"][i] - df_pelvis_pos["pelvis_x"][i-1])
            CoT_list.append(df_met_energy[i] / dx_list[i])
            time.append(i*0.01)
            
        df_CoT["time"] = np.array(time)
        df_CoT["energy"] = df_met_energy
        df_CoT["pelvis_x"] = df_pelvis_pos["pelvis_x"]
        df_CoT["pelvis_x_dx"] = np.array(dx_list)
        df_CoT["CoT"] = np.array(CoT_list)
        
        return df_CoT 

    # Get Range Of Motions of joints
    def get_ROM(self, df_kinematics,joint):  
        n_segments = int(df_kinematics["segments"].max())
        ROM_list = []
        for i in range(1,n_segments+1):
            max_segment = df_kinematics[joint].where(df_kinematics["segments"] == i).max()
            min_segment = df_kinematics[joint].where(df_kinematics["segments"] == i).min()
            ROM_list.append(abs(max_segment - min_segment))
        ROM_list = np.array(ROM_list) 
        return np.mean(ROM_list), np.std(ROM_list)
        
    # Get max angle during swing phase
    def get_max_angle_swing(self, df_kinematics, joint):
        n_segments = int(df_kinematics["segments"].max())
        max_joint_list = [] 
        for i in range (1,n_segments+1):
            max_joint_list.append(df_kinematics[joint].where(df_kinematics["swing_phase"] == i).max())
        max_joint_list = np.array(max_joint_list) 
        return np.mean(max_joint_list), np.std(max_joint_list)

    # Get min angle knee in swing 
    def get_min_angle_swing(self, df_kinematics, joint):
        n_segments = int(df_kinematics["segments"].max())
        min_joint_list = [] 
        for i in range (1,n_segments+1):
            min_joint_list.append(df_kinematics[joint].where(df_kinematics["swing_phase"] == i).min())
        min_joint_list = np.array(min_joint_list)
        return np.mean(min_joint_list), np.std(min_joint_list)
    
    # Get min peak force 
    def get_min_peak_force(self, df_kinetics, force_name):
        n_segments = int(df_kinetics["segments"].max())
        min_force = [] 
        for i in range (1,n_segments+1):
            min_force.append(df_kinetics[force_name].where(df_kinetics["segments"] == i).min())
        min_force = np.array(min_force) 
        return np.mean(min_force), np.std(min_force)
    
    # Get max peak force
    def get_max_peak_force(self, df_kinetics, force_name):
        n_segments = int(df_kinetics["segments"].max())
        max_force = [] 
        for i in range (1,n_segments+1):
            max_force.append(df_kinetics[force_name].where(df_kinetics["segments"] == i).max())
        max_force = np.array(max_force) 
        return np.mean(max_force), np.std(max_force)

    # Disp kinematic parameters 
    def show_kinematic_parameters(self):
        ### ROM ###
        # left
        ankle_l_mean, ankle_l_std = self.get_ROM(self.df_seg_joint_left,"ankle_l")
        hip_l_mean, hip_l_std = self.get_ROM(self.df_seg_joint_left,"hip_l")
        ankle_r_mean, ankle_r_std = self.get_ROM(self.df_seg_joint_right,"ankle_r") 
        hip_r_mean, hip_r_std = self.get_ROM(self.df_seg_joint_right,"hip_r") 
    
        ### MAX Knee SWING ###
        knee_l_max_mean, knee_l_max_std = self.get_max_angle_swing(self.df_seg_joint_left, "knee_l") 
        knee_r_max_mean, knee_r_max_std = self.get_max_angle_swing(self.df_seg_joint_right, "knee_r")
    
        print("Ankle left ROM mean: {}, std: {}".format(ankle_l_mean,ankle_l_std))
        print("Ankle right ROM mean: {}, std: {}".format(ankle_r_mean,ankle_r_std))
        
        print("Knee left swing max mean: {}, std: {}".format(knee_l_max_mean,knee_l_max_std))
        print("Knee right swing max mean: {}, std: {}".format(knee_r_max_mean,knee_r_max_std))
        
        print("Hip left ROM mean: {}, std: {}".format(hip_l_mean,hip_l_std))
        print("Hip right ROM mean: {}, std: {}".format(hip_r_mean,hip_r_std))
    
       
    # Disp grf parameters 
    def show_kinetic_parameters(self):
        wgt = -79.7*9.81
        pnr_max_mean, pnr_max_std = self.get_min_peak_force(self.df_seg_grf_right, "foot_r")
        pbr_max_mean , pbr_max_std = self.get_max_peak_force(self.df_seg_grf_right, "foot_r_pa")
        ppr_max_mean, ppr_max_std = self.get_min_peak_force(self.df_seg_grf_right, "foot_r_pa")
        pnl_max_mean, pnl_max_std = self.get_min_peak_force(self.df_seg_grf_left, "foot_l")
        pbl_max_mean , pbl_max_std = self.get_max_peak_force(self.df_seg_grf_left, "foot_l_pa")
        ppl_max_mean, ppl_max_std = self.get_min_peak_force(self.df_seg_grf_left, "foot_l_pa")
        
        print("Peak right vertical force mean: {}, std: {}".format(pnr_max_mean/wgt,pnr_max_std/wgt))
        print("Peak right braking force mean: {}, std: {}".format(pbr_max_mean/wgt,pbr_max_std/wgt))
        print("Peak right propulsion force mean: {}, std: {}".format(ppr_max_mean/wgt,ppr_max_std/wgt))
    
        print("Peak left vertical force mean: {}, std: {}".format(pnl_max_mean/wgt,pnl_max_std/wgt))
        print("Peak left braking force mean: {}, std: {}".format(pbl_max_mean/wgt,pbl_max_std/wgt))
        print("Peak left propulsion force mean: {}, std: {}".format(ppl_max_mean/wgt,ppl_max_std/wgt))

    # This function gets the Winter data set    
    def get_winter_data(self,file_name, sheet_name):
        self.df_Winter = pd.read_excel(file_name, sheet_name = sheet_name, header = [0,1])
        return self.df_Winter

    # This function plots Winter data
    def plot_winter_data(self, joint, walking_speed="normal"):
        x_data = self.df_Winter["time"]["%_stride"]
        y_data = self.df_Winter[walking_speed][joint + "_avg"]
        y_data_std = self.df_Winter[walking_speed][joint + "_std"]
        plt.plot(x_data,y_data, label = "Winter et al.", color = "gray", linestyle= "--")
        plt.fill_between(x_data, y_data + y_data_std,
                         y_data - y_data_std, color = "gray" ,alpha = 0.5, label="std")
        plt.legend()    
    
    # Interpolate data and get mean and std
    def interpolate_data(self,df_segmented):
        dfs = []
        # Pick one segment
        segment_range = df_segmented["segments"].max()
        for i in range(1, int(segment_range+1)):
            y_old = df_segmented.where(df_segmented["segments"] == i).dropna().reset_index(drop=True)
            y_old = y_old.drop(columns=["segments","swing_phase","index"])
            x_old = np.linspace(0,100,num = int(len(y_old)))
            
            f_int = interp1d(x_old, y_old.T, kind='cubic')
            #_df
            x_new = np.linspace(0,100,num = 101)
            y_new = pd.DataFrame(f_int(x_new).T, columns=(y_old.columns.values))
            dfs.append(y_new)
        y_mean_int = pd.DataFrame(np.mean(dfs,axis=0), columns=(y_old.columns.values))
        y_std_int = pd.DataFrame(np.std(dfs,axis=0), columns=(y_old.columns.values))
        return y_mean_int, y_std_int
            
    # Show correlation        
    def show_correlation(self, df_left,df_right,df_ref):
        print("Pearson L-R: corr = {}; pval = {}".format(pearsonr(df_left,df_right)[0],pearsonr(df_left,df_right)[1]))
        print("Pearson L-ref: corr = {}; pval = {}".format(pearsonr(df_left,df_ref)[0],pearsonr(df_left,df_ref)[1]))
        print("Pearson R-ref: corr = {}; pval = {}".format(pearsonr(df_right,df_ref)[0],pearsonr(df_right,df_ref)[1]))
    
    
    # Show joint correlation
    def show_joint_correlation(self):
        print("Joint correlations \nankle")
        
        self.show_correlation(self.df_mean_joint_left_int["ankle_l"],
                              self.df_mean_joint_right_int["ankle_r"],
                              self.df_Bovi_joints["ankle_avg"]) 
        
        print("knee")
        self.show_correlation(self.df_mean_joint_left_int["knee_l"],
                              self.df_mean_joint_right_int["knee_r"],
                              self.df_Bovi_joints["knee_avg"]) 
        
        print("hip")
        self.show_correlation(self.df_mean_joint_left_int["hip_l"],
                              self.df_mean_joint_right_int["hip_r"],
                              self.df_Bovi_joints["hip_avg"]) 
    
    def show_grf_correlation(self):
        print("GRF correlation \n vertical")
        self.show_correlation(self.df_mean_grf_int_left["foot_l"],
                              self.df_mean_grf_int_right["foot_r"],
                              self.df_Bovi_grf["ver_avg"]) 
        print("horizontal")
        self.show_correlation(self.df_mean_grf_int_left["foot_l_pa"],
                              self.df_mean_grf_int_right["foot_r_pa"],
                              self.df_Bovi_grf["pa_avg"]) 
    
    # Handle all data
    def handle_joint_data(self):
        # load data
        self.df_joint_pos = self.get_joint_df(par="pos") * 180 / np.pi # get joint data
        self.df_joint_pos["knee_r"] = -self.df_joint_pos["knee_r"]
        self.df_joint_pos["knee_l"] = -self.df_joint_pos["knee_l"]

        # make left and right data frames joints 
        # Segment kinematic data
        self.df_seg_joint_right = self.add_segment_info(self.df_joint_pos, ref_leg = "right")
        self.df_seg_joint_left = self.add_segment_info(self.df_joint_pos, ref_leg = "left")
    
        # Interpolate data
        self.df_mean_joint_right_int, self.df_std_joint_right_int = self.interpolate_data(self.df_seg_joint_right)
        self.df_mean_joint_left_int, self.df_std_joint_left_int = self.interpolate_data(self.df_seg_joint_left)
    
    def handle_grf_data(self): 
        self.df_grf = self.get_grf_df()
        
        self.df_grf_filt = self.filter_grf_df(self.filter_grf_cut_off, 
                                              100, order=self.filter_grf_order)
        df_seg_grf_right = self.add_segment_info(self.df_grf_filt, ref_leg = "right")
        df_seg_grf_left = self.add_segment_info(self.df_grf_filt, ref_leg = "left")
        self.df_seg_grf_right = self.floor_segmented_grf_data(df_seg_grf_right, 
                                                              ref_side = "right")
        self.df_seg_grf_left = self.floor_segmented_grf_data(df_seg_grf_left, 
                                                             ref_side = "left")
    
        # Interpolate data
        self.df_mean_grf_int_right, self.df_std_grf_int_right = self.interpolate_data(df_seg_grf_right)
        self.df_mean_grf_int_left, self.df_std_grf_int_left = self.interpolate_data(df_seg_grf_left)
        
        # norm to weight        
        wgt = -79.7*9.81 
        self.df_mean_grf_int_right /= wgt
        self.df_std_grf_int_right /= wgt
        self.df_mean_grf_int_left /= wgt
        self.df_std_grf_int_left /= wgt  
        self.df_mean_grf_int_left.loc[0] = 0
        self.df_mean_grf_int_right.loc[0] = 0
        
    def handle_all_data(self):
        self.load_simulation()
        self.handle_grf_data()
        self.handle_joint_data()
        
class plot_data:
    def __init__(self, walking_speed="Natural"): 
        load_Bovi = load_Bovi_data()
        self.df_Bovi_joints = load_Bovi.load_kinematic_Bovi_data(walking_speed=walking_speed)
        self.df_Bovi_grf = load_Bovi.load_kinetic_Bovi_data(walking_speed=walking_speed)
        
    def start_plot(self):
        plt.figure()
        
    def add_data(self, df_mean, df_std, element, label_name, color, linestyle):
        x = np.linspace(0,100,len(df_mean))
        plt.plot(x, df_mean[element], label = label_name, color=color, linestyle=linestyle)
        # plt.fill_between(x,(df_mean[element] + df_std[element]), 
        #                     (df_mean[element] - df_std[element]), color=color, alpha=0.20) 
    
    def add_reference(self, element, ref_type="kinematics",walking_speed="Natural"):
        if ref_type == "kinematics":
            element = element[:-2]
            self.plot_Bovi(element, walking_speed=walking_speed)
        elif ref_type == "grf":
            self.plot_Bovi_grf(element, walking_speed=walking_speed)
            
    def add_toe_off_line(self, line_x):
        plt.axvline(x=line_x, linestyle="dotted", color="black")
            
    def end_plot(self, y_min = -20, y_max =120, save_fig=False, title = None,
                 legend = True, save_name = "DDPG_kinematics", 
                 x_label = "% gait cycle", y_label = "Knee flexion [deg]"):
        if title:
            plt.title(title, fontsize=18)
        if legend:       
            plt.legend(loc='lower center',bbox_to_anchor=(0.5, -0.31),
          fancybox=True, shadow=True, ncol=5, fontsize=12)
        plt.ylim((y_min,y_max))
        plt.xlim((0,100))
        plt.xlabel(x_label, size=18)
        plt.ylabel(y_label, size=18)
        plt.subplots_adjust(bottom=0.2)
        plt.grid()
        if save_fig:
            plt.savefig(save_name)

    # Plot Bovi joint
    def plot_Bovi(self, element, walking_speed = "Natural"):
        x_data = self.df_Bovi_joints["time"] 
        y_data_mean = self.df_Bovi_joints[element + "_avg"]
        y_data_std = self.df_Bovi_joints[element+ "_std"]
        # plt.plot(x_data,y_data_mean, color = "gray", linestyle= "--", label = "Bovi et al.")
        plt.fill_between(x_data,(y_data_mean + y_data_std), 
                            (y_data_mean - y_data_std),  color = "gray", alpha=0.5, label = "Experimental") 
    
    # Plot Bovi grf
    def plot_Bovi_grf(self, element, walking_speed = "Natural"):
        if element == "foot_l" or element == "foot_r":
            element = "ver"
        elif element == "foot_l_pa" or element == "foot_r_pa":
            element = "pa" 
        
        x_data = self.df_Bovi_grf["time"] 
        y_data_mean = self.df_Bovi_grf[element + "_avg"]
        y_data_std = self.df_Bovi_grf[element+ "_std"]
        # plt.plot(x_data,y_data_mean, color = "gray", linestyle= "--", label = "Bovi et al.")
        plt.fill_between(x_data,(y_data_mean + y_data_std), 
                            (y_data_mean - y_data_std),  color = "gray", alpha=0.5, label = "Experimetal") 
    
    
if __name__ == '__main__':
    filename0 = "./Results_experiment1/policies/logk00/DDPG_CoT__1200000_steps.zip"
    DDPGk0 = gait_analysis(filename0, trim_begin=1, trim_end=4, visualize=False,
                            walking_speed = "L")
    fileCoL = "D:\Master_thesis/New_storage\Test_CoL\Hyper_parameters2\CoL_test16/logs\DDPG_CoT__1200000_steps.zip"
    COLk0 = gait_analysis(fileCoL, trim_begin=0, trim_end=3, visualize=False, walking_speed ="L")
    filename1 = "./Results_experiment1/policies/logk01/DDPG_CoT__1100000_steps.zip"
    DDPGk1 = gait_analysis(filename1, trim_begin=3, trim_end=6, visualize=False, walking_speed ="L")
    filename2 = "./Results_experiment1/policies/logk02/DDPG_CoT__1200000_steps.zip"
    DDPGk2 = gait_analysis(filename2, trim_begin=1, trim_end=4, visualize=False, walking_speed ="L")
    filename3 = "./Results_experiment1/policies/logk03/DDPG_CoT__1200000_steps.zip"
    DDPGk3 = gait_analysis(filename3, trim_begin=0, trim_end=3, visualize=False, walking_speed ="L")
    
    fileBC3 = "D:\Master_thesis/New_storage\Test_BC\Final\BC_DDPG03/data/test/model_step_1200000.pkl"
    BCk3 = gait_analysis(fileBC3, trim_begin=0, trim_end=3, visualize=False, walking_speed ="L")

    
    
    
    # Ankle
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk0.df_mean_joint_right_int,DDPGk0.df_std_joint_right_int, "ankle_r",label_name="DDPG",color = "C0", linestyle="solid")
    pt.add_data(COLk0.df_mean_joint_right_int,COLk0.df_std_joint_right_int, "ankle_r",label_name="COL",color = "C0", linestyle="dotted")
    pt.add_reference("ankle_r")
    pt.end_plot(title = "Ankle k = 0", y_min = -60, y_max =60, y_label = "Joint flexion [deg]", save_fig=False, save_name="Ankle_k00")
    
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk1.df_mean_joint_left_int,DDPGk1.df_std_joint_left_int, "ankle_l",label_name="DDPG",color = "C1", linestyle="solid")
    pt.add_reference("ankle_r")
    pt.end_plot(title = "Ankle k = 1", y_min = -60, y_max =60, y_label = "Joint flexion [deg]", save_fig=False, save_name="Ankle_k01")

    
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk2.df_mean_joint_right_int,DDPGk2.df_std_joint_right_int, "ankle_r",label_name="DDPG",color = "C2", linestyle="solid")
    pt.add_reference("ankle_r")
    pt.end_plot(title = "Ankle k = 2", y_min = -60, y_max =60, y_label = "Joint flexion [deg]", save_fig=False, save_name="Ankle_k02")
    
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk3.df_mean_joint_left_int,DDPGk3.df_std_joint_left_int, "ankle_l",label_name="DDPG",color = "C3", linestyle="solid")
    pt.add_reference("ankle_r")
    pt.end_plot(title = "Ankle k = 3", y_min = -60, y_max =60, y_label = "Joint flexion [deg]", save_fig=False, save_name="Ankle_k03")
    
    # Knee
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk0.df_mean_joint_right_int,DDPGk0.df_std_joint_right_int, "knee_r",label_name="DDPG",color = "C0", linestyle="solid")

    pt.add_data(COLk0.df_mean_joint_right_int,COLk0.df_std_joint_right_int, "knee_r",label_name="COL",color = "C0", linestyle="dotted")
    pt.add_reference("knee_r")
    pt.end_plot(title = "Knee k = 0", y_min = -20, y_max =120, y_label = "Joint flexion [deg]", save_fig=False, save_name="Knee_k00")
    
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk1.df_mean_joint_left_int,DDPGk1.df_std_joint_left_int, "knee_l",label_name="DDPG",color = "C1", linestyle="solid")
    pt.add_reference("knee_r")
    pt.end_plot(title = "Knee k = 1", y_min = -20, y_max =120, y_label = "Joint flexion [deg]", save_fig=False, save_name="Knee_k01")

    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk2.df_mean_joint_right_int,DDPGk2.df_std_joint_right_int, "knee_r",label_name="DDPG",color = "C2", linestyle="solid")
    pt.add_reference("knee_r")
    pt.end_plot(title = "Knee k = 2", y_min = -20, y_max =120, y_label = "Joint flexion [deg]", save_fig=False, save_name="Knee_k02")

    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk3.df_mean_joint_left_int,DDPGk3.df_std_joint_left_int, "knee_l",label_name="DDPG",color = "C3", linestyle="solid")
    pt.add_reference("knee_r")
    pt.end_plot(title = "Knee k = 3", y_min = -20, y_max =120, y_label = "Joint flexion [deg]", save_fig=False, save_name="Knee_k03")
    
     # Hip
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk0.df_mean_joint_right_int,DDPGk0.df_std_joint_right_int, "hip_r",label_name="DDPG",color = "C0", linestyle="solid")

    pt.add_data(COLk0.df_mean_joint_right_int,COLk0.df_std_joint_right_int, "hip_r",label_name="COL",color = "C0", linestyle="dotted")
    pt.add_reference("hip_r")
    pt.end_plot(title = "Hip k = 0", y_min = -50, y_max =70, y_label = "Joint flexion [deg]", save_fig=False, save_name="Hip_k00")
    
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk1.df_mean_joint_left_int,DDPGk1.df_std_joint_left_int, "hip_l",label_name="DDPG",color = "C1", linestyle="solid")
    pt.add_reference("hip_l")
    pt.end_plot(title = "Hip k = 1", y_min = -50, y_max =70, y_label = "Joint flexion [deg]", save_fig=False, save_name="Hip_k01")

    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk2.df_mean_joint_right_int,DDPGk2.df_std_joint_right_int, "hip_r",label_name="DDPG",color = "C2", linestyle="solid")
    pt.add_reference("hip_r")
    pt.end_plot(title = "Hip k = 2", y_min = -50, y_max =70, y_label = "Joint flexion [deg]", save_fig=False, save_name="Hip_k02")

    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk3.df_mean_joint_left_int,DDPGk3.df_std_joint_left_int, "hip_l",label_name="DDPG",color = "C3", linestyle="solid")
    pt.add_reference("hip_r")
    pt.end_plot(title = "Hip k = 3", y_min = -50, y_max =70, y_label = "Joint flexion [deg]", save_fig=False, save_name="Hip_k03")
    
    #%%
    
    # Vertical
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk0.df_mean_grf_int_right,DDPGk0.df_std_grf_int_right, "foot_r",label_name="DDPG",color = "C0", linestyle="solid")
    pt.add_data(COLk0.df_mean_grf_int_left,COLk0.df_std_grf_int_left, "foot_l",label_name="COL",color = "C0", linestyle="dotted")
    pt.add_reference("foot_r", ref_type="grf")
    pt.end_plot(title = "Vertical GRF k = 0", y_min = -0.2, y_max =2.5, y_label = "Force [%BW]", save_fig=False, save_name="Vertical_k00")
    
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk1.df_mean_grf_int_left,DDPGk1.df_std_grf_int_left, "foot_l",label_name="DDPG",color = "C1", linestyle="solid")
    pt.add_reference("foot_r", ref_type="grf")
    pt.end_plot(title = "Vertical GRF k = 1", y_min = -0.2, y_max =2.5, y_label = "Force [%BW]", save_fig=False, save_name="Vertical_k01")
    
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk2.df_mean_grf_int_right,DDPGk2.df_std_grf_int_right, "foot_r",label_name="DDPG",color = "C2", linestyle="solid")
    pt.add_reference("foot_r", ref_type="grf")
    pt.end_plot(title = "Vertical GRF k = 2", y_min = -0.2, y_max =2.5, y_label = "Force [%BW]", save_fig=False, save_name="Vertical_k02")

    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk3.df_mean_grf_int_left,DDPGk3.df_std_grf_int_left, "foot_l",label_name="DDPG",color = "C3", linestyle="solid")
    pt.add_reference("foot_r", ref_type="grf")
    pt.end_plot(title = "Vertical GRF k = 3", y_min = -0.2, y_max =2.5, y_label = "Force [%BW]", save_fig=False, save_name="Vertical_k03")
    
    
    # Anterior
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk0.df_mean_grf_int_right,DDPGk0.df_std_grf_int_right, "foot_r_pa",label_name="DDPG",color = "C0", linestyle="solid")
    pt.add_data(COLk0.df_mean_grf_int_left,COLk0.df_std_grf_int_left, "foot_l_pa",label_name="COL",color = "C0", linestyle="dotted")
    pt.add_reference("foot_r_pa", ref_type="grf")
    pt.end_plot(title = "Anterior GRF k = 0", y_min = -0.75, y_max =0.75, y_label = "Force [%BW]", save_fig=False, save_name="Anterior_k00")
    
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk1.df_mean_grf_int_left,DDPGk1.df_std_grf_int_left, "foot_l_pa",label_name="DDPG",color = "C1", linestyle="solid")
    pt.add_reference("foot_r_pa", ref_type="grf")
    pt.end_plot(title = "Anterior GRF k = 1", y_min = -0.75, y_max =0.75, y_label = "Force [%BW]", save_fig=False, save_name="Anterior_k01")
    
    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk2.df_mean_grf_int_right,DDPGk2.df_std_grf_int_right, "foot_r_pa",label_name="DDPG",color = "C2", linestyle="solid")
    pt.add_reference("foot_r_pa", ref_type="grf")
    pt.end_plot(title = "Anterior GRF k = 2", y_min = -0.75, y_max =0.75, y_label = "Force [%BW]", save_fig=False, save_name="Anterior_k02")

    pt = plot_data(walking_speed="L")
    pt.start_plot()
    pt.add_data(DDPGk3.df_mean_grf_int_left,DDPGk3.df_std_grf_int_left, "foot_l_pa",label_name="DDPG",color = "C3", linestyle="solid")
    pt.add_reference("foot_r_pa", ref_type="grf")
    pt.end_plot(title = "Anterior GRF k = 3", y_min = -0.75, y_max =0.75, y_label = "Force [%BW]", save_fig=False, save_name="Anterior_k03")
    
    
    
    
    # pt.start_plot()
    # pt.add_data(DDPGk0.df_mean_joint_right_int,DDPGk0.df_std_joint_right_int, "knee_r",label_name="DDPG k = 0.0",color = "C0", linestyle="solid")
    # pt.add_data(DDPGk0.df_mean_joint_left_int,DDPGk0.df_std_joint_left_int, "knee_l",label_name="DDPG k = 0.0",color = "C1", linestyle="solid")
    # pt.add_reference("knee_r")
    # pt.end_plot(title = "Right leg", y_min = -90, y_max =90, y_label = "Knee flexion [deg]", save_fig=False)
    
    # pt.start_plot()
    # pt.add_data(DDPGk0.df_mean_joint_right_int,DDPGk0.df_std_joint_right_int, "hip_r",label_name="DDPG k = 0.0",color = "C0", linestyle="solid")
    # pt.add_reference("hip_r")
    # pt.end_plot(title = "Right leg", y_min = -60, y_max =60, y_label = "Hip flexion [deg]", save_fig=False)
    
    
    # pt = plot_data(walking_speed="L")
    # pt.start_plot()
    # pt.add_data(DDPGk1.df_mean_joint_left_int,DDPGk1.df_std_joint_left_int, "ankle_l",label_name="DDPG k = 0.1",color = "C0", linestyle="solid")
    # pt.add_reference("ankle_l")
    # pt.end_plot(title = "Left leg", y_min = -60, y_max =60, y_label = "Ankle flexion [deg]", save_fig=False)
    
    # pt.start_plot()
    # pt.add_data(DDPGk1.df_mean_joint_left_int,DDPGk1.df_std_joint_left_int, "knee_l",label_name="DDPG k = 0.1",color = "C0", linestyle="solid")
    # pt.add_reference("knee_l")
    # pt.end_plot(title = "Right leg", y_min = -60, y_max =60, y_label = "Knee flexion [deg]", save_fig=False)
    
    # pt.start_plot()
    # pt.add_data(DDPGk1.df_mean_joint_left_int,DDPGk1.df_std_joint_left_int, "Hip_l",label_name="DDPG k = 0.1",color = "C0", linestyle="solid")
    # pt.add_reference("hip_l")
    # pt.end_plot(title = "Left leg", y_min = -60, y_max =60, y_label = "Hip flexion [deg]", save_fig=False)
    
    
    
    
    
    
    
    
    # pt.start_plot()
    # pt.add_data(k0.df_mean_joint_right_int,k0.df_std_joint_right_int, "knee_r", label_name= "DDPG k = 0.0" )
    # pt.add_data(k1.df_mean_joint_left_int,k1.df_std_joint_left_int, "knee_l", label_name= "DDPG k = 0.1" )
    # toe_off = k0.get_toe_off_gait(side="right")
    # pt.add_toe_off_line(toe_off)
    # pt.add_reference("knee_r",walking_speed="L")
    # pt.end_plot()
    
    # pt.start_plot()
    # pt.add_data(k0.df_mean_joint_right_int,k0.df_std_joint_right_int, "hip_r", label_name= "DDPG k = 0.0" )
    # pt.add_data(k1.df_mean_joint_left_int,k1.df_std_joint_left_int, "hip_l", label_name= "DDPG k = 0.1" )
    # pt.add_reference("hip_r",walking_speed="L")
    # pt.end_plot(title = "Hip flexion", y_min=-50, y_max=75)
       
    
   
    
    """
    

    # Plot
    plot_int_data(df_mean_right_int, df_std_right_int, ["knee_r"], title = "Right leg", label_name="DDPG k=0."+str(k), 
                  walking_speed="L", x_label = "% gait cycle", y_label = "Knee flexion [deg]", 
                  y_min = -20, y_max =120, save_fig=True, save_name="DDPG_KneeR0"+str(k))
    plot_int_data(df_mean_right_int, df_std_right_int, ["hip_r"], title = "Right leg", label_name="DDPG k=0."+str(k), 
                  walking_speed="L", x_label = "% gait cycle", y_label = "Hip flexion [deg]", 
                  y_min = -50, y_max =70, save_fig=True, save_name="DDPG_HipR0"+str(k))
    plot_int_data(df_mean_right_int, df_std_right_int, ["ankle_r"], title = "Right leg", label_name="DDPG k=0."+str(k), 
                  walking_speed="L", x_label = "% gait cycle", y_label = "Ankle flexion [deg]", 
                  y_min = -60, y_max =60, save_fig=True, save_name="DDPG_AnkleR0"+str(k))
    
    plot_int_data(df_mean_left_int, df_std_left_int, ["knee_l"], title = "Left leg", label_name="DDPG k=0."+str(k), 
                  walking_speed="L", x_label = "% gait cycle", y_label = "Knee flexion [deg]", 
                  y_min = -20, y_max =120, save_fig=True, save_name="DDPG_KneeL0"+str(k))
    plot_int_data(df_mean_left_int, df_std_left_int, ["hip_l"], title = "Left leg", label_name="DDPG k=0."+str(k), 
                  walking_speed="L", x_label = "% gait cycle", y_label = "Hip flexion [deg]", 
                  y_min = -50, y_max =70, save_fig=True, save_name="DDPG_HipL0"+str(k))
    plot_int_data(df_mean_left_int, df_std_left_int, ["ankle_l"], title = "Left leg", label_name="DDPG k=0."+str(k), 
                  walking_speed="L", x_label = "% gait cycle", y_label = "Ankle flexion [deg]", 
                  y_min = -60, y_max =60, save_fig=True, save_name="DDPG_AnkleL0"+str(k))
    


    plot_int_data(df_mean_grf_int_right, df_std_grf_int_right, ["foot_r"], title = "Right leg", label_name="DDPG k=0."+str(k), 
                  walking_speed="L", x_label = "% gait cycle", y_label = "GRF vertical %BW", 
                  y_min = -0.2, y_max = 2.5, save_fig=True, save_name="DDPG_GRF_vertival_R0"+str(k), ref_type="grf")
    
    plot_int_data(df_mean_grf_int_right, df_std_grf_int_right, ["foot_r_pa"], title = "Right leg", label_name="DDPG k=0."+str(k), 
                  walking_speed="L", x_label = "% gait cycle", y_label = "GRF anterior %BW", 
                  y_min = -0.75, y_max =0.75, save_fig=True, save_name="DDPG_GRF_anerior_R0"+str(k), ref_type="grf")
    
    plot_int_data(df_mean_grf_int_left, df_std_grf_int_left, ["foot_l"], title = "Left leg", label_name="DDPG k=0."+str(k), 
                  walking_speed="L", x_label = "% gait cycle", y_label = "GRF vertical %BW", 
                  y_min = -0.2, y_max = 2.5, save_fig=True, save_name="DDPG_GRF_vertival_L0"+str(k), ref_type="grf")
    
    plot_int_data(df_mean_grf_int_left, df_std_grf_int_left, ["foot_l_pa"], title = "Left leg", label_name="DDPG k=0."+str(k), 
                  walking_speed="L", x_label = "% gait cycle", y_label = "GRF anterior %BW", 
                  y_min = -0.75, y_max =0.75, save_fig=True, save_name="DDPG_GRF_anterior_L0"+str(k), ref_type="grf")
    
    

    
    """

        

