# -*- coding: utf-8 -*-
import os
import sys
import time
import parser
import csv
from time import sleep
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import pymoduleconnector
from pymoduleconnector import ModuleConnector
from pymoduleconnector.ids import *


class X4m300DataCollectionRaw:
    def __init__(self, usb_port: str, FPS: int, enable_downconversion: bool, file_name: str = None):
        try:
            # 设置传感器连接的USB端口
            self.usb_port = usb_port
            self.file_name = file_name
            
            self.fps = FPS
            self.enable_downconversion = enable_downconversion
        except Exception as e:
            raise RuntimeError(f"传感器连接失败，请检查端口 {self.usb_port} 是否正确：{e}")

    def reset_module(self):
        mc = pymoduleconnector.ModuleConnector(self.usb_port)
        xep = mc.get_xep()
        xep.module_reset()
        mc.close()
        sleep(3)
        
    def initilization(self):
        # 获取x4m300雷达传感器的应用接口
        self.mc = ModuleConnector(self.usb_port)
        self.x4m300 = self.mc.get_x4m300()
        self.xep = self.mc.get_xep()
        
        # 打印传感器信息
        self.display_sys_info()
        
        # 清空之前缓冲的雷达帧
        # self._clear_frame_buffer()
        
        # 先停止传感器，并载入配置文件，设置要输出的内容
        self.x4m300.set_sensor_mode(XTID_SM_STOP, 0)
        self.x4m300.load_profile(XTS_ID_APP_PRESENCE_2)
        self.x4m300.set_sensor_mode(XTID_SM_STOP, 0)
        self.x4m300.set_sensor_mode(XTID_SM_MANUAL, 0)
    
    def _clear_frame_buffer(self):
        """清空之前缓冲的雷达帧
        """
        while self.xep.peek_message_data_float():
            self.xep.read_message_data_float()
        
        if self.xep.peek_message_data_float() == 0:
            print("-" * 40)
            print("旧数据帧已清空")

     
    def display_sys_info(self):
        """打印传感器信息
        """
        print("-" * 40)
        print("FirmWareID =", self.xep.get_system_info(2))
        print("Version =", self.xep.get_system_info(3))
        print("Build =", self.xep.get_system_info(4))
        print("VersionList =", self.xep.get_system_info(7))
        
    def display_radar_params(self):
        """打印传感器设置的参数
        """
        print("-" * 40)
        print(f"iterations = {self.xep.x4driver_get_iterations()}" )
        print(f"pulses_per_step = {self.xep.x4driver_get_pulses_per_step()}")
        print(f"dac_min = {self.xep.x4driver_get_dac_min()}, dac_max = {self.xep.x4driver_get_dac_max()}")
        print(f"frame_area_offset = {self.xep.x4driver_get_frame_area_offset()}")
        print(f"frame_area_start: {self.xep.x4driver_get_frame_area().start}, frame_area_end: {self.xep.x4driver_get_frame_area().end}")
        print(f"num_of_range_bins: {self.xep.x4driver_get_frame_bin_count()}")
        print(f"TX center frequency: {self.xep.x4driver_get_tx_center_frequency()}")
        print(f"PRF divider: {self.xep.x4driver_get_prf_div()}")
        
        # if self.enable_downconversion:
        #     print(f"num of range_bins after downconversion: {self.xep.x4driver_get_frame_bin_count()/self.xep.x4driver_get_prf_div()}")
            
            
    def set_radar_param(self, iteration: int = 16, pulses_per_step: int = 300, dac_min: int = 949, dac_max: int = 1100, frame_area_offset: float = 0.18, 
                         frame_area_start: float = 0.4, frame_area_end: float = 9.8, tx_center_freq: float = 3, prf_divider: int = 16):
        """设置雷达基本参数

        Args:
            iteration (int, optional): 设置形成一个frame要进行多少次完整的DAC sweep. Defaults to 16.
            pulses_per_step (int, optional): 设置每一个DAC sweep的每一个setp要进行多少次扫描. Defaults to 300.
            dac_min (int, optional): 设置DAC扫描最小阈值. Defaults to 949.
            dac_max (int, optional): 设置DAC扫描最小阈值. Defaults to 1100.
            frame_area_offset (float, optional): 设置检测范围的偏置. Defaults to 0.18.
            frame_area_start (float, optional): 设置最小检测距离. Defaults to 0.4.
            frame_area_end (float, optional): 设置最大检测距离. Defaults to 9.8.
            tx_center_freq (float, optional): 设置发射脉冲频率，3 对应的中心频率为:7.29GHz, 4对应的中心频率为8.748GHz. Defaults to 3.
            prf_divider (int, optional): 设置雷达脉冲的PRF(pulse_repetition_frequency), prf = 243MHz / prf_divider. Defaults to 16.
        """
        
        self.xep.x4driver_init()
        
        if self.enable_downconversion:
            self.xep.x4driver_set_downconversion(1)
        else:
            self.xep.x4driver_set_downconversion(0)


        self.xep.x4driver_set_iterations(iteration)
        self.xep.x4driver_set_pulses_per_step(pulses_per_step)
        self.xep.x4driver_set_dac_min(dac_min)
        self.xep.x4driver_set_dac_max(dac_max)
        self.xep.x4driver_set_tx_center_frequency(tx_center_freq)
        self.xep.x4driver_set_prf_div(prf_divider)

        # Set frame area offset
        self.xep.x4driver_set_frame_area_offset(frame_area_offset)

        # Set frame area
        self.xep.x4driver_set_frame_area(frame_area_start, frame_area_end)
    
        self.display_radar_params()
        
    def read_one_radar_frame(self):
        data = self.xep.read_message_data_float().data
        data_length = (len(data))
        if self.enable_downconversion:
            i_vec = np.array(data[:data_length//2])
            q_vec = np.array(data[data_length//2:])
            iq_vec = i_vec + 1j*q_vec

            ph_ampli = abs(iq_vec)                       #振幅
            ph_phase = np.arctan2(q_vec, i_vec)          #相位

            return ph_ampli, ph_phase
        
        return np.array(data)
        
    def start_radar(self):
        self.xep.x4driver_set_fps(self.fps)
        
    def stop_radar(self):
        self.xep.x4driver_set_fps(0)
        
    
    def plot_radar_frame(self):
        if self.enable_downconversion:
            bin_length = 8 * 1.5e8 / 23.328e9 
            amplitude, phase = self.read_one_radar_frame()
            ax_x = np.arange((self.xep.x4driver_get_frame_area().start - 1e-5), (self.xep.x4driver_get_frame_area().end - 1e-5 + bin_length), bin_length)
        else:
            bin_length = 1.5e8 / 23.328e9
            strength = self.read_one_radar_frame()
            ax_x = np.arange((self.xep.x4driver_get_frame_area().start - 1e-5), (self.xep.x4driver_get_frame_area().end - 1e-5), bin_length)
            
         
        
        fig = plt.figure()
        if self.enable_downconversion:
            fig.suptitle("radar frame downconversion") 
            ax1 = fig.add_subplot(2,1,1)
            ax1.set_title("amplitude")
            ax2 = fig.add_subplot(2,1,2)
            ax2.set_title("phase")


            ax1.set_ylim(0,0.02) #keep graph in frame (FIT TO YOUR DATA)
            # ax2.set_xlim(0,100)
            
            line1, = ax1.plot(ax_x, amplitude)
            line2, = ax2.plot(ax_x, phase)
            
        else:
            fig.suptitle("raw radar frame") 
            ax1 = fig.add_subplot(1,1,1)

            ax1.set_ylim(-0.02,0.02) #keep graph in frame (FIT TO YOUR DATA)
            # ax2.set_xlim(0,100)
            
            line1, = ax1.plot(ax_x, strength)
            


            self._clear_frame_buffer()
            
        fig.tight_layout()
        
        def animate(i):
            if self.enable_downconversion:
                amplitude, phase = self.read_one_radar_frame()
                line1.set_ydata(amplitude)
                line2.set_ydata(phase)
                return line1, line2,
            else:
                strength = self.read_one_radar_frame()
                line1.set_ydata(strength)
                return line1
        
        ani = FuncAnimation(fig, animate, interval=1000 / self.fps)
        plt.show()
                
        
        
        
        
        
        
        
            
            


if __name__ == "__main__":
    try:
        reader = X4m300DataCollectionRaw(usb_port="COM11", FPS=17, enable_downconversion=False)
        reader.reset_module()
        reader.initilization()
        reader.set_radar_param(64, 87, 949, 1100, 0.18, 0.5, 7.0)
        reader.start_radar()
        reader.plot_radar_frame()
    finally:
        reader.stop_radar()
    # reader.plot_radar_frame()
    
    

        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        