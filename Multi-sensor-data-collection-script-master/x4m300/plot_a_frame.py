import numpy as np
import matplotlib.pyplot as plt
from time import sleep


import pymoduleconnector
from pymoduleconnector import DataType
from pymoduleconnector.ids import *

my_device = "COM11"


def reset(device_name):
    mc = pymoduleconnector.ModuleConnector(device_name)
    xep = mc.get_xep()
    xep.module_reset()
    mc.close()
    sleep(3)

def clear_buffer(mc):
    """Clears the frame buffer"""
    xep = mc.get_xep()
    while xep.peek_message_data_float(): #查看缓冲区中有多少数据
        xep.read_message_data_float() #将缓冲区中的数据逐一读出，每读出一个，就会将对应的从缓冲区中删除

def get_config_info(mc):
    xep = mc.get_xep()
    # print("The DAC iteration is", xep.x4driver_get_iterations(), "\n")
    # print(xep.x4driver_get_dac_min())
    print("The FPS is %.1f \n" % xep.x4driver_get_fps())  #返回配置的FPS
    print("the Frame area is from %.1f to %.1f:" % (xep.x4driver_get_frame_area().start, \
        xep.x4driver_get_frame_area().end)) # 返回配置的扫描区域距离
    # print("get decimation factor %d" % xep.get_decimation_factor())



if __name__ == "__main__":

    FPS = 10
    reset(my_device)
    mc = pymoduleconnector.ModuleConnector(my_device)

    xep = mc.get_xep()
    print(xep.get_system_info(2))  # 获得系统信息，通过输入的值的到对应的系统信息。
    print(dir(xep))
    
    """
    XTID_SSIC_ITEMNUMBER = 0x00 -> Returns the internal Novelda PCBA Item Number, including revision. 
    This is programmed in Flash during manufacturing XTID_SSIC_ORDERCODE = 0x01 -> Returns the PCBA / PCBA stack order code. 
    XTID_SSIC_FIRMWAREID = 0x02 -> Returns the installed Firmware ID. As viewed from the "highest" level of the software, "X4M300". 
    XTID_SSIC_VERSION = 0x03 -> Returns the installed Firmware Version. As viewed from the "highest" level of the software. 
    XTID_SSIC_BUILD = 0x04 -> Returns information of the SW Build installed on the device 
    XTID_SSIC_SERIALNUMBER = 0x06 -> Returns the PCBA serial number XTID_SSIC_VERSIONLIST = 0x07 -> Returns ID and version of all components. 
    Calls all components and compound a string. E.g. "X4M300:1.0.0.3;XEP:2.3.4.5;X4C51:1.0.0.0;DSP:1.1.1.1"
    """
    
    # 初始化x4驱动器
    xep.x4driver_init()
    
    # Set DAC range
    xep.x4driver_set_dac_min(900) #设置dac的最大最小值
    xep.x4driver_set_dac_max(1150)

    # Set integration
    xep.x4driver_set_iterations(16) #设置iteration
    xep.x4driver_set_pulses_per_step(26) #设置
    
    xep.x4driver_set_frame_area_offset(0.2)
    xep.x4driver_set_frame_area(1.0, 8.0)
    
    # Start streaming of data
    xep.x4driver_set_fps(FPS) #
    
    xep.set_sensor_mode(XTID_SM_RUN, 0)

    get_config_info(mc)

    #read a frame
    d = xep.read_message_data_float()
    print(dir(d))